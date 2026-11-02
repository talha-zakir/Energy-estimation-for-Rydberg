"""Run reproducible MNIST comparisons without contacting quantum hardware."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

from .data import load_mnist, split_indices
from .quantum import Physics, RydbergCircuit, encode_angles, product_states
from .readout import fit_readout, projected_loss


def scores_metrics(scores, y):
    predicted = scores.argmax(axis=1)
    correct, count = int(np.sum(predicted == y)), len(y)
    # Wilson interval describes binomial sampling uncertainty conditional on this run.
    z = 1.96
    p = correct / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * np.sqrt(p*(1-p)/count + z*z/(4*count**2)) / (1+z*z/count)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "correct": correct, "n": count, "accuracy_wilson95": [center-half, center+half],
            "confusion_matrix": confusion_matrix(y, predicted, labels=np.arange(scores.shape[1])).tolist()}


def select_readout(train, val, y_train, y_val, alphas):
    """Choose alpha using validation MSE, never test labels."""
    candidates = []
    for alpha in alphas:
        w, b = fit_readout(train, y_train, alpha)
        mse = np.mean((val @ w + b - y_val)**2)
        candidates.append((float(mse), alpha, w, b))
    _, alpha, w, b = min(candidates, key=lambda c: c[0])
    return alpha, w, b


def train_circuit(circuit, train_states, val_states, y_train, y_val, initial, alpha, maxiter):
    history = []
    best = {"mse": np.inf, "parameters": initial.copy(), "iteration": 0}
    evaluations = 0

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        output, tape = circuit.forward(train_states, parameters, tape=True)
        loss, dfeatures, _, _ = projected_loss(circuit.features(output), y_train, alpha)
        return loss, circuit.backward(output, tape, dfeatures)

    def checkpoint(parameters):
        train_features = circuit.features(circuit.forward(train_states, parameters))
        val_features = circuit.features(circuit.forward(val_states, parameters))
        loss, _, w, b = projected_loss(train_features, y_train, alpha)
        val_scores = val_features @ w + b
        val_mse = float(np.mean((val_scores - y_val)**2))
        record = {"iteration": len(history), "train_objective": loss,
                  "val_mse": val_mse,
                  "val_accuracy": float(np.mean(val_scores.argmax(axis=1) == y_val.argmax(axis=1)))}
        history.append(record)
        if val_mse < best["mse"]:
            best.update(mse=val_mse, parameters=parameters.copy(), iteration=record["iteration"])
        if record["iteration"] % 5 == 0:
            print(f"Iteration {record['iteration']}: loss={loss:.5f}, "
                  f"val_accuracy={record['val_accuracy']:.3f}", flush=True)

    checkpoint(initial)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      callback=checkpoint, options={"maxiter": maxiter, "ftol": 1e-9})
    return best["parameters"], history, {
        "method": "L-BFGS-B with exact adjoint / variable projection",
        "objective_evaluations": evaluations, "iterations": int(result.nit),
        "converged": bool(result.success), "message": str(result.message),
        "checkpoint_selection": "minimum validation MSE",
        "selected_iteration": best["iteration"], "alpha": float(alpha)}


def run(config, output, cache):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    physics = Physics(**config["physics"])
    architecture = config.get("architecture", "digital_analog")
    digits = np.array(config["digits"])
    if len(set(digits)) != len(digits) or len(digits) < 2 or np.any((digits < 0) | (digits > 9)):
        raise ValueError("digits must contain 2..10 unique MNIST labels")
    if config["maxiter"] < 1 or config["shots"] < 0:
        raise ValueError("maxiter must be positive; shots must be nonnegative")
    rng = np.random.default_rng(config["seed"])
    images, labels, test_images, test_labels = load_mnist(cache)
    indices = split_indices(labels, test_labels, digits, config["n_train"],
                            config["n_val"], config["n_test"], config["seed"])
    train_idx, val_idx, test_idx = indices
    pixels = [images[train_idx] / 255., images[val_idx] / 255., test_images[test_idx] / 255.]
    targets = [labels[train_idx], labels[val_idx], test_labels[test_idx]]
    mapped = [np.array([np.flatnonzero(digits == y)[0] for y in t]) for t in targets]
    onehot = [np.eye(len(digits))[y] for y in mapped]
    pca = PCA(n_components=2 * physics.n_qubits, svd_solver="full")
    components = [pca.fit_transform(pixels[0]), pca.transform(pixels[1]), pca.transform(pixels[2])]
    states = [product_states(encode_angles(x)) for x in components]
    circuit = RydbergCircuit(physics, architecture)
    prefix = "digital" if architecture == "digital" else "rydberg"
    frozen_name = f"frozen_{prefix}_linear"
    trained_name = f"trained_{prefix}_linear"
    initial = rng.uniform(-np.pi, np.pi, circuit.n_params)
    alphas = config["alpha_grid"]

    # All model/alpha/checkpoint selection uses only train and validation.
    frozen_trainval = [circuit.features(circuit.forward(x, initial)) for x in states[:2]]
    quantum_alpha, _, _ = select_readout(*frozen_trainval, *onehot[:2], alphas)
    selected, history, optimizer = train_circuit(circuit, *states[:2], *onehot[:2],
                                                initial, quantum_alpha, config["maxiter"])

    feature_sets = {}
    # Same compressed input budget for the classical comparison; scale on training.
    pca_std = components[0].std(axis=0)
    pca_std[pca_std == 0] = 1
    feature_sets["classical_pca_linear"] = [x / pca_std for x in components]
    feature_sets["encoding_only_linear"] = [circuit.features(s) for s in states]
    frozen_outputs = [circuit.forward(s, initial) for s in states]
    selected_outputs = [circuit.forward(s, selected) for s in states]
    feature_sets[frozen_name] = [circuit.features(s) for s in frozen_outputs]
    feature_sets[trained_name] = [circuit.features(s) for s in selected_outputs]
    if config["shots"]:
        shot_rng = np.random.default_rng(config["seed"] + 10000)
        feature_sets[trained_name + "_shots"] = [
            circuit.features(s, config["shots"], shot_rng) for s in selected_outputs]

    models = {}
    # Lock readouts before evaluating any model on the test labels.
    for name, features in feature_sets.items():
        if name == trained_name:
            alpha = quantum_alpha
            w, b = fit_readout(features[0], onehot[0], alpha)
        else:
            alpha, w, b = select_readout(*features[:2], *onehot[:2], alphas)
        models[name] = (alpha, w, b)

    metrics, predictions = {}, {}
    archive = {"parameters": selected, "initial_parameters": initial,
               "pca_components": pca.components_, "pca_mean": pca.mean_,
               "pca_std": pca_std, "digits": digits,
               "physics_json": json.dumps(asdict(physics)), "architecture": architecture}
    for name, features in feature_sets.items():
        alpha, w, b = models[name]
        metrics[name] = {"alpha": float(alpha), "feature_count": features[0].shape[1]}
        archive[name + "_weights"], archive[name + "_bias"] = w, b
        for split, f, y in zip(("train", "validation", "test"), features, mapped):
            scores = f @ w + b
            metrics[name][split] = scores_metrics(scores, y)
            predictions[name + "_" + split] = digits[scores.argmax(axis=1)]
        print(name, "test_accuracy=", metrics[name]["test"]["accuracy"], flush=True)

    norms = np.concatenate([np.sum(np.abs(s)**2, axis=1) for s in selected_outputs])
    report = {"status": "CPU simulation; not a hardware experiment or paper reproduction",
              "architecture": architecture, "circuit": circuit.describe(),
              "config": config, "physics": physics.describe() if architecture == "digital_analog" else None,
              "optimizer": optimizer,
              "models": metrics, "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
              "max_state_norm_error": float(np.max(np.abs(norms - 1))),
              "runtime_seconds": time.perf_counter() - start,
              "simulation_counters": {"forward_calls": circuit.forward_calls,
                "sample_circuit_evaluations": circuit.sample_circuit_evaluations,
                "note": "Simulator workload counts; adjoint backward passes excluded. Not QPU shots or energy."},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(output / "model.npz", **archive)
    np.savez_compressed(output / "splits_predictions.npz", train_idx=train_idx,
                        validation_idx=val_idx, test_idx=test_idx,
                        train_labels=targets[0], validation_labels=targets[1],
                        test_labels=targets[2], **predictions)
    plot_report(report, history, output)
    return report


def plot_report(report, history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report["models"])
    accuracies = [report["models"][n]["test"]["accuracy"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax[0].barh([n.replace("_", " ") for n in names], accuracies)
    ax[0].set(xlim=(0, 1), xlabel="Held-out test accuracy", title="MNIST CPU simulation")
    for i, a in enumerate(accuracies):
        ax[0].text(max(0, a - .02), i, f"{a:.1%}", va="center", ha="right", color="white")
    ax[1].plot([h["iteration"] for h in history], [h["train_objective"] for h in history], label="Train objective")
    ax[1].plot([h["iteration"] for h in history], [h["val_mse"] for h in history], label="Validation MSE")
    ax[1].set(xlabel="L-BFGS iteration", ylabel="Loss", title="Linear readout + trained rotations")
    ax[1].legend()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def predict_saved(model_path, images, model_name=None):
    """Predict exact-expectation labels from uint8-like (N,784) MNIST images."""
    if model_name is not None and model_name.endswith("_shots"):
        raise ValueError("Sampled inference needs an explicit shot budget/RNG; use the experiment runner")
    with np.load(model_path, allow_pickle=False) as saved:
        physics = Physics(**json.loads(str(saved["physics_json"])))
        # Archives from the initial analog experiments predate this selector.
        architecture = str(saved["architecture"]) if "architecture" in saved else "digital_analog"
        prefix = "digital" if architecture == "digital" else "rydberg"
        if model_name is None:
            model_name = f"trained_{prefix}_linear"
        if model_name not in ("classical_pca_linear", "encoding_only_linear",
                              f"frozen_{prefix}_linear", f"trained_{prefix}_linear"):
            raise ValueError(f"Model {model_name!r} is not an exact model for {architecture}")
        components = (np.asarray(images, dtype=float) / 255 - saved["pca_mean"]) @ saved["pca_components"].T
        if model_name == "classical_pca_linear":
            features = components / saved["pca_std"]
        else:
            circuit = RydbergCircuit(physics, architecture)
            states = product_states(encode_angles(components))
            if model_name != "encoding_only_linear":
                key = "initial_parameters" if model_name == f"frozen_{prefix}_linear" else "parameters"
                states = circuit.forward(states, saved[key])
            features = circuit.features(states)
        scores = features @ saved[model_name + "_weights"] + saved[model_name + "_bias"]
        return saved["digits"][scores.argmax(axis=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--cache", default="data/mnist")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--architecture", choices=("digital_analog", "digital"),
                        help="Override the config's circuit architecture")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.architecture is not None:
        config["architecture"] = args.architecture
    with threadpool_limits(limits=config.get("threads", 4)):
        run(config, args.output, args.cache)


if __name__ == "__main__":
    main()
