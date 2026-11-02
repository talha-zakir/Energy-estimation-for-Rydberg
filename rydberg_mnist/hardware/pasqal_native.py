"""Global-pulse variational classifier compiled/emulated with QoolQit/Pulser."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from pulser.backend import EmulationConfig, default_observables as obs
from pulser_simulation import QutipBackendV2
import qoolqit as qq
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from ..data import load_mnist, split_indices
from ..experiment import scores_metrics, select_readout
from ..readout import fit_readout, projected_loss
from .pasqal import load_target, require_supported, sequence_ledger


def make_sequence(device, config, features, parameters):
    require_supported(device, "native_analog")
    n, slots = config["n_qubits"], config["input_slots"]
    if len(features) != 2*slots or len(parameters) != 2*slots:
        raise ValueError("Expected two input values and two parameters per pulse slot")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(parameters)):
        raise ValueError("Pulse inputs and parameters must be finite")
    if not 2 <= n <= device.max_atom_num:
        raise ValueError("Native register exceeds target limits")
    channel = next(c for c in device.channels.values() if c.basis == "ground-rydberg" and c.addressing == "Global")
    rise, slot = config["rise_ns"], config["slot_duration_ns"]
    if rise <= 0 or slot <= 2*rise or rise%channel.clock_period or slot%channel.clock_period:
        raise ValueError("Pulse knots must use positive clock-aligned ramp and hold durations")
    fraction = config["amplitude_fraction_of_limit"]
    if not 0 < fraction < 1:
        raise ValueError("Amplitude fraction must lie strictly between zero and one")
    # Interaction-limited compilation fixes geometry/time across images and
    # parameter updates: nearest-neighbor J=1 and all controls stay below caps.
    j0 = device.interaction_coeff/device.min_atom_distance**6
    omega_cap = fraction*channel.max_amp/j0
    delta_cap = min(config["detuning_to_amplitude_ratio"]*omega_cap, .8*channel.max_abs_detuning/j0)
    if delta_cap <= 0:
        raise ValueError("Detuning range must be positive")
    amplitudes = omega_cap*(.5+.4*np.tanh(features[:slots]+parameters[:slots]))
    detunings = delta_cap*np.tanh(features[slots:]+parameters[slots:])
    durations, amp_values, det_values = [], [0.], [0.]
    for amplitude, detuning in zip(amplitudes, detunings):
        durations += [rise*.001*j0, (slot-2*rise)*.001*j0, rise*.001*j0]
        amp_values += [float(amplitude), float(amplitude), 0.]
        det_values += [float(detuning), float(detuning), 0.]
    coords = [(float(q-(n-1)/2), 0.) for q in range(n)]
    program = qq.QuantumProgram(qq.Register.from_coordinates(coords), qq.Drive(
        amplitude=qq.PiecewiseLinearWaveform(durations, amp_values),
        detuning=qq.PiecewiseLinearWaveform(durations, det_values)))
    program.compile_to(qq.Device(pulser_device=device), profile="max_energy")
    sequence = program.compiled_sequence
    actual = np.array(list(sequence.register.qubits.values()))
    expected = np.array(coords)*device.min_atom_distance
    if not np.allclose(actual, expected, atol=1e-6, rtol=0):
        raise RuntimeError("Compiler changed the intended fixed geometry")
    if sequence.get_duration() != slots*slot:
        raise RuntimeError("Compiler changed the intended fixed instruction duration")
    return sequence


def moments_to_features(occupation, correlation):
    occupation, correlation = np.asarray(occupation), np.asarray(correlation)
    n = len(occupation)
    return np.array([*(1-2*occupation), *[1-2*occupation[i]-2*occupation[j]+4*correlation[i, j]
                                       for i in range(n) for j in range(i+1, n)]])


def sampled_features(counts, n):
    total = sum(counts.values())
    if not total:
        raise ValueError("No measured bitstrings")
    matrix = np.array([[1-2*int(bit) for bit in key] for key in counts], dtype=float)
    probability = np.array(list(counts.values()))/total
    observables = np.column_stack([matrix]+[matrix[:, i]*matrix[:, j]
                                         for i in range(n) for j in range(i+1, n)])
    return probability@observables


def emulate_sequence(sequence, device, *, noisy=False, shots=0, seed=7, trajectories=1):
    if noisy and device.noise_model is None:
        raise ValueError("No supplied device noise model")
    observables = [obs.Occupation(one_state="r"), obs.CorrelationMatrix(one_state="r")]
    if shots:
        observables.append(obs.BitStrings(num_shots=shots, one_state="r"))
    np.random.seed(seed)
    config = EmulationConfig(observables=observables, with_modulation=True,
                             prefer_device_noise_model=noisy, n_trajectories=trajectories,
                             default_num_shots=shots or 1000)
    result = QutipBackendV2(sequence, config=config).run()
    features = moments_to_features(result.get_result("occupation", 1.), result.get_result("correlation_matrix", 1.))
    if not shots:
        return features, None
    counts = {str(k): int(v) for k, v in dict(result.get_result("bitstrings", 1.)).items()}
    return features, sampled_features(counts, len(sequence.register.qubit_ids))


class Evaluator:
    def __init__(self, device, config):
        self.device, self.config = device, config
        self.sequence_count = 0

    def features(self, data, parameters, noisy=False, shots=0):
        expected, sampled = [], []
        for index, features in enumerate(data):
            sequence = make_sequence(self.device, self.config, features, parameters)
            values, measured = emulate_sequence(sequence, self.device, noisy=noisy, shots=shots,
                                                seed=self.config["seed"]+10000+index,
                                                trajectories=self.config.get("noise_trajectories", 1))
            expected.append(values)
            if measured is not None:
                sampled.append(measured)
            self.sequence_count += 1
        return np.array(expected), np.array(sampled) if sampled else None


def train(evaluator, data, targets, parameters, alpha):
    config = evaluator.config
    rng = np.random.default_rng(config["seed"]+100)
    history, best = [], {"validation_mse": np.inf, "parameters": parameters.copy(), "iteration": 0}

    def loss(theta):
        features, _ = evaluator.features(data[0], theta)
        return projected_loss(features, targets[0], alpha)[0]

    def checkpoint(theta, iteration):
        train_features, _ = evaluator.features(data[0], theta)
        val_features, _ = evaluator.features(data[1], theta)
        objective, _, w, b = projected_loss(train_features, targets[0], alpha)
        mse = float(np.mean((val_features@w+b-targets[1])**2))
        accuracy = float(np.mean((val_features@w+b).argmax(1)==targets[1].argmax(1)))
        history.append({"iteration": iteration, "train_objective": objective,
                        "validation_mse": mse, "validation_accuracy": accuracy})
        if mse < best["validation_mse"]:
            best.update(validation_mse=mse, parameters=theta.copy(), iteration=iteration)
        print(f"Native pulse checkpoint {iteration}: loss {objective:.5f}, validation accuracy {accuracy:.3f}", flush=True)

    checkpoint(parameters, 0)
    for k in range(config["iterations"]):
        delta = rng.choice([-1., 1.], size=len(parameters))
        c = config["spsa_perturbation"]/(k+1)**.101
        a = config["spsa_learning_rate"]/(k+5)**.602
        plus, minus = loss(parameters+c*delta), loss(parameters-c*delta)
        parameters = parameters-a*(plus-minus)/(2*c)*delta
        if (k+1)%config["checkpoint_every"] == 0 or k+1 == config["iterations"]:
            checkpoint(parameters, k+1)
    return best["parameters"], history, best["iteration"]


def run(config, output, cache="data/mnist"):
    if config.get("architecture") != "pasqal_native_analog":
        raise ValueError("This trainer implements only the separate Pasqal-native analog architecture")
    for key in ("iterations", "checkpoint_every", "spsa_learning_rate", "spsa_perturbation", "shots"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    target, device = load_target(config["target"])
    require_supported(device, "native_analog")
    images, labels, test_images, test_labels = load_mnist(cache)
    train_idx, val_idx, test_idx = split_indices(labels, test_labels, config["digits"],
             config["n_train"], config["n_val"], config["n_test"], config["seed"])
    pixels = [images[train_idx]/255., images[val_idx]/255., test_images[test_idx]/255.]
    digits = np.array(config["digits"])
    y = [labels[train_idx], labels[val_idx], test_labels[test_idx]]
    mapped = [np.array([int(np.flatnonzero(digits == label)[0]) for label in values]) for values in y]
    onehot = [np.eye(len(digits))[values] for values in mapped]
    pca = PCA(n_components=2*config["input_slots"], svd_solver="full")
    raw = [pca.fit_transform(pixels[0]), pca.transform(pixels[1]), pca.transform(pixels[2])]
    scale = raw[0].std(axis=0); scale[scale == 0] = 1.
    data = [np.clip(features/scale, -3, 3) for features in raw]
    evaluator = Evaluator(device, config)
    initial = np.random.default_rng(config["seed"]).normal(0, .2, 2*config["input_slots"])
    frozen_train, _ = evaluator.features(data[0], initial)
    frozen_val, _ = evaluator.features(data[1], initial)
    alpha, frozen_w, frozen_b = select_readout(frozen_train, frozen_val, *onehot[:2], config["alpha_grid"])
    parameters, history, selected_iteration = train(evaluator, data, onehot, initial.copy(), alpha)
    trained_train, _ = evaluator.features(data[0], parameters)
    w, b = fit_readout(trained_train, onehot[0], alpha)
    classical_alpha, classical_w, classical_b = select_readout(*data[:2], *onehot[:2], config["alpha_grid"])
    # Freeze model selection and all linear heads before test scoring.
    frozen_test, _ = evaluator.features(data[2], initial)
    clean_test, _ = evaluator.features(data[2], parameters)
    noise_test, sampled_test = evaluator.features(data[2], parameters, noisy=True, shots=config["shots"])
    models = {"classical_pca_linear": (data[2]@classical_w+classical_b, classical_alpha),
              "frozen_native_pulses_linear": (frozen_test@frozen_w+frozen_b, alpha),
              "trained_native_pulses_linear": (clean_test@w+b, alpha),
              "trained_native_pulses_device_noise": (noise_test@w+b, alpha),
              "trained_native_pulses_device_noise_shots": (sampled_test@w+b, alpha)}
    report = {"status": "Pasqal-native global analog classifier; local simulation only",
              "scope": "Separate architecture from the preserved digital/digital-analog VQCs; "
                       "small single-seed binary-MNIST validation, not a matched accuracy/energy comparison",
              "config": config, "target": target, "parameter_count": len(parameters),
              "training_backend": "QoolQit compilation + Pulser QutipBackendV2 with finite bandwidth, no training noise",
              "training_method": "SPSA with analytically fitted linear regression readout",
              "selected_iteration": selected_iteration, "optimizer_converged": False,
              "optimizer_note": "Fixed SPSA iteration budget; no convergence claim",
              "device_noise_protocol": "Supplied FRESNEL snapshot noise; quantum parameters/readout frozen before noisy inference. "
                     "Occupation/correlation scores precede detector errors; bitstring scores include SDK SPAM.",
              "models": {name: {"alpha": float(a), "test": scores_metrics(scores, mapped[2])}
                          for name, (scores, a) in models.items()},
              "compiled_emulated_sequences": evaluator.sequence_count,
              "runtime_seconds": time.perf_counter()-started,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("qoolqit", "pulser-core", "pulser-simulation", "qutip", "numpy", "scipy")}}
    sequence = make_sequence(device, config, data[2][0], parameters)
    energy, _ = sequence_ledger(sequence, target, modulation=True, shots=config["shots"])
    report["first_test_input_energy"] = energy
    (output/"compiled_example.json").write_text(sequence.to_abstract_repr()+"\n")
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"history.json").write_text(json.dumps(history, indent=2)+"\n")
    np.savez_compressed(output/"model.npz", parameters=parameters, initial_parameters=initial,
               pca_components=pca.components_, pca_mean=pca.mean_, pca_scale=scale,
               weights=w, bias=b, frozen_weights=frozen_w, frozen_bias=frozen_b,
               classical_weights=classical_w, classical_bias=classical_b, digits=digits,
               config_json=json.dumps(config), target_json=json.dumps(target))
    np.savez_compressed(output/"splits_predictions.npz", train_idx=train_idx, validation_idx=val_idx,
               test_idx=test_idx, test_labels=y[2], **{name: digits[scores.argmax(1)] for name, (scores, _) in models.items()})
    print(json.dumps({name: item["test"]["accuracy"] for name, item in report["models"].items()}, indent=2))
    return report


def predict_saved(path, images, target_path=None):
    with np.load(path, allow_pickle=False) as archive:
        config = json.loads(str(archive["config_json"]))
        target, device = load_target(target_path or config["target"])
        if target["device_sha256"] != json.loads(str(archive["target_json"]))["device_sha256"]:
            raise ValueError("Inference target differs from the trained device snapshot")
        features = (np.asarray(images)/255.-archive["pca_mean"]) @ archive["pca_components"].T
        features = np.clip(features/archive["pca_scale"], -3, 3)
        measured, _ = Evaluator(device, config).features(features, archive["parameters"])
        return archive["digits"][(measured@archive["weights"]+archive["bias"]).argmax(1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pasqal_native_mnist.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/mnist")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=config.get("threads", 4)):
        run(config, args.output, args.cache)


if __name__ == "__main__":
    main()
