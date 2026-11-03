"""Fixed Rydberg reservoir with destructive prefix readouts and a linear head."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from pulser.backend import EmulationConfig, default_observables as obs
from pulser_simulation import QutipBackendV2
from threadpoolctl import threadpool_limits

from ..data import load_mnist, split_indices
from ..experiment import scores_metrics, select_readout
from .literature_cost import positive_integer
from .pasqal import load_target, sequence_ledger
from .pasqal_native import make_sequence, moments_to_features, sampled_features


def allocate_shots(total, nodes):
    """Divide a total hardware-shot budget between destructive readouts."""
    positive_integer(total, "total shots")
    positive_integer(nodes, "readout nodes")
    if total < nodes:
        raise ValueError("Each readout needs at least one shot")
    quotient, remainder = divmod(total, nodes)
    return [quotient+int(i < remainder) for i in range(nodes)]


def prefix_values(values, slots, prefix):
    """Keep early amplitude AND early detuning inputs; never see future slots."""
    positive_integer(prefix, "prefix")
    values = np.asarray(values)
    if values.shape != (2*slots,) or not np.all(np.isfinite(values)) or prefix > slots:
        raise ValueError("Invalid input/parameter shape or pulse prefix")
    return np.r_[values[:prefix], values[slots:slots+prefix]]


def reservoir_sequence(device, config, features, biases, prefix):
    slots = config["input_slots"]
    return make_sequence(device, {**config, "input_slots": prefix},
                         prefix_values(features, slots, prefix), prefix_values(biases, slots, prefix))


def emulate_counts(sequence, device, *, noisy=False, shots=0, seed=0, trajectories=1):
    """One SDK execution supplies noisy expectations and SPAM-affected counts."""
    if noisy and device.noise_model is None:
        raise ValueError("No device noise model; refusing noiseless fallback")
    observables = [obs.Occupation(one_state="r"), obs.CorrelationMatrix(one_state="r")]
    if shots:
        observables.append(obs.BitStrings(num_shots=shots, one_state="r"))
    np.random.seed(seed)
    emulation = EmulationConfig(observables=observables, with_modulation=True,
        prefer_device_noise_model=noisy, n_trajectories=trajectories if noisy else 1,
        default_num_shots=shots or 1000)
    result = QutipBackendV2(sequence, config=emulation).run()
    features = moments_to_features(result.get_result("occupation", 1.), result.get_result("correlation_matrix", 1.))
    counts = None if not shots else {str(k): int(v) for k, v in dict(result.get_result("bitstrings", 1.)).items()}
    if counts is not None and sum(counts.values()) != shots:
        raise RuntimeError("SDK returned a different number of shots than requested")
    return features, counts


def subset_counts(counts, shots, rng):
    """Random subset of a larger physical-shot sample, keeping site correlations."""
    positive_integer(shots, "shots")
    if not counts or any(v < 0 or not isinstance(v, int) for v in counts.values()):
        raise ValueError("Invalid bitstring counts")
    if shots > sum(counts.values()):
        raise ValueError("Cannot create extra measured shots by resampling")
    keys = sorted(counts)
    subset = rng.multivariate_hypergeometric([counts[k] for k in keys], shots)
    return {key: int(count) for key, count in zip(keys, subset) if count}


def transform(images, model):
    values = (np.asarray(images)/255.-model["pca_mean"])@model["pca_components"].T
    return np.clip(values/model["pca_scale"], -3, 3)


def load_reference(directory, cache):
    """Reuse and verify the exact VQC split, training-only PCA and device."""
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text())
    with np.load(directory/"model.npz", allow_pickle=False) as archive:
        model = {key: archive[key].copy() for key in archive.files}
    config, saved_target = json.loads(str(model["config_json"])), json.loads(str(model["target_json"]))
    if config != report["config"] or config["architecture"] != "pasqal_native_analog":
        raise ValueError("Reference model/report configuration mismatch")
    target, device = load_target(config["target"])
    if target["device_sha256"] != saved_target["device_sha256"] or saved_target != report["target"]:
        raise ValueError("Reference model/report/device snapshot mismatch")
    with np.load(directory/"splits_predictions.npz", allow_pickle=False) as archive:
        saved_split = {key: archive[key].copy() for key in archive.files}
    images, labels, test_images, test_labels = load_mnist(cache)
    expected = split_indices(labels, test_labels, config["digits"], config["n_train"],
                             config["n_val"], config["n_test"], config["seed"])
    indices = [saved_split[k] for k in ("train_idx", "validation_idx", "test_idx")]
    for actual, reference in zip(indices, expected):
        if not np.array_equal(actual, reference):
            raise ValueError("Reference split differs from its declared deterministic split")
    if np.intersect1d(indices[0], indices[1]).size:
        raise ValueError("Training and validation overlap")
    y = [labels[indices[0]], labels[indices[1]], test_labels[indices[2]]]
    if not np.array_equal(y[2], saved_split["test_labels"]):
        raise ValueError("Reference test labels mismatch")
    digits = np.asarray(config["digits"])
    mapped = [np.array([np.flatnonzero(digits == v)[0] for v in values]) for values in y]
    data = [transform(images[indices[0]], model), transform(images[indices[1]], model),
            transform(test_images[indices[2]], model)]
    return report, model, target, device, indices, y, mapped, data


def validate_config(config, reference):
    if config.get("architecture") != "pasqal_qrc":
        raise ValueError("Expected the pasqal_qrc architecture")
    slots = config["readout_slots"]
    if not isinstance(slots, list) or not slots:
        raise ValueError("readout_slots must be a nonempty ordered list")
    for slot in slots:
        positive_integer(slot, "readout slot")
    if slots != sorted(set(slots)) or slots[-1] != reference["input_slots"]:
        raise ValueError("Readout slots must be unique, ordered, and include the final input slot")
    for key in ("shots_per_readout", "noise_trajectories", "sampling_seed"):
        positive_integer(config[key], key)
    allocate_shots(config["total_shots_per_prediction"], len(slots))


class ReservoirEvaluator:
    def __init__(self, device, native_config, biases, readout_slots, trajectories, target):
        self.device, self.native_config = device, native_config
        self.biases = np.asarray(biases).copy()
        self.biases.setflags(write=False)
        self.readout_slots, self.trajectories = readout_slots, trajectories
        self.target = target
        self.sequence_count = 0
        self.control_rows = []

    def features(self, data, indices, *, split, noisy=False, shots=0, seed=0, example_dir=None):
        expected, counts = [], []
        for i, (values, index) in enumerate(zip(data, indices)):
            nodes, measured = [], []
            for node, prefix in enumerate(self.readout_slots):
                sequence = reservoir_sequence(self.device, self.native_config, values, self.biases, prefix)
                features, sample = emulate_counts(sequence, self.device, noisy=noisy, shots=shots,
                    seed=seed+1009*i+node, trajectories=self.trajectories)
                nodes.append(features)
                measured.append(sample)
                self.sequence_count += 1
                if not noisy:
                    energy, _ = sequence_ledger(sequence, self.target, modulation=True, shots=1)
                    self.control_rows.append({"split": split, "mnist_index": int(index), "prefix": prefix,
                        "scheduled_control_us": energy["scheduled_control_window_us_per_shot"],
                        "omega_squared_integral_rad2_per_us": energy["channels"]["rydberg_global"]["omega_squared_integral_rad2_per_us"]})
                    if example_dir is not None and i == 0:
                        (Path(example_dir)/f"qrc_prefix{prefix}.json").write_text(sequence.to_abstract_repr()+"\n")
            expected.append(np.concatenate(nodes)); counts.append(measured)
            if (i+1)%20 == 0 or i+1 == len(data):
                print(f"QRC {split} {'noisy' if noisy else 'coherent'}: {i+1}/{len(data)} images", flush=True)
        return np.asarray(expected), counts


def features_from_counts(measured, allocation, n, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([np.concatenate([sampled_features(subset_counts(counts, shots, rng), n)
                        for counts, shots in zip(nodes, allocation)]) for nodes in measured])


def run(config, output, cache="data/mnist"):
    started = time.perf_counter()
    reference, model, target, device, indices, labels, mapped, data = load_reference(config["reference_run"], cache)
    native = reference["config"]
    validate_config(config, native)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    example_dir = output/"compiled_examples"; example_dir.mkdir()
    slots, n = config["readout_slots"], native["n_qubits"]
    per_node = allocate_shots(config["total_shots_per_prediction"], len(slots))
    endpoint_shots = config["total_shots_per_prediction"]
    # One larger shot sample supports both budgets; the smaller uses a random
    # subset without replacement. Prefix samples never share a physical shot.
    sample_size = max(config["shots_per_readout"], max(per_node), endpoint_shots)
    evaluator = ReservoirEvaluator(device, native, model["initial_parameters"], slots, config["noise_trajectories"], target)
    train_features, _ = evaluator.features(data[0], indices[0], split="train", seed=config["sampling_seed"])
    val_features, _ = evaluator.features(data[1], indices[1], split="validation", seed=config["sampling_seed"]+1)
    onehot = [np.eye(len(native["digits"]))[values] for values in mapped]
    alpha, weights, bias = select_readout(train_features, val_features, *onehot[:2], native["alpha_grid"])
    width = n*(n+1)//2
    endpoint_alpha, endpoint_w, endpoint_b = select_readout(train_features[:, -width:], val_features[:, -width:],
                                                           *onehot[:2], native["alpha_grid"])
    # The single-node reservoir is exactly the pre-existing frozen VQC ablation.
    np.testing.assert_allclose(endpoint_w, model["frozen_weights"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(endpoint_b, model["frozen_bias"], atol=1e-7, rtol=1e-7)
    test_features, _ = evaluator.features(data[2], indices[2], split="test", seed=config["sampling_seed"]+2,
                                           example_dir=example_dir)
    noisy_features, measured = evaluator.features(data[2], indices[2], split="test", noisy=True,
        shots=sample_size, seed=config["sampling_seed"], example_dir=None)
    sampled_equal = features_from_counts(measured, per_node, n, config["sampling_seed"]+30000)
    sampled_per_node = features_from_counts(measured, [config["shots_per_readout"]]*len(slots), n,
                                            config["sampling_seed"]+40000)
    sampled_endpoint = features_from_counts([[nodes[-1]] for nodes in measured], [endpoint_shots], n,
                                            config["sampling_seed"]+50000)
    vqc_clean, vqc_noise, vqc_samples, vqc_controls = [], [], [], []
    for i, (values, index) in enumerate(zip(data[2], indices[2])):
        sequence = make_sequence(device, native, values, model["parameters"])
        clean, _ = emulate_counts(sequence, device)
        noise, counts = emulate_counts(sequence, device, noisy=True, shots=endpoint_shots,
            seed=config["sampling_seed"]+1009*i+len(slots)-1, trajectories=config["noise_trajectories"])
        vqc_clean.append(clean); vqc_noise.append(noise); vqc_samples.append(sampled_features(counts, n))
        ledger, _ = sequence_ledger(sequence, target, modulation=True, shots=1)
        vqc_controls.append({"split": "test", "mnist_index": int(index), "prefix": native["input_slots"],
            "scheduled_control_us": ledger["scheduled_control_window_us_per_shot"],
            "omega_squared_integral_rad2_per_us": ledger["channels"]["rydberg_global"]["omega_squared_integral_rad2_per_us"]})
        if i == 0:
            (example_dir/"vqc.json").write_text(sequence.to_abstract_repr()+"\n")
        if (i+1)%20 == 0 or i+1 == len(data[2]):
            print(f"Reference VQC reevaluation: {i+1}/{len(data[2])} test images", flush=True)
    tests = {
        "classical_pca_linear": {"coherent": data[2]@model["classical_weights"]+model["classical_bias"]},
        "vqc": {"coherent": np.asarray(vqc_clean)@model["weights"]+model["bias"],
                "device_noise_expectations": np.asarray(vqc_noise)@model["weights"]+model["bias"],
                "equal_total_shots": np.asarray(vqc_samples)@model["weights"]+model["bias"]},
        "qrc_endpoint": {"coherent": test_features[:, -width:]@endpoint_w+endpoint_b,
                "device_noise_expectations": noisy_features[:, -width:]@endpoint_w+endpoint_b,
                "equal_total_shots": sampled_endpoint@endpoint_w+endpoint_b},
        "qrc_temporal": {"coherent": test_features@weights+bias,
                "device_noise_expectations": noisy_features@weights+bias,
                "equal_total_shots": sampled_equal@weights+bias,
                "equal_shots_per_node": sampled_per_node@weights+bias}}
    predictions = {name+"__"+condition: np.asarray(native["digits"])[scores.argmax(1)]
                   for name, conditions in tests.items() for condition, scores in conditions.items()}
    # Fresh coherent evaluation must reproduce the archived native task scores.
    archive_predictions = np.load(Path(config["reference_run"])/"splits_predictions.npz", allow_pickle=False)
    for fresh, saved in (("vqc__coherent", "trained_native_pulses_linear"),
                         ("qrc_endpoint__coherent", "frozen_native_pulses_linear"),
                         ("classical_pca_linear__coherent", "classical_pca_linear")):
        np.testing.assert_array_equal(predictions[fresh], archive_predictions[saved])
    archive_predictions.close()
    model_info = {
        "classical_pca_linear": {"feature_count": data[0].shape[1], "quantum_parameters_trained": 0},
        "vqc": {"feature_count": width, "quantum_parameters_trained": len(model["parameters"]),
                "alpha": reference["models"]["trained_native_pulses_linear"]["alpha"]},
        "qrc_endpoint": {"feature_count": width, "quantum_parameters_trained": 0, "alpha": float(endpoint_alpha)},
        "qrc_temporal": {"feature_count": width*len(slots), "quantum_parameters_trained": 0, "alpha": float(alpha)}}
    report = {"status": "Local QoolQit/Pulser QRC versus saved Pasqal-native VQC; no hardware submission",
        "config": config, "native_config": native, "target": target,
        "reference_provenance": {"directory": config["reference_run"],
            **{name+"_sha256": hashlib.sha256((Path(config["reference_run"])/name).read_bytes()).hexdigest()
               for name in ("model.npz", "report.json", "splits_predictions.npz")}},
        "pairing": {"identical_split_indices": True, "identical_training_only_pca": True,
            "identical_target_geometry_and_encoding": True, "identical_noise_and_detector_model": True,
            "readouts_fitted_on_coherent_train_only": True, "all_model_selection_before_test_scoring": True,
            "coherent_vqc_and_endpoint_predictions_reproduce_archive": True,
            "reservoir_biases_equal_archived_vqc_initialization": True,
            "matching_limit": "Temporal QRC has more readout features and prefix executions; ansatz/optimization budgets differ"},
        "protocol": {"reset_between_images_and_prefix_runs": True,
            "prefix_readouts_are_separate_destructive_experiments": True,
            "shared_prefix_hardware_shots": False, "equal_total_shots_allocation": per_node,
            "equal_shots_per_node_allocation": [config["shots_per_readout"]]*len(slots),
            "vqc_and_endpoint_shots_per_image": endpoint_shots,
            "quantum_training_steps_qrc": 0, "readout_selection": "validation MSE over the archived alpha grid",
            "noise_expectations": "Include device dynamics; precede detector errors",
            "finite_shots": "Include SDK detector/SPAM; small-budget counts are random subsets of the large sample",
            "training_hardware_budget": "Hypothetical finite-shot replay; exact-expectation training ran on CPU",
            "noise_seed": config["sampling_seed"], "noise_trajectories": config["noise_trajectories"]},
        "models": {name: {**model_info[name], "conditions": {condition: scores_metrics(scores, mapped[2])
                    for condition, scores in conditions.items()}} for name, conditions in tests.items()},
        "simulator_sequence_executions": {"qrc": evaluator.sequence_count, "vqc_test_reevaluation": 2*len(data[2])},
        "runtime_seconds": time.perf_counter()-started,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("qoolqit", "pulser-core", "pulser-simulation", "qutip", "numpy", "scipy", "scikit-learn")}}
    np.savez_compressed(output/"model.npz", reservoir_biases=model["initial_parameters"],
        weights=weights, bias=bias, endpoint_weights=endpoint_w, endpoint_bias=endpoint_b,
        pca_components=model["pca_components"], pca_mean=model["pca_mean"], pca_scale=model["pca_scale"],
        digits=model["digits"], config_json=json.dumps(config), native_config_json=json.dumps(native),
        target_json=json.dumps(target))
    np.savez_compressed(output/"features.npz", train=train_features, validation=val_features, test=test_features,
        device_noise_expectations=noisy_features, equal_total_shots=sampled_equal,
        equal_shots_per_node=sampled_per_node, endpoint_shots=sampled_endpoint,
        vqc_coherent=vqc_clean, vqc_device_noise_expectations=vqc_noise, vqc_shots=vqc_samples)
    np.savez_compressed(output/"splits_predictions.npz", train_idx=indices[0], validation_idx=indices[1],
        test_idx=indices[2], train_labels=labels[0], validation_labels=labels[1], test_labels=labels[2], **predictions)
    (output/"measured_counts.json").write_text(json.dumps({"qrc_test": measured}, indent=2)+"\n")
    from .qrc_compare import finish_report
    finish_report(report, evaluator.control_rows, vqc_controls, predictions, labels[2], output)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({name: {k: v["accuracy"] for k, v in item["conditions"].items()}
                      for name, item in report["models"].items()}, indent=2), flush=True)
    return report


def predict_saved(path, images, variant="qrc_temporal", target_path=None):
    if variant not in ("qrc_temporal", "qrc_endpoint"):
        raise ValueError("Unknown QRC readout variant")
    with np.load(path, allow_pickle=False) as model:
        config = json.loads(str(model["config_json"]))
        native = json.loads(str(model["native_config_json"]))
        target, device = load_target(target_path or native["target"])
        if target["device_sha256"] != json.loads(str(model["target_json"]))["device_sha256"]:
            raise ValueError("Inference target differs from the trained device snapshot")
        slots = config["readout_slots"] if variant == "qrc_temporal" else [native["input_slots"]]
        evaluator = ReservoirEvaluator(device, native, model["reservoir_biases"], slots, 1, target)
        features, _ = evaluator.features(transform(images, model), np.arange(len(images)), split="inference")
        prefix = "endpoint_" if variant == "qrc_endpoint" else ""
        return model["digits"][(features@model[prefix+"weights"]+model[prefix+"bias"]).argmax(1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pasqal_qrc_mnist.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/mnist")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=config.get("threads", 4)):
        run(config, args.output, args.cache)


if __name__ == "__main__":
    main()
