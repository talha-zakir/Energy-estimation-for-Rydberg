"""Finite-shot, equal-readout QRC/VQC energy-to-validation-target experiments."""
import argparse
from dataclasses import dataclass
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from ..data import load_mnist, split_indices
from ..experiment import scores_metrics
from ..readout import fit_readout
from .calibrated_cost import budget, validate_calibration
from .finite_backend import FeatureEngine, PulserCountsBackend
from .target_profile import AnalogProfile


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    alpha: float

    def predict(self, features):
        return ((features-self.mean)/self.scale)@self.weights+self.bias

    @classmethod
    def fit(cls, train, validation, targets, validation_targets, alphas):
        """Identical regression/standardization and validation rule for both methods."""
        mean, scale = train.mean(0), train.std(0)
        scale[scale < 1e-12] = 1.
        x, v = (train-mean)/scale, (validation-mean)/scale
        candidates = []
        for alpha in alphas:
            if not np.isfinite(alpha) or alpha < 0:
                raise ValueError("Readout regularization must be finite and nonnegative")
            weights, bias = fit_readout(x, targets, alpha)
            candidates.append((float(np.mean((v@weights+bias-validation_targets)**2)),
                               float(alpha), weights, bias))
        _, alpha, weights, bias = min(candidates, key=lambda item: item[0])
        return cls(mean, scale, weights, bias, alpha)


def first_target_hit(checkpoints, target):
    """Selection sees validation only; unavailable targets remain censored."""
    if not 0 < target <= 1:
        raise ValueError("Accuracy target must lie in (0,1]")
    return next((c for c in checkpoints if c["validation_accuracy"] >= target), None)


def get_data(images, labels, test_images, test_labels, config, seed):
    indices = split_indices(labels, test_labels, config["digits"], config["n_train"],
                             config["n_val"], config["n_test"], seed)
    pixel = [images[indices[0]]/255., images[indices[1]]/255., test_images[indices[2]]/255.]
    pca = PCA(n_components=2*config["regime"]["input_slots"], svd_solver="full")
    raw = [pca.fit_transform(pixel[0]), pca.transform(pixel[1]), pca.transform(pixel[2])]
    scale = raw[0].std(0); scale[scale < 1e-12] = 1.
    features = [np.clip(values/scale, -3, 3) for values in raw]
    original = [labels[indices[0]], labels[indices[1]], test_labels[indices[2]]]
    mapped = [np.array([config["digits"].index(int(label)) for label in values]) for values in original]
    onehot = [np.eye(len(config["digits"]))[values] for values in mapped]
    return indices, features, mapped, onehot, pca, scale, original


def validate_config(config):
    for key in ("n_train", "n_val", "n_test", "shots_per_image", "noise_trajectories", "vqc_steps", "minibatch_size", "checkpoint_every"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not config["repeat_seeds"] or len(set(config["repeat_seeds"])) != len(config["repeat_seeds"]):
        raise ValueError("Use distinct nonempty repeat seeds")
    for seed in config["repeat_seeds"]:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("Invalid repeat seed")
    if config["minibatch_size"] > config["n_train"]:
        raise ValueError("Minibatch exceeds training set")
    for key in ("spsa_learning_rate", "spsa_perturbation", "parameter_bound"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not config["alpha_grid"] or any(not np.isfinite(alpha) or alpha < 0 for alpha in config["alpha_grid"]):
        raise ValueError("Supply a nonempty finite nonnegative alpha grid")
    if not config["accuracy_targets"] or any(not 0 < target <= 1 for target in config["accuracy_targets"]):
        raise ValueError("Invalid predeclared accuracy target")
    if not 0 <= config["test_accuracy_tolerance"] <= 1:
        raise ValueError("Invalid held-out accuracy tolerance")


def run(config, output, cache="data/mnist", backend_factory=None, profile_factory=None):
    validate_config(config)
    started = time.perf_counter()
    profile = (AnalogProfile.build(config["target"], config["regime"]) if profile_factory is None else
               profile_factory(config["target"], config["regime"]))
    calibration = json.loads(Path(config["calibration"]).read_text())
    validate_calibration(calibration, profile.target)
    literature = json.loads(Path(config["literature_config"]).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    for key in ("target", "calibration", "literature_config"):
        (output/(key+"_snapshot.json")).write_bytes(Path(config[key]).read_bytes())
    images, labels, test_images, test_labels = load_mnist(cache)
    report = {"status": "Finite-shot, noisy, common-readout comparison; calibration completeness reported explicitly",
        "config": config, "device_profile": profile.metadata, "target": profile.target,
        "calibration": calibration, "literature_configuration": literature,
        "input_sha256": {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                         for key in ("target", "calibration", "literature_config")},
        "protocol": {"quantum_features_identical_for_both_methods": True,
            "linear_head": "Training-only feature scaling + multi-output linear least squares / ridge; same alpha grid and validation MSE",
            "training_and_validation_use_finite_shot_device_noise_and_spam": True,
            "no_exact_expectation_training": True,
            "selection_uses_test_labels": False,
            "target_rule": "First checkpoint meeting predeclared validation accuracy; unreachable targets are censored",
            "hardware_training_recipe": "Alternating sampled linear-head fits and minibatch SPSA with a fixed head between checkpoints",
            "common_shots_per_image": config["shots_per_image"],
            "test_matching_rule": "Report both independent held-out scores and flag compatibility within the declared tolerance; never tune to test accuracy",
            "prior_test_inspection": "Exploratory sample; earlier work used some of the same official test images",
            "default_energy_calibration": "Unknown until traceable target-specific measurements are supplied",
            "hardware_jobs_submitted_by_default": False},
        "repeats": [], "versions": {name: importlib.metadata.version(name) for name in
            ("pulser-core", "pulser-simulation", "qutip", "numpy", "scipy", "scikit-learn")}}
    for repeat, seed in enumerate(config["repeat_seeds"]):
        folder = output/f"seed{seed}"; folder.mkdir()
        backend = (backend_factory(profile, seed) if backend_factory else
                   PulserCountsBackend(profile.device, config["noise_trajectories"]))
        engine = FeatureEngine(profile, backend, config["readout_slots"], config["shots_per_image"])
        indices, data, mapped, onehot, pca, scale, original = get_data(images, labels, test_images, test_labels, config, seed)
        rng = np.random.default_rng(seed)
        initial = rng.normal(0., .15, 2*config["regime"]["input_slots"])
        theta = initial.copy()
        checkpoints = {"qrc": [], "vqc": []}
        stored = {}
        def checkpoint(algorithm, parameters, iteration):
            features = engine.features(data[0], parameters, indices[0], algorithm=algorithm, stage="train",
                                       checkpoint=iteration, seed=seed*100000+iteration*10000+10)
            val = engine.features(data[1], parameters, indices[1], algorithm=algorithm, stage="validation",
                                  checkpoint=iteration, seed=seed*100000+iteration*10000+5000)
            head = LinearHead.fit(features, val, *onehot[:2], config["alpha_grid"])
            scores = head.predict(val)
            record = {"iteration": iteration, "validation_accuracy": float(np.mean(scores.argmax(1) == mapped[1])),
                "validation_mse": float(np.mean((scores-onehot[1])**2)), "alpha": head.alpha,
                "cumulative_requested_shots": sum(r["shots"] for r in engine.records if r["algorithm"] == algorithm),
                "cumulative_pulse_programs": sum(r["algorithm"] == algorithm for r in engine.records),
                "ledger_end": len(engine.records)}
            checkpoints[algorithm].append(record)
            stored[(algorithm, iteration)] = (parameters.copy(), head)
            print(f"seed {seed} {algorithm} step {iteration}: validation {record['validation_accuracy']:.1%}; "
                  f"requested shots {record['cumulative_requested_shots']:,}", flush=True)
            return head
        checkpoint("qrc", initial, 0)
        head = checkpoint("vqc", theta, 0)
        for step in range(1, config["vqc_steps"]+1):
            subset = rng.choice(config["n_train"], config["minibatch_size"], replace=False)
            direction = rng.choice([-1., 1.], len(theta))
            perturbation = config["spsa_perturbation"]/(step**.101)
            learning = config["spsa_learning_rate"]/((step+4)**.602)
            objectives = []
            for sign, offset in ((1, 0), (-1, 5000)):
                features = engine.features(data[0][subset], theta+sign*perturbation*direction, indices[0][subset],
                    algorithm="vqc", stage="spsa_objective", checkpoint=step,
                    seed=seed*100000+step*20000+10000000+offset)
                objectives.append(float(np.mean((head.predict(features)-onehot[0][subset])**2)))
            gradient = (objectives[0]-objectives[1])/(2*perturbation)*direction
            theta = np.clip(theta-learning*gradient, -config["parameter_bound"], config["parameter_bound"])
            if step%config["checkpoint_every"] == 0 or step == config["vqc_steps"]:
                head = checkpoint("vqc", theta, step)
                if config.get("early_stop_all_targets", False) and all(first_target_hit(checkpoints["vqc"], t)
                                                                        for t in config["accuracy_targets"]):
                    break
        # All target/checkpoint choices are frozen before any test evaluation.
        choices = {algorithm: {str(t): first_target_hit(points, t) for t in config["accuracy_targets"]}
                   for algorithm, points in checkpoints.items()}
        # A validation-selected fallback exposes observed performance even when
        # the requested target is unreachable; it is never relabeled a target hit.
        fallback = {algorithm: min(points, key=lambda point: point["validation_mse"])
                    for algorithm, points in checkpoints.items()}
        selected = {(algorithm, point["iteration"]) for algorithm, by_target in choices.items()
                    for point in by_target.values() if point is not None}
        selected.update((algorithm, point["iteration"]) for algorithm, point in fallback.items())
        test_results, saved_arrays, export = {}, {}, []
        predictions = {}
        for algorithm, iteration in sorted(selected):
            parameters, selected_head = stored[(algorithm, iteration)]
            features = engine.features(data[2], parameters, indices[2], algorithm=algorithm, stage="test",
                checkpoint=iteration, seed=seed*100000+iteration*20000+20000000)
            scores = selected_head.predict(features)
            key = f"{algorithm}_step{iteration}"
            test_results[key] = scores_metrics(scores, mapped[2])
            predictions[key] = np.asarray(config["digits"])[scores.argmax(1)]
            saved_arrays.update({key+"_parameters": parameters, key+"_feature_mean": selected_head.mean,
                key+"_feature_scale": selected_head.scale, key+"_weights": selected_head.weights,
                key+"_bias": selected_head.bias, key+"_test_features": features})
            for prefix, shots in zip(engine.readout_slots, engine.allocation):
                sequence = profile.sequence(data[2][0], parameters, prefix)
                entry = profile.export_example(sequence, folder/f"{key}_prefix{prefix}.json", shots)
                export.append(entry)
        def energy(records):
            return budget(records, calibration, profile.target, literature)
        target_results = {}
        for target in config["accuracy_targets"]:
            row = {"target_validation_accuracy": target, "algorithms": {}, "paired_target_reached": False}
            for algorithm in ("qrc", "vqc"):
                hit = choices[algorithm][str(target)]
                all_training = [r for r in engine.records if r["algorithm"] == algorithm and r["stage"] != "test"]
                if hit is None:
                    row["algorithms"][algorithm] = {"status": "not_reached_within_budget",
                        "best_validation_accuracy": max(c["validation_accuracy"] for c in checkpoints[algorithm]),
                        "training_budget_spent": energy(all_training)}
                    continue
                records = [r for r in engine.records[:hit["ledger_end"]] if r["algorithm"] == algorithm and r["stage"] != "test"]
                test_records = [r for r in engine.records if r["algorithm"] == algorithm and r["stage"] == "test"
                                and r["checkpoint"] == hit["iteration"]]
                key = f"{algorithm}_step{hit['iteration']}"
                row["algorithms"][algorithm] = {"status": "validation_target_reached", "checkpoint": hit,
                    "held_out_test": test_results[key], "energy_to_validation_target": energy(records),
                    "held_out_inference_energy": energy(test_records),
                    "training_plus_held_out_inference_energy": energy(records+test_records)}
            if all(item["status"] == "validation_target_reached" for item in row["algorithms"].values()):
                row["paired_target_reached"] = True
                a, b = [row["algorithms"][name]["held_out_test"]["accuracy"] for name in ("qrc", "vqc")]
                row["held_out_accuracy_difference_qrc_minus_vqc"] = a-b
                row["held_out_accuracies_within_declared_tolerance"] = abs(a-b) <= config["test_accuracy_tolerance"]+1e-12
            target_results[str(target)] = row
        classical = LinearHead.fit(data[0], data[1], *onehot[:2], config["alpha_grid"])
        repetition = {"seed": seed, "backend": backend.execution_kind, "checkpoints": checkpoints,
            "targets": target_results, "validation_selected_fallback": {algorithm: {
                "checkpoint": point, "held_out_test": test_results[f"{algorithm}_step{point['iteration']}"]}
                for algorithm, point in fallback.items()},
            "classical_linear_test": scores_metrics(classical.predict(data[2]), mapped[2]),
            "complete_study_execution_budget": {algorithm: energy([r for r in engine.records if r["algorithm"] == algorithm])
                                                 for algorithm in ("qrc", "vqc")},
            "feature_count_each": config["regime"]["n_qubits"]*(config["regime"]["n_qubits"]+1)//2*len(config["readout_slots"]),
            "export_examples": export, "pca_explained_variance": float(pca.explained_variance_ratio_.sum())}
        report["repeats"].append(repetition)
        np.savez_compressed(folder/"models_features.npz", **saved_arrays, initial_parameters=initial,
            pca_components=pca.components_, pca_mean=pca.mean_, pca_scale=scale, digits=config["digits"],
            config_json=json.dumps(config), profile_json=json.dumps(profile.metadata))
        np.savez_compressed(folder/"splits_predictions.npz", train_idx=indices[0], validation_idx=indices[1],
            test_idx=indices[2], train_labels=original[0], validation_labels=original[1], test_labels=original[2], **predictions)
        with (folder/"execution_ledger.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=engine.records[0].keys(), lineterminator="\n")
            writer.writeheader(); writer.writerows(engine.records)
        engine.save_counts(folder/"counts.jsonl.gz")
        (folder/"report.json").write_text(json.dumps(repetition, indent=2)+"\n")
        # Retain completed repeats if a later run fails, without claiming completion.
        (output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
    report["runtime_seconds"] = time.perf_counter()-started
    from .accuracy_energy_summary import summarize
    summarize(report, output)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"progress.json").unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/accuracy_energy_fresnel_beta.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/mnist")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=config.get("threads", 4)):
        run(config, args.output, args.cache)


def predict_saved(path, images, model_key="qrc_step0", seed=7, backend_factory=None, profile_factory=None):
    """Finite-shot inference with the saved PCA, pulse offsets and linear head."""
    with np.load(path, allow_pickle=False) as archive:
        config = json.loads(str(archive["config_json"]))
        saved = json.loads(str(archive["profile_json"]))
        profile = (AnalogProfile.build(config["target"], config["regime"]) if profile_factory is None else
                   profile_factory(config["target"], config["regime"]))
        if profile.target["device_sha256"] != saved["device_sha256"]:
            raise ValueError("Inference device differs from the saved model snapshot")
        if model_key+"_parameters" not in archive:
            raise ValueError("Requested model/checkpoint was not saved")
        data = (np.asarray(images)/255.-archive["pca_mean"])@archive["pca_components"].T
        data = np.clip(data/archive["pca_scale"], -3, 3)
        backend = (backend_factory(profile, seed) if backend_factory else
                   PulserCountsBackend(profile.device, config["noise_trajectories"]))
        engine = FeatureEngine(profile, backend, config["readout_slots"], config["shots_per_image"])
        features = engine.features(data, archive[model_key+"_parameters"], np.arange(len(data)),
            algorithm=model_key, stage="inference", checkpoint=0, seed=seed)
        scores = ((features-archive[model_key+"_feature_mean"])/archive[model_key+"_feature_scale"])@archive[model_key+"_weights"]+archive[model_key+"_bias"]
        return archive["digits"][scores.argmax(1)]


if __name__ == "__main__":
    main()
