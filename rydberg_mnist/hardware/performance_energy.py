"""Paired QRC/VQC performance and full-consumed-resource study (no accuracy target)."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..data import load_mnist
from ..experiment import scores_metrics
from .accuracy_energy import LinearHead, get_data, validate_config as validate_legacy_config
from .calibrated_cost import budget, validate_calibration
from .finite_backend import FeatureEngine, PulserCountsBackend
from .target_profile import AnalogProfile
from .dataset_protocol import load_test_exclusions, protocol_description, save_exclusions_snapshot, split_metadata


PROTOCOL = "full_training_performance_energy_v1"


def validate_config(config):
    # Reuse basic input checks without giving target accuracy any role in this protocol.
    validate_legacy_config(dict(config, accuracy_targets=[.5], test_accuracy_tolerance=0.))
    protocol_description(config)
    for key in ("validation_repetitions", "plateau_patience"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    value = config["plateau_relative_improvement"]
    if isinstance(value, bool) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("Plateau relative improvement must lie in (0,1)")
    if any(key in config for key in ("accuracy_targets", "early_stop_all_targets")):
        raise ValueError("Accuracy-target settings belong to the separate accuracy_energy protocol")


class Plateau:
    """Meaningful progress controls patience; absolute best still selects the model."""
    def __init__(self, initial_loss, patience, relative_improvement):
        self.reference = initial_loss
        self.patience = patience
        self.relative_improvement = relative_improvement
        self.stale = 0

    def observe(self, loss):
        if loss < self.reference and loss <= self.reference*(1-self.relative_improvement):
            self.reference, self.stale = loss, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def execution_seed(seed, step, stage, repetition=0):
    return int(np.random.SeedSequence([seed, step, stage, repetition]).generate_state(1)[0])


def account_repeat(repetition, records, energy):
    """Reusable by measurement-only reaccounting; full training never truncates at selection."""
    for algorithm, item in repetition["algorithms"].items():
        root = f"seed{repetition['seed']}/{algorithm}"
        own = [r for r in records if r["algorithm"] == algorithm]
        training = [r for r in own if r["stage"] != "test"]
        item["training_budget"] = energy(training, root+"/all_training")
        for label in ("selected", "terminal"):
            step = item[label+"_checkpoint"]["iteration"]
            inference = [r for r in own if r["stage"] == "test" and r["checkpoint"] == step]
            test_key = f"{root}/test_step{step}"
            item[label+"_inference_budget"] = energy(inference, test_key)
            item["training_plus_"+label+"_inference_budget"] = energy(training+inference,
                f"{root}/all_training_and_test_step{step}", (root+"/all_training", test_key))
        item["complete_study_execution_budget"] = energy(own, root+"/full_study")
        for point in repetition["checkpoints"][algorithm]:
            prefix = [r for r in records[:point["ledger_end"]]
                      if r["algorithm"] == algorithm and r["stage"] != "test"]
            point["training_budget_through_checkpoint"] = energy(prefix, f"{root}/train_to_step{point['iteration']}")
        # JSON reloads do not preserve the aliases present in an in-memory run.
        by_step = {p["iteration"]: p for p in repetition["checkpoints"][algorithm]}
        for label in ("selected", "terminal"):
            item[label+"_checkpoint"] = by_step[item[label+"_checkpoint"]["iteration"]]


def run(config, output, cache="data/mnist", backend_factory=None, profile_factory=None):
    validate_config(config)
    excluded, exclusion = load_test_exclusions(config)
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
    save_exclusions_snapshot(config, output, exclusion)
    device_source = Path(config["target"]).parent/profile.target["device_json"]
    device_bytes = device_source.read_bytes()
    if hashlib.sha256(device_bytes).hexdigest() != profile.target["device_sha256"]:
        raise ValueError("Device snapshot changed after compilation")
    (output/"device_snapshot.json").write_bytes(device_bytes)
    # This manifest can be relocated and loaded by the same compiler without the source repository.
    snapshot_target = dict(profile.target, device_json="device_snapshot.json")
    (output/"portable_target.json").write_text(json.dumps(snapshot_target, indent=2)+"\n")
    report = {"protocol_id": PROTOCOL, "status": "running", "config": config,
        "dataset_protocol": protocol_description(config),
        "device_profile": profile.metadata, "target": profile.target, "calibration": calibration,
        "literature_configuration": literature,
        "input_sha256": {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                         for key in ("target", "calibration", "literature_config")},
        "protocol": {"accuracy_is_outcome_not_stopping_target": True,
            "same_unique_images_qubits_pca_encoding_observables_shots_and_linear_head": True,
            "readout": "Training-only scaling; multi-output OLS/ridge, same alpha grid, minimum mean validation MSE",
            "validation_rule": "Fit on one finite-shot training feature set; average loss over independent validation measurement repetitions",
            "optimizer": "Alternating linear-head refits and sampled minibatch SPSA, frozen head between checkpoints",
            "selection": "Minimum mean validation MSE; ties retain earliest checkpoint; test labels never select models",
            "completion": "Validation-loss plateau or maximum SPSA steps; neither proves global convergence",
            "cost": "All consumed training, repeated validation, objective and discarded-checkpoint work, regardless of selected step",
            "initialization": "Paired initial offsets and common random measurement seeds at step zero; validation repetitions independent",
            "inference": "Selected and terminal models evaluated after selection; same-checkpoint measurements reused and charged once",
            "prior_test_inspection": exclusion["scope"],
            "hardware_jobs_submitted_by_default": False,
            "calibrated_energy": "Unknown without traceable target-specific power/cycle/setup and external classical measurements",
            "noise_trajectories": "Monte Carlo emulator approximation, not extra physical shots or independent experimental repeats"},
        "repeats": [], "versions": {name: importlib.metadata.version(name) for name in
            ("pulser-core", "pulser-simulation", "qutip", "numpy", "scipy", "scikit-learn")}}
    images, labels, test_images, test_labels = load_mnist(cache)
    if config.get("test_exclusions"):
        report["input_sha256"]["test_exclusions"] = exclusion["sha256"]
    for seed in config["repeat_seeds"]:
        repeat_started = time.perf_counter()
        folder = output/f"seed{seed}"; folder.mkdir()
        backend = backend_factory(profile, seed) if backend_factory else PulserCountsBackend(profile.device, config["noise_trajectories"])
        engine = FeatureEngine(profile, backend, config["readout_slots"], config["shots_per_image"])
        indices, data, mapped, onehot, pca, scale, original = get_data(images, labels, test_images, test_labels, config, seed,
                                                                   excluded_test_indices=excluded)
        rng = np.random.default_rng(seed)
        initial = rng.normal(0., .15, 2*config["regime"]["input_slots"])
        theta = initial.copy()
        checkpoints, stored, arrays = {"qrc": [], "vqc": []}, {}, {}

        def checkpoint(algorithm, parameters, step):
            tick = time.perf_counter()
            features = engine.features(data[0], parameters, indices[0], algorithm=algorithm, stage="train",
                checkpoint=step, seed=execution_seed(seed, step, 1))
            val = np.stack([engine.features(data[1], parameters, indices[1], algorithm=algorithm,
                stage="validation", checkpoint=step, seed=execution_seed(seed, step, 2, repetition))
                for repetition in range(config["validation_repetitions"])])
            flat = val.reshape(-1, val.shape[-1])
            head = LinearHead.fit(features, flat, onehot[0], np.tile(onehot[1], (len(val), 1)), config["alpha_grid"])
            scores = np.array([head.predict(v) for v in val])
            losses = np.mean((scores-onehot[1])**2, axis=(1, 2))
            accuracies = np.mean(scores.argmax(2) == mapped[1], axis=1)
            own = [r for r in engine.records if r["algorithm"] == algorithm]
            key = f"{algorithm}_step{step}"
            record = {"iteration": step, "model_key": key,
                "validation_mse": float(losses.mean()), "validation_accuracy": float(accuracies.mean()),
                "validation_mse_repetitions": losses.tolist(), "validation_accuracy_repetitions": accuracies.tolist(),
                "alpha": head.alpha, "cumulative_requested_shots": sum(r["shots"] for r in own),
                "cumulative_pulse_programs": len(own), "ledger_end": len(engine.records),
                "classical_head_candidates_fit": len(config["alpha_grid"]),
                "checkpoint_cpu_wall_seconds": time.perf_counter()-tick}
            checkpoints[algorithm].append(record)
            stored[(algorithm, step)] = (parameters.copy(), head)
            arrays.update({key+"_parameters": parameters.copy(), key+"_feature_mean": head.mean,
                key+"_feature_scale": head.scale, key+"_weights": head.weights, key+"_bias": head.bias,
                key+"_train_features": features, key+"_validation_features": val})
            print(f"seed {seed} {algorithm} step {step}: validation {record['validation_accuracy']:.1%}, "
                  f"MSE {record['validation_mse']:.5f}; shots {record['cumulative_requested_shots']:,}", flush=True)
            return head

        checkpoint("qrc", initial, 0)
        head = checkpoint("vqc", theta, 0)
        monitor = Plateau(checkpoints["vqc"][0]["validation_mse"], config["plateau_patience"], config["plateau_relative_improvement"])
        optimizer = []
        stop_reason = "budget_limited"
        for step in range(1, config["vqc_steps"]+1):
            subset = rng.choice(config["n_train"], config["minibatch_size"], replace=False)
            direction = rng.choice([-1., 1.], len(theta))
            perturbation = config["spsa_perturbation"]/step**.101
            learning = config["spsa_learning_rate"]/(step+4)**.602
            objectives = []
            for sign, stage_code in ((1, 3), (-1, 4)):
                trial = np.clip(theta+sign*perturbation*direction, -config["parameter_bound"], config["parameter_bound"])
                features = engine.features(data[0][subset], trial, indices[0][subset], algorithm="vqc",
                    stage="spsa_objective", checkpoint=step, seed=execution_seed(seed, step, stage_code))
                objectives.append(float(np.mean((head.predict(features)-onehot[0][subset])**2)))
            theta = np.clip(theta-learning*(objectives[0]-objectives[1])/(2*perturbation)*direction,
                            -config["parameter_bound"], config["parameter_bound"])
            optimizer.append({"iteration": step, "source_indices": indices[0][subset].tolist(),
                "objective_plus": objectives[0], "objective_minus": objectives[1],
                "learning_rate": learning, "perturbation": perturbation, "parameters_after_update": theta.tolist()})
            if step % config["checkpoint_every"] == 0 or step == config["vqc_steps"]:
                head = checkpoint("vqc", theta, step)
                plateau = monitor.observe(checkpoints["vqc"][-1]["validation_mse"])
                checkpoints["vqc"][-1]["plateau_stale_checks"] = monitor.stale
                if plateau:
                    stop_reason = "validation_loss_plateau"
                    break
        # Freeze both choices before accessing held-out outcomes.
        selected = {name: min(points, key=lambda p: p["validation_mse"]) for name, points in checkpoints.items()}
        terminal = {name: points[-1] for name, points in checkpoints.items()}
        tests, predictions, exports = {}, {}, []
        choices = {(name, point["iteration"]) for group in (selected, terminal) for name, point in group.items()}
        for algorithm, step in sorted(choices):
            parameters, model = stored[(algorithm, step)]
            features = engine.features(data[2], parameters, indices[2], algorithm=algorithm, stage="test",
                checkpoint=step, seed=execution_seed(seed, step, 5))
            key = f"{algorithm}_step{step}"
            scores = model.predict(features)
            tests[key] = scores_metrics(scores, mapped[2])
            predictions[key] = np.asarray(config["digits"])[scores.argmax(1)]
            arrays[key+"_test_features"] = features
            for prefix, total in zip(engine.readout_slots, engine.allocation):
                sequence = profile.sequence(data[2][0], parameters, prefix)
                for job, shots in enumerate(engine.job_shots(total)):
                    exports.append(profile.export_example(sequence, folder/f"{key}_prefix{prefix}_job{job}.json", shots))
        classical = LinearHead.fit(data[0], data[1], *onehot[:2], config["alpha_grid"])
        repetition = {"seed": seed, "backend": backend.execution_kind, "checkpoints": checkpoints,
            "dataset": split_metadata(indices, original, test_labels, config, exclusion, excluded_test_indices=excluded),
            "optimizer_coverage": {"image_draws": len(optimizer)*config["minibatch_size"],
                "distinct_training_images_in_objectives": len({i for entry in optimizer for i in entry["source_indices"]}),
                "fit_training_images": config["n_train"],
                "scope": "SPSA objective coverage only; every checkpoint fits the head on all training images"},
            "optimizer_history": optimizer, "algorithms": {name: {
                "completion": {"reason": "fixed_reservoir_fit_complete" if name == "qrc" else stop_reason,
                    "optimizer_steps": 0 if name == "qrc" else len(optimizer),
                    "global_convergence_claimed": False},
                "selected_checkpoint": selected[name], "terminal_checkpoint": terminal[name],
                "selected_test": tests[selected[name]["model_key"]], "terminal_test": tests[terminal[name]["model_key"]]}
                for name in ("qrc", "vqc")},
            "classical_linear_test": scores_metrics(classical.predict(data[2]), mapped[2]),
            "feature_count_each": config["regime"]["n_qubits"]*(config["regime"]["n_qubits"]+1)//2*len(config["readout_slots"]),
            "export_examples": exports, "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "runtime_seconds_local_simulation": time.perf_counter()-repeat_started}
        account_repeat(repetition, engine.records, lambda rows, key, components=(): budget(rows, calibration, profile.target, literature))
        report["repeats"].append(repetition)
        np.savez_compressed(folder/"models_features.npz", **arrays, initial_parameters=initial,
            pca_components=pca.components_, pca_mean=pca.mean_, pca_scale=scale, digits=config["digits"],
            config_json=json.dumps(config), profile_json=json.dumps(profile.metadata))
        np.savez_compressed(folder/"splits_predictions.npz", train_idx=indices[0], validation_idx=indices[1],
            test_idx=indices[2], train_labels=original[0], validation_labels=original[1], test_labels=original[2], **predictions)
        with (folder/"execution_ledger.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=engine.records[0].keys(), lineterminator="\n")
            writer.writeheader(); writer.writerows(engine.records)
        engine.save_counts(folder/"counts.jsonl.gz")
        (folder/"report.json").write_text(json.dumps(repetition, indent=2)+"\n")
        (output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
    report["status"] = "completed_local_simulation" if backend_factory is None else "completed_injected_backend"
    report["runtime_seconds"] = time.perf_counter()-started
    from .performance_energy_summary import summarize
    summarize(report, output)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"progress.json").unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/performance_energy_fresnel_beta.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/mnist")
    parser.add_argument("--prepare-only", action="store_true", help="Save verified real-data splits/PCA and resource bounds; do not train or submit jobs")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=config.get("threads", 4)):
        if args.prepare_only:
            from .dataset_protocol import prepare
            validate_config(config)
            prepare(config, args.output, args.cache, AnalogProfile.build(config["target"], config["regime"]))
        else:
            run(config, args.output, args.cache)


if __name__ == "__main__":
    main()
