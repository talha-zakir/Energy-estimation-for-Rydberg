"""Literature-sized MNIST protocols, frozen holdout exclusions and dataset preflight."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from ..data import FILES, load_mnist
from .accuracy_energy import get_data


PAPER = "https://arxiv.org/html/2407.02553v1"
PRESETS = {
    "kornjaca_binary_hardware": {"digits": [3, 8], "n_train": 800, "n_val": 200, "n_test": 200},
    "kornjaca_tenclass_hardware": {"digits": list(range(10)), "n_train": 1200, "n_val": 300, "n_test": 500},
    "kornjaca_tenclass_simulation": {"digits": list(range(10)), "n_train": 8000, "n_val": 2000, "n_test": 1000},
    "exploratory_smoke": {"digits": [3, 8], "n_train": 80, "n_val": 20, "n_test": 40},
}


def protocol_description(config):
    name = config.get("dataset_protocol", "custom")
    if name != "custom":
        if name not in PRESETS:
            raise ValueError(f"Unknown dataset protocol: {name}")
        if any(config[key] != value for key, value in PRESETS[name].items()):
            raise ValueError("Dataset counts/classes disagree with the named protocol; use custom for an adaptation")
    literature = name.startswith("kornjaca_")
    return {"name": name, "source": PAPER if literature else None,
        "dataset_size_source": PAPER+"#S2.SS1" if name.endswith("simulation") else PAPER+"#S1.SS1.SSS1" if literature else None,
        "validation_source": PAPER+"#S1.SS2.SSS2" if literature else None,
        "training_pool_size": config["n_train"]+config["n_val"],
        "fit_training_size": config["n_train"], "validation_size": config["n_val"], "test_size": config["n_test"],
        "validation_is_inside_published_training_pool": literature,
        "scope": "Dataset sizes and internal 80/20 split adapted from the paper; FRESNEL compiler, regression readout and "
                 "finite-shot optimization are our protocol. Configured prior-test exclusions are an additional adaptation. "
                 "This is not a reproduction of the paper's Aquila/SVM experiment." if literature else
                 "Custom or historical exploratory configuration; no claim of matching a published dataset protocol"}


def load_test_exclusions(config):
    if not config.get("test_exclusions"):
        return None, {"manifest": None, "sha256": None, "excluded_count": 0,
                      "scope": "No prior-inspection exclusions; do not describe this as a fresh holdout"}
    path = Path(config["test_exclusions"])
    payload = path.read_bytes(); manifest = json.loads(payload)
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") != "original_mnist_official_test"
            or manifest.get("official_test_size") != 10000
            or manifest.get("official_test_labels_md5") != FILES["t10k-labels-idx1-ubyte.gz"]):
        raise ValueError("Test exclusion manifest does not identify the original MNIST test set")
    indices = manifest.get("excluded_test_indices")
    if (not isinstance(indices, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 10000 for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError("Test exclusion manifest requires sorted unique official-test indices")
    return indices, {"manifest": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
        "excluded_count": len(indices), "scope": manifest["scope"]}


def save_exclusions_snapshot(config, output, exclusion):
    if config.get("test_exclusions"):
        payload = Path(config["test_exclusions"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != exclusion["sha256"]:
            raise ValueError("Test exclusion manifest changed after loading")
        (Path(output)/"test_exclusions_snapshot.json").write_bytes(payload)


def split_metadata(indices, original, test_labels, config, exclusion, *, excluded_test_indices=None):
    if exclusion["manifest"] and excluded_test_indices is None:
        raise ValueError("Use the frozen loaded exclusions when recording a split")
    excluded = excluded_test_indices
    eligible = np.flatnonzero(np.isin(test_labels, config["digits"]))
    eligible = eligible[~np.isin(eligible, excluded if excluded is not None else [])]
    digest = hashlib.sha256(json.dumps([values.tolist() for values in indices], separators=(",", ":")).encode()).hexdigest()
    return {"protocol": protocol_description(config), "split_sha256": digest,
        "source_namespaces": ["official_train", "official_train", "official_test"],
        "class_counts": {name: {str(digit): int(np.sum(values == digit)) for digit in config["digits"]}
                         for name, values in zip(("train", "validation", "test"), original)},
        "test_exclusions": exclusion, "eligible_test_images_after_known_exclusions": len(eligible),
        "selection_uses_prediction_accuracy": False,
        "repeat_test_sets_may_overlap": True,
        "holdout_scope": exclusion["scope"]}


def resource_plan(config, max_runs):
    """Upper bounds for this declared workload, never observed accuracy or calibrated energy."""
    from .qrc import allocate_shots
    allocation = allocate_shots(config["shots_per_image"], len(config["readout_slots"]))
    jobs_per_image = sum(math.ceil(shots/max_runs) for shots in allocation)
    checkpoints = 1+math.ceil(config["vqc_steps"]/config["checkpoint_every"])
    checkpoint_images = config["n_train"]+config["validation_repetitions"]*config["n_val"]
    objective_images = 2*config["minibatch_size"]*config["vqc_steps"]
    draws = config["minibatch_size"]*config["vqc_steps"]
    def workload(images):
        return {"image_feature_evaluations": images, "requested_shots": images*config["shots_per_image"],
                "count_requests_after_job_splitting": images*jobs_per_image}
    return {"status": "planned_bounds_not_executed", "qrc_training": workload(checkpoint_images),
        "vqc_training_at_safety_cap": workload(checkpoints*checkpoint_images+objective_images),
        "one_test_pass": workload(config["n_test"]), "maximum_vqc_checkpoint_fits": checkpoints,
        "maximum_full_study_shots_all_seeds": len(config["repeat_seeds"])*config["shots_per_image"]*
            (checkpoint_images+checkpoints*checkpoint_images+objective_images+3*config["n_test"]),
        "vqc_optimizer_image_draws_at_cap": draws, "optimizer_draws_per_training_image_at_cap": draws/config["n_train"],
        "optimizer_scope": "Repeated minibatch draws; not guaranteed dataset coverage. Linear-head checkpoints use all fit-training images. "
                           "Unchanged optimizer safety cap is not proof of sufficient training on the larger dataset.",
        "calibrated_qpu_energy_j": None, "hardware_jobs_submitted": 0}


def prepare(config, output, cache, profile):
    """Verify real data/splits/PCA and budget before an expensive noisy quantum run."""
    description = protocol_description(config)
    excluded, exclusion = load_test_exclusions(config)
    images, labels, test_images, test_labels = load_mnist(cache)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    save_exclusions_snapshot(config, output, exclusion)
    plan = {"status": "prepared_not_trained", "performance_results_available": False,
        "dataset_protocol": description, "config": config, "device_profile": profile.metadata,
        "device_sha256": profile.target["device_sha256"], "official_dataset_checksums_md5": FILES,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "resource_plan": resource_plan(config, profile.device.max_runs), "repeats": []}
    selected_tests = {}
    for seed in config["repeat_seeds"]:
        indices, data, _, _, pca, scale, original = get_data(images, labels, test_images, test_labels, config, seed,
                                                           excluded_test_indices=excluded)
        folder = output/f"seed{seed}"; folder.mkdir()
        np.savez_compressed(folder/"dataset.npz", train_idx=indices[0], validation_idx=indices[1], test_idx=indices[2],
            train_features=data[0], validation_features=data[1], test_features=data[2],
            pca_components=pca.components_, pca_mean=pca.mean_, pca_scale=scale)
        metadata = split_metadata(indices, original, test_labels, config, exclusion, excluded_test_indices=excluded)
        (folder/"dataset.json").write_text(json.dumps(metadata, indent=2)+"\n")
        plan["repeats"].append(dict(metadata, seed=seed))
        selected_tests[seed] = indices[2]
    plan["pairwise_test_overlap_counts"] = {f"{a}/{b}": len(np.intersect1d(selected_tests[a], selected_tests[b]))
        for a in selected_tests for b in selected_tests if a < b}
    (output/"dataset_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    lines = ["# Prepared literature-sized dataset; quantum study not run", "",
        f"MNIST {config['digits']}: {config['n_train']} fit-training + {config['n_val']} validation + {config['n_test']} test images per seed.",
        f"Source: {description['dataset_size_source']}", "",
        "The validation images are inside the training-pool count. Both algorithms use the same deterministic splits and training-only PCA.",
        f"Frozen prior-test exclusions: {exclusion['excluded_count']}; manifest SHA-256: {exclusion['sha256']}.",
        "These files contain dataset preparation and planned resource bounds. They contain no new QRC/VQC accuracies or measured QPU energy.", "",
        f"Planned QRC fit/validation shots per seed: {plan['resource_plan']['qrc_training']['requested_shots']:,}.",
        f"VQC fit/validation/objective shots per seed at the safety cap: {plan['resource_plan']['vqc_training_at_safety_cap']['requested_shots']:,}.",
        "Plateau stopping can consume less; every actual request will be accounted for by a completed run.", "",
        plan["resource_plan"]["optimizer_scope"], ""]
    (output/"SUMMARY.md").write_text("\n".join(lines))
    return plan
