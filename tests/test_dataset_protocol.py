"""Published pool sizes, frozen test exclusions, train-only PCA and untrained preflight."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from rydberg_mnist.data import split_indices

ROOT = Path(__file__).resolve().parents[1]


class ExcludedSplitTests(unittest.TestCase):
    def test_paired_stratified_official_splits_exclude_known_test_history(self):
        labels = np.tile([3, 8], 1200)
        test_labels = np.tile([3, 8], 500)
        excluded = list(range(100))
        args = (labels, test_labels, [3, 8], 800, 200, 200, 7)
        left = split_indices(*args, excluded_test_indices=excluded)
        right = split_indices(*args, excluded_test_indices=excluded)
        for a, b, size in zip(left, right, (800, 200, 200)):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(np.unique(a)), size)
        self.assertFalse(np.intersect1d(left[0], left[1]).size)
        self.assertFalse(np.intersect1d(left[2], excluded).size)
        self.assertEqual(np.sum(test_labels[left[2]] == 3), 100)
        other = split_indices(*args[:-1], 11, excluded_test_indices=excluded)
        self.assertFalse(np.array_equal(left[2], other[2]))

    def test_invalid_or_exhausted_test_exclusions_fail(self):
        labels = np.tile([3, 8], 40)
        for excluded in ([True], [0.5], [-1], [80], [[0]]):
            with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                split_indices(labels, labels, [3, 8], 40, 20, 20, 7, excluded_test_indices=excluded)
        with self.assertRaisesRegex(ValueError, "available"):
            split_indices(labels, labels, [3, 8], 40, 20, 20, 7, excluded_test_indices=list(range(70)))
        np.testing.assert_array_equal(split_indices(labels, labels, [3, 8], 40, 20, 20, 7)[2],
            split_indices(labels, labels, [3, 8], 40, 20, 20, 7, excluded_test_indices=[])[2])


@unittest.skipUnless(importlib.util.find_spec("pulser"), "Install hardware extras")
class LiteratureDatasetTests(unittest.TestCase):
    def config(self, filename="performance_energy_fresnel_beta.json"):
        config = json.loads((ROOT/"configs"/filename).read_text())
        for key in ("target", "calibration", "literature_config", "test_exclusions"):
            if key in config:
                config[key] = str(ROOT/config[key])
        return config

    def test_published_training_pool_contains_validation_in_all_presets(self):
        from rydberg_mnist.hardware.dataset_protocol import protocol_description
        from rydberg_mnist.hardware.performance_energy import validate_config
        for filename, fit, validation, test in (
            ("performance_energy_fresnel_beta.json", 800, 200, 200),
            ("performance_energy_fresnel_beta_tenclass.json", 1200, 300, 500),
            ("performance_energy_fresnel_beta_tenclass_simulation.json", 8000, 2000, 1000)):
            config = self.config(filename)
            validate_config(config)
            spec = protocol_description(config)
            self.assertEqual((config["n_train"], config["n_val"], config["n_test"]), (fit, validation, test))
            self.assertEqual(spec["training_pool_size"], fit+validation)
            self.assertAlmostEqual(validation/(fit+validation), .2)
        invalid = self.config(); invalid["n_train"] = 1000
        with self.assertRaisesRegex(ValueError, "named protocol"):
            validate_config(invalid)
        invalid["dataset_protocol"] = "custom"
        validate_config(invalid)

    def test_frozen_history_matches_recorded_archive_hashes_and_indices(self):
        from rydberg_mnist.hardware.dataset_protocol import load_test_exclusions
        config = self.config()
        excluded, info = load_test_exclusions(config)
        manifest_path = Path(config["test_exclusions"])
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(info["sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        recorded = set()
        for source in manifest["source_archives"]:
            archive_path = ROOT/source["path"]
            self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), source["sha256"])
            with np.load(archive_path, allow_pickle=False) as archive:
                recorded.update(map(int, archive["test_idx"]))
                self.assertEqual(len(archive["test_idx"]), source["test_index_count"])
        self.assertEqual(excluded, sorted(recorded))
        # Reject a manifest from another dataset and malformed/duplicate indices.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.json"
            for changes in ({"official_test_labels_md5": "wrong"}, {"excluded_test_indices": [1, 1]},
                            {"excluded_test_indices": [True]}, {"excluded_test_indices": [10000]}):
                path.write_text(json.dumps(dict(manifest, **changes)))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    load_test_exclusions(dict(config, test_exclusions=str(path)))

    def test_validation_and_test_pixels_never_fit_pca_or_scaling(self):
        from rydberg_mnist.hardware.accuracy_energy import get_data
        rng = np.random.default_rng(1)
        images = rng.integers(0, 256, (160, 16), dtype=np.uint8)
        tests = rng.integers(0, 256, (80, 16), dtype=np.uint8)
        labels, test_labels = np.tile([3, 8], 80), np.tile([3, 8], 40)
        config = self.config(); config.update(n_train=80, n_val=20, n_test=20, dataset_protocol="custom")
        with threadpool_limits(limits=1):
            first = get_data(images, labels, tests, test_labels, config, 7, excluded_test_indices=[0, 1])
            changed = images.copy(); changed[first[0][1]] = 255
            second = get_data(changed, labels, np.full_like(tests, 255), test_labels, config, 7,
                              excluded_test_indices=[0, 1])
        np.testing.assert_array_equal(first[4].components_, second[4].components_)
        np.testing.assert_array_equal(first[4].mean_, second[4].mean_)
        np.testing.assert_array_equal(first[5], second[5])
        np.testing.assert_array_equal(first[1][0], second[1][0])

    def test_planned_counts_follow_actual_recipe_and_are_not_measured_energy(self):
        from rydberg_mnist.hardware.dataset_protocol import resource_plan
        config = self.config(); plan = resource_plan(config, 500)
        self.assertEqual(plan["qrc_training"]["requested_shots"], 1_400_000)
        self.assertEqual(plan["vqc_training_at_safety_cap"]["requested_shots"], 24_824_000)
        self.assertEqual(plan["maximum_full_study_shots_all_seeds"], 80_472_000)
        self.assertEqual(plan["one_test_pass"]["requested_shots"], 200_000)
        self.assertEqual(plan["vqc_optimizer_image_draws_at_cap"], 512)
        self.assertEqual(plan["optimizer_draws_per_training_image_at_cap"], .64)
        self.assertEqual(resource_plan(config, 100)["one_test_pass"]["count_requests_after_job_splitting"], 2400)
        self.assertIsNone(plan["calibrated_qpu_energy_j"])

    def test_preparation_uses_full_declared_counts_but_never_runs_quantum_backend(self):
        from rydberg_mnist.hardware.dataset_protocol import prepare
        from rydberg_mnist.hardware.target_profile import AnalogProfile
        rng = np.random.default_rng(2)
        dataset = (rng.integers(0, 256, (1400, 16), dtype=np.uint8), np.tile([3, 8], 700),
                   rng.integers(0, 256, (600, 16), dtype=np.uint8), np.tile([3, 8], 300))
        config = self.config(); config.pop("test_exclusions"); config["repeat_seeds"] = [7]
        profile = AnalogProfile.build(config["target"], config["regime"])
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), \
             patch("rydberg_mnist.hardware.dataset_protocol.load_mnist", return_value=dataset), \
             patch("rydberg_mnist.hardware.finite_backend.PulserCountsBackend.sample_counts", side_effect=AssertionError("no quantum job")):
            output = Path(directory)/"prepared"
            result = prepare(config, output, "unused", profile)
            self.assertEqual(result["status"], "prepared_not_trained")
            self.assertFalse(result["performance_results_available"])
            self.assertFalse((output/"report.json").exists())
            with np.load(output/"seed7/dataset.npz") as saved:
                self.assertEqual([len(saved[key+"_idx"]) for key in ("train", "validation", "test")], [800, 200, 200])
                np.testing.assert_allclose(saved["pca_mean"], (dataset[0][saved["train_idx"]]/255.).mean(0))
            self.assertEqual(result["repeats"][0]["protocol"]["training_pool_size"], 1000)
            self.assertIsNone(result["repeats"][0]["test_exclusions"]["manifest"])
            self.assertIn("No prior-inspection exclusions", result["repeats"][0]["holdout_scope"])


if __name__ == "__main__":
    unittest.main()
