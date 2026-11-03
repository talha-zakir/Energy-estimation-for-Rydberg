"""Causal reservoir dynamics, destructive-readout budgets, and saved inference."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from rydberg_mnist.hardware.qrc_compare import composite_energy, paired_result

ROOT = Path(__file__).resolve().parents[1]


class QRCCostTests(unittest.TestCase):
    def test_every_destructive_prefix_pays_reset_and_readout(self):
        literature = json.loads((ROOT/"configs/literature_energy.json").read_text())
        literature["transferred_scenarios"]["system_power_scenarios"] = [
            {"id": "test", "power_w": 2., "source": "synthetic unit check"}]
        value = composite_energy([.1, .2], [100, 100], literature)
        # Two independent batches, each 100 shots and 20 ms overhead per shot.
        self.assertEqual(value["physical_shots"], 200)
        self.assertAlmostEqual(value["system_scenarios"]["test"]["electrical_j"], 8.00006)
        self.assertAlmostEqual(value["system_scenarios"]["test"]["allocated_hardware_s"], 4.00003)
        self.assertIsNone(value["measured_device_energy_j"])
        with self.assertRaises(ValueError):
            composite_energy([.1, .2], [100], literature)

    def test_paired_test_uses_discordant_images(self):
        value = paired_result([3, 3, 8, 8], [3, 8, 8, 3], [3, 8, 8, 8])
        self.assertEqual(value["vqc_correct_qrc_wrong"], 1)
        self.assertEqual(value["qrc_correct_vqc_wrong"], 1)
        self.assertEqual(value["mcnemar_exact_two_sided_p"], 1.)
        with self.assertRaises(ValueError):
            paired_result([3], [3, 8], [3])


@unittest.skipUnless(importlib.util.find_spec("qoolqit"), "Install hardware extras for QRC")
class PulserReservoirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rydberg_mnist.hardware.pasqal import load_target
        cls.target, cls.device = load_target(ROOT/"configs/pasqal_target.json")
        cls.native = json.loads((ROOT/"configs/pasqal_native_mnist.json").read_text())
        cls.native.update(n_qubits=2, input_slots=2)

    def test_budget_allocation_and_count_subsetting(self):
        from rydberg_mnist.hardware.qrc import allocate_shots, subset_counts
        self.assertEqual(allocate_shots(1000, 4), [250]*4)
        self.assertEqual(allocate_shots(1000, 3), [334, 333, 333])
        with self.assertRaises(ValueError):
            allocate_shots(3, 4)
        counts = {"00": 50, "11": 50}
        subset = subset_counts(counts, 25, np.random.default_rng(7))
        self.assertEqual(sum(subset.values()), 25)
        self.assertLessEqual(subset.get("00", 0), 50)
        self.assertTrue(set(subset).issubset(counts))
        with self.assertRaises(ValueError):
            subset_counts(counts, 101, np.random.default_rng(7))

    def test_prefixes_cannot_see_future_input(self):
        from pulser.sampler import sample
        from rydberg_mnist.hardware.qrc import reservoir_sequence
        first = np.zeros(4); second = first.copy(); second[[1, 3]] = 3.
        a = reservoir_sequence(self.device, self.native, first, np.zeros(4), 1)
        b = reservoir_sequence(self.device, self.native, second, np.zeros(4), 1)
        for field in ("amp", "det"):
            np.testing.assert_array_equal(getattr(sample(a).channel_samples["rydberg"], field),
                                          getattr(sample(b).channel_samples["rydberg"], field))

    def test_terminal_prefix_matches_vqc_and_retains_input_history(self):
        from rydberg_mnist.hardware.pasqal_native import make_sequence
        from rydberg_mnist.hardware.qrc import reservoir_sequence, emulate_counts
        first = np.zeros(4); changed = first.copy(); changed[0] = 3.
        a = reservoir_sequence(self.device, self.native, first, np.zeros(4), 2)
        b = make_sequence(self.device, self.native, first, np.zeros(4))
        # Compare actual samples, avoiding metadata identifiers in JSON reprs.
        from pulser.sampler import sample
        np.testing.assert_array_equal(sample(a).channel_samples["rydberg"].amp,
                                      sample(b).channel_samples["rydberg"].amp)
        x, _ = emulate_counts(a, self.device)
        y, _ = emulate_counts(reservoir_sequence(self.device, self.native, changed, np.zeros(4), 2), self.device)
        self.assertGreater(np.linalg.norm(x-y), 1e-4)
        np.testing.assert_array_equal(list(a.register.qubits.values()), list(b.register.qubits.values()))

    def test_reservoir_parameters_are_immutable_and_slot_validation_is_strict(self):
        from rydberg_mnist.hardware.qrc import ReservoirEvaluator, validate_config
        evaluator = ReservoirEvaluator(self.device, self.native, np.zeros(4), [1, 2], 1, self.target)
        with self.assertRaises(ValueError):
            evaluator.biases[0] = 1.
        config = json.loads((ROOT/"configs/pasqal_qrc_mnist.json").read_text())
        for slots in ([2, 1], [1, 1, 2], [1], [0, 2], [True, 2]):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                validate_config({**config, "readout_slots": slots}, self.native)

    def test_readout_only_training_and_inference_roundtrip(self):
        from rydberg_mnist.hardware.pasqal_native import run as native_run
        from rydberg_mnist.hardware.qrc import run, predict_saved, load_reference
        rng = np.random.default_rng(18)
        images = rng.integers(0, 256, (16, 784), dtype=np.uint8)
        labels = np.tile([3, 8], 8)
        test_images = rng.integers(0, 256, (4, 784), dtype=np.uint8)
        test_labels = np.array([3, 8, 3, 8])
        dataset = (images, labels, test_images, test_labels)
        native = {**self.native, "target": str(ROOT/"configs/pasqal_target.json"),
            "n_train": 8, "n_val": 4, "n_test": 4, "iterations": 1, "checkpoint_every": 1, "shots": 64}
        config = json.loads((ROOT/"configs/pasqal_qrc_mnist.json").read_text())
        config.update(readout_slots=[1, 2], total_shots_per_prediction=64, shots_per_readout=64,
                      noise_trajectories=1, literature_config=str(ROOT/"configs/literature_energy.json"))
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
            reference = Path(directory)/"reference"
            with patch("rydberg_mnist.hardware.pasqal_native.load_mnist", return_value=dataset):
                native_run(native, reference)
            config["reference_run"] = str(reference)
            output = Path(directory)/"qrc"
            with patch("rydberg_mnist.hardware.qrc.load_mnist", return_value=dataset), \
                 patch("rydberg_mnist.hardware.pasqal_native.train", side_effect=AssertionError("QRC must not optimize quantum controls")):
                report = run(config, output)
                with np.load(output/"splits_predictions.npz") as split:
                    for variant in ("qrc_temporal", "qrc_endpoint"):
                        predicted = predict_saved(output/"model.npz", test_images[split["test_idx"]], variant=variant)
                        np.testing.assert_array_equal(predicted, split[variant+"__coherent"])
                # A reference with changed split indices is rejected before evaluation.
                path = reference/"splits_predictions.npz"
                with np.load(path) as split:
                    altered = {key: split[key] for key in split.files}
                altered["test_idx"] = altered["test_idx"][::-1]
                np.savez_compressed(path, **altered)
                with self.assertRaisesRegex(ValueError, "split differs"):
                    load_reference(reference, "unused")
            self.assertEqual(report["protocol"]["quantum_training_steps_qrc"], 0)
            self.assertEqual(report["models"]["qrc_temporal"]["feature_count"], 6)
            self.assertEqual(report["models"]["qrc_endpoint"]["feature_count"], 3)
            budget = report["energy"]["training_projection"]["qrc_temporal_equal_total_shots"]
            self.assertEqual(budget["physical_shots"], 12*64)
            self.assertEqual(budget["circuit_feature_evaluations"], 24)
            self.assertTrue(report["pairing"]["coherent_vqc_and_endpoint_predictions_reproduce_archive"])
            self.assertTrue((output/"compiled_examples/qrc_prefix1.json").exists())


if __name__ == "__main__":
    unittest.main()
