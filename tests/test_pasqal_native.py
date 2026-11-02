"""Global-pulse classification checks; synthetic data is never an accuracy claim."""
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

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("qoolqit"), "Install hardware extras for native pulse tests")
class PasqalNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rydberg_mnist.hardware.pasqal import load_target
        _, cls.device = load_target(ROOT/"configs/pasqal_target.json")
        cls.config = json.loads((ROOT/"configs/pasqal_native_mnist.json").read_text())

    def test_geometry_and_instruction_duration_do_not_encode_inputs(self):
        from rydberg_mnist.hardware.pasqal_native import make_sequence
        from pulser.sampler import sample
        sequences = [make_sequence(self.device, self.config, np.full(8, x), np.full(8, theta))
                     for x, theta in ((-3., -4.), (0., 0.), (3., 4.))]
        reference = sequences[0]
        for sequence in sequences:
            self.assertEqual(sequence.get_duration(), 512)
            self.assertEqual(set(sequence.declared_channels), {"rydberg"})
            self.assertEqual(sequence.declared_channels["rydberg"].addressing, "Global")
            np.testing.assert_array_equal(list(sequence.register.qubits.values()), list(reference.register.qubits.values()))
        a = np.asarray(sample(sequences[0]).channel_samples["rydberg"].amp)
        b = np.asarray(sample(sequences[-1]).channel_samples["rydberg"].amp)
        self.assertGreater(np.linalg.norm(a-b), 1.)

    def test_joint_moments_and_shared_bitstrings_agree(self):
        from rydberg_mnist.hardware.pasqal_native import moments_to_features, sampled_features
        moment = moments_to_features([.5, .5], [[.5, .5], [.5, .5]])
        sampled = sampled_features({"00": 500, "11": 500}, 2)
        np.testing.assert_array_equal(moment, [0., 0., 1.])
        np.testing.assert_array_equal(moment, sampled)
        np.testing.assert_array_equal(moments_to_features([.5, .5], [[.5, 0], [0, .5]]), [0., 0., -1.])

    def test_pulse_parameters_change_emulated_quantum_features(self):
        from rydberg_mnist.hardware.pasqal_native import make_sequence, emulate_sequence
        config = {**self.config, "n_qubits": 2, "input_slots": 2}
        data = np.zeros(4)
        a, _ = emulate_sequence(make_sequence(self.device, config, data, np.zeros(4)), self.device)
        b, _ = emulate_sequence(make_sequence(self.device, config, data, np.ones(4)), self.device)
        self.assertGreater(np.linalg.norm(a-b), 1e-4)

    def test_training_and_saved_native_inference(self):
        from rydberg_mnist.hardware.pasqal_native import run, predict_saved
        rng = np.random.default_rng(4)
        images = rng.integers(0, 256, (16, 784), dtype=np.uint8)
        labels = np.tile([3, 8], 8)
        test_images = rng.integers(0, 256, (4, 784), dtype=np.uint8)
        test_labels = np.array([3, 8, 3, 8])
        config = {**self.config, "target": str(ROOT/"configs/pasqal_target.json"),
                  "n_qubits": 2, "input_slots": 2, "n_train": 8, "n_val": 4, "n_test": 4,
                  "iterations": 1, "checkpoint_every": 1, "shots": 100}
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), \
             patch("rydberg_mnist.hardware.pasqal_native.load_mnist", return_value=(images, labels, test_images, test_labels)), \
             redirect_stdout(io.StringIO()):
            output = Path(directory)/"experiment"
            report = run(config, output)
            with np.load(output/"splits_predictions.npz") as archive:
                predicted = predict_saved(output/"model.npz", test_images[archive["test_idx"]])
                np.testing.assert_array_equal(predicted, archive["trained_native_pulses_linear"])
            self.assertEqual(report["parameter_count"], 4)
            self.assertGreater(report["compiled_emulated_sequences"], 4)
            self.assertIsNone(report["first_test_input_energy"]["end_to_end_pipeline_energy_j"])
            self.assertTrue((output/"compiled_example.json").exists())


if __name__ == "__main__":
    unittest.main()
