"""Verify target provenance and reject unsupported architecture claims."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("qoolqit"), "Install hardware extras for Pasqal target checks")
class PasqalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rydberg_mnist.hardware.pasqal import load_target
        cls.config, cls.device = load_target(ROOT/"configs/pasqal_target.json")

    def test_global_only_target_rejects_both_original_vqcs(self):
        from rydberg_mnist.hardware.pasqal import require_supported, UnsupportedPasqalCircuit
        require_supported(self.device, "analog_block")
        for architecture in ("digital", "digital_analog"):
            with self.assertRaises(UnsupportedPasqalCircuit):
                require_supported(self.device, architecture)

    def test_unsupported_target_never_reaches_emulator(self):
        from rydberg_mnist.hardware.pasqal import run, UnsupportedPasqalCircuit
        with tempfile.TemporaryDirectory() as directory, \
             patch("rydberg_mnist.hardware.pasqal.QutipBackendV2") as backend:
            out = Path(directory)/"run"
            with self.assertRaises(UnsupportedPasqalCircuit):
                run(ROOT/"configs/pasqal_target.json", out, architecture="digital")
            backend.assert_not_called()
            self.assertFalse(json.loads((out/"compatibility.json").read_text())["original_digital_vqc"]["supported_as_written"])

    def test_snapshot_tampering_is_rejected(self):
        from rydberg_mnist.hardware.pasqal import load_target
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/"target.json").write_text(json.dumps(self.config))
            original = ROOT/"configs"/self.config["device_json"]
            (directory/self.config["device_json"]).write_bytes(original.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_target(directory/"target.json")

    def test_compilation_respects_supplied_channel_limits(self):
        from rydberg_mnist.hardware.pasqal import compile_analog_block
        from rydberg_mnist.quantum import Physics
        from pulser.sampler import sample
        sequence = compile_analog_block(self.device, Physics(n_qubits=2))
        channel = sequence.declared_channels["rydberg"]
        values = sample(sequence).channel_samples["rydberg"]
        self.assertLessEqual(np.max(np.abs(values.amp)), channel.max_amp+1e-10)
        self.assertLessEqual(np.max(np.abs(values.det)), channel.max_abs_detuning+1e-10)
        self.assertEqual(sequence.get_duration()%channel.clock_period, 0)
        self.assertLessEqual(sequence.get_duration(), self.device.max_sequence_duration)
        coords = list(sequence.register.qubits.values())
        self.assertGreaterEqual(np.linalg.norm(coords[1]-coords[0]), self.device.min_atom_distance)

    def test_device_noise_is_explicit_and_missing_noise_does_not_fall_back(self):
        from pulser import AnalogDevice
        from rydberg_mnist.hardware.pasqal import backend_config
        noisy = backend_config(self.device, noisy=True, modulation=True)
        self.assertTrue(noisy.prefer_device_noise_model)
        self.assertTrue(noisy.with_modulation)
        with self.assertRaisesRegex(ValueError, "no supplied noise model"):
            backend_config(AnalogDevice, noisy=True, modulation=True)

    def test_pasqal_target_cannot_be_used_as_the_research_pulse_profile(self):
        from rydberg_mnist.hardware.schedule import load_profile
        with self.assertRaisesRegex(ValueError, "QoolQit/Pulser"):
            load_profile(ROOT/"configs/pasqal_target.json")


if __name__ == "__main__":
    unittest.main()
