"""Full-training selection, stopping, finite-shot repetition and workload accounting."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("pulser"), "Install hardware extras")
class PerformanceEnergyTests(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT/"configs/performance_energy_fresnel_beta.json").read_text())
        config["dataset_protocol"] = "custom"
        config.pop("test_exclusions", None)
        for key in ("target", "calibration", "literature_config"):
            config[key] = str(ROOT/config[key])
        config["regime"].update(n_qubits=2, input_slots=2)
        config.update(n_train=8, n_val=4, n_test=4, repeat_seeds=[7], vqc_steps=6,
            minibatch_size=4, checkpoint_every=1, shots_per_image=64, readout_slots=[1, 2], plateau_patience=2)
        return config

    def test_plateau_threshold_is_distinct_from_absolute_best_selection(self):
        from rydberg_mnist.hardware.performance_energy import Plateau
        monitor = Plateau(1., 2, .01)
        self.assertFalse(monitor.observe(.995))
        self.assertEqual(monitor.stale, 1)
        self.assertFalse(monitor.observe(.988))
        self.assertEqual(monitor.stale, 0)
        self.assertFalse(monitor.observe(.99))
        self.assertTrue(monitor.observe(.984))
        # .984 is the absolute best but less than 1% better than .988.
        self.assertEqual(monitor.reference, .988)

    def test_target_options_cannot_silently_change_primary_completion(self):
        from rydberg_mnist.hardware.performance_energy import validate_config
        config = self.config()
        validate_config(config)
        for key, value in (("accuracy_targets", [.5]), ("plateau_patience", 0),
                           ("validation_repetitions", True), ("plateau_relative_improvement", 0)):
            invalid = dict(config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_chunked_jobs_pool_counts_and_charge_reported_retries_and_attempts(self):
        from rydberg_mnist.hardware.finite_backend import CallableCountsBackend, ExecutionCounts, FeatureEngine
        from rydberg_mnist.hardware.calibrated_cost import budget
        target = json.loads((ROOT/"configs/fresnel_beta_target.json").read_text())
        compiler = SimpleNamespace(config={"n_qubits": 2, "input_slots": 1}, target=target,
            max_shots_per_job=9, device=SimpleNamespace(max_runs=3), sequence=lambda *args: {"same_program": True},
            account_sequence=lambda seq: {"control_us": 1., "serialized_program": json.dumps(seq)})
        calls = []
        def execute(**kw):
            calls.append(kw)
            return ExecutionCounts({"00": kw["shots"]} if len(calls) == 1 else {"11": kw["shots"]},
                attempted_shots=kw["shots"]+1, submitted_jobs=2, source_record="synthetic retry fixture")
        engine = FeatureEngine(compiler, CallableCountsBackend(execute, target["device_sha256"]), [1], 8)
        result = engine.features(np.zeros((1, 2)), np.zeros(2), [0], algorithm="qrc", stage="train", checkpoint=0, seed=4)
        self.assertEqual([c["shots"] for c in calls], [3, 3, 2])
        self.assertEqual(len({c["seed"] for c in calls}), 3)
        np.testing.assert_allclose(result, [[-.25, -.25, 1.]])
        self.assertEqual(len({r["program_sha256"] for r in engine.records}), 1)
        self.assertEqual([r["job_index"] for r in engine.records], [0, 1, 2])
        calibration = json.loads((ROOT/"configs/fresnel_beta_calibration.json").read_text())
        calibration.update(calibration_status="measured", electrical_power_w=2., full_cycle_s_per_attempt=3.,
            attempts_per_returned_shot=1.5, setup_energy_j_per_program=.5)
        calibration["provenance"].update(measured_at_utc="2026-11-03T00:00:00Z",
            instrument_or_provider_record="synthetic fixture", measurement_scope="whole-machine fixture")
        literature = json.loads((ROOT/"configs/literature_energy_fresnel_beta.json").read_text())
        value = budget(engine.records, calibration, target, literature, classical_energy_j=7.)
        self.assertEqual(value["submitted_jobs_including_reported_retries"], 6)
        self.assertEqual(value["calibrated_shot_execution_energy_j"], 11*6.)
        self.assertEqual(value["calibrated_setup_energy_j"], 3.)
        self.assertEqual(value["complete_workflow_energy_j"], 76.)

    def test_account_access_alone_does_not_verify_spec_and_export_obeys_shot_cap(self):
        from rydberg_mnist.hardware.target_profile import AnalogProfile
        config = self.config()
        target_path = Path(config["target"])
        target = json.loads(target_path.read_text())
        target.update(account_access_verified=True, device_json=str(target_path.parent/target["device_json"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"target.json"; path.write_text(json.dumps(target))
            profile = AnalogProfile.build(path, config["regime"])
            self.assertFalse(profile.metadata["account_specific_spec_verified"])
            self.assertFalse(profile.metadata["hardware_submission_verified"])
            sequence = profile.sequence(np.zeros(4), np.zeros(4), 2)
            with self.assertRaisesRegex(ValueError, "shot limit"):
                profile.export_example(sequence, Path(directory)/"invalid.json", profile.device.max_runs+1)
            self.assertFalse((Path(directory)/"invalid.json").exists())

    def test_full_training_survives_early_selection_and_reaccounts_without_rerun(self):
        from rydberg_mnist.hardware.performance_energy import run
        from rydberg_mnist.hardware.accuracy_energy import predict_saved
        from rydberg_mnist.hardware.finite_backend import CallableCountsBackend
        from rydberg_mnist.hardware.reaccount import recompute, main as reaccount_main
        rng = np.random.default_rng(10)
        dataset = (rng.integers(0, 256, (16, 784), dtype=np.uint8), np.tile([3, 8], 8),
            rng.integers(0, 256, (4, 784), dtype=np.uint8), np.array([3, 8, 3, 8]))
        def factory(profile, seed):
            return CallableCountsBackend(lambda **kw: {"00": kw["shots"]}, profile.target["device_sha256"])
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), redirect_stdout(io.StringIO()), \
             patch("rydberg_mnist.hardware.performance_energy.load_mnist", return_value=dataset):
            for cap, patience, reason, expected_steps in ((6, 2, "validation_loss_plateau", 2), (1, 6, "budget_limited", 1)):
                config = self.config(); config.update(vqc_steps=cap, plateau_patience=patience)
                output = Path(directory)/f"run{cap}"
                report = run(config, output, backend_factory=factory)
                rep = report["repeats"][0]
                qrc, vqc = [rep["algorithms"][a] for a in ("qrc", "vqc")]
                self.assertEqual(vqc["completion"]["reason"], reason)
                self.assertEqual(vqc["completion"]["optimizer_steps"], expected_steps)
                self.assertFalse(vqc["completion"]["global_convergence_claimed"])
                self.assertEqual(vqc["selected_checkpoint"]["iteration"], 0)
                self.assertEqual(vqc["terminal_checkpoint"]["iteration"], expected_steps)
                train_shots = (expected_steps+1)*(8+3*4)*64 + expected_steps*2*4*64
                self.assertEqual(vqc["training_budget"]["requested_returned_shots"], train_shots)
                self.assertEqual(qrc["training_budget"]["requested_returned_shots"], 20*64)
                self.assertEqual(qrc["complete_study_execution_budget"]["requested_returned_shots"], 24*64)
                self.assertEqual(vqc["complete_study_execution_budget"]["requested_returned_shots"], train_shots+2*4*64)
                self.assertEqual(vqc["training_plus_selected_inference_budget"]["requested_returned_shots"], train_shots+4*64)
                self.assertIsNone(vqc["training_budget"]["complete_workflow_energy_j"])
                with (output/"seed7/execution_ledger.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                validation = [r for r in rows if r["algorithm"] == "qrc" and r["stage"] == "validation"]
                self.assertEqual(len(validation), 3*4*2)
                fixed = [r for r in validation if r["source_index"] == validation[0]["source_index"] and r["prefix"] == "1"]
                self.assertEqual(len({r["noise_seed"] for r in fixed}), 3)
                archive = output/"seed7/models_features.npz"
                with np.load(archive) as saved:
                    for step in range(expected_steps+1):
                        for suffix in ("parameters", "weights", "bias", "feature_mean", "feature_scale", "validation_features"):
                            self.assertIn(f"vqc_step{step}_{suffix}", saved)
                with np.load(output/"seed7/splits_predictions.npz") as splits:
                    predicted = predict_saved(archive, dataset[2][splits["test_idx"]], backend_factory=factory)
                    np.testing.assert_array_equal(predicted, splits["qrc_step0"])
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                target = report["target"]
                self.assertEqual(hashlib.sha256((output/"device_snapshot.json").read_bytes()).hexdigest(), target["device_sha256"])
                self.assertFalse(report["device_profile"]["hardware_submission_verified"])
                calibration = deepcopy(report["calibration"])
                calibration.update(calibration_status="measured", electrical_power_w=2., full_cycle_s_per_attempt=3.,
                    attempts_per_returned_shot=1., setup_energy_j_per_program=0.)
                calibration["provenance"].update(measured_at_utc="2026-11-03T00:00:00Z",
                    instrument_or_provider_record="synthetic reaccount fixture", measurement_scope="whole-machine fixture")
                evidence = {"device_sha256": target["device_sha256"], "workloads": {
                    f"seed7/vqc/{key}": {"energy_j": joules, "measured_at_utc": "2026-11-03T00:00:00Z",
                        "source_record": "synthetic external meter", "scope": key}
                    for key, joules in (("all_training", 3.), ("test_step0", 2.))}}
                updated = recompute(deepcopy(report), output, calibration, evidence)
                value = updated["repeats"][0]["algorithms"]["vqc"]["training_plus_selected_inference_budget"]
                self.assertEqual(value["complete_workflow_energy_j"], (train_shots+4*64)*6+5.)
                self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), digest)
                # CLI dispatch supports both report formats without invoking the backend.
                cal_path = Path(directory)/"cal.json"; cal_path.write_text(json.dumps(calibration))
                args = ["reaccount", "--source", str(output), "--calibration", str(cal_path), "--output", str(output)+"_reaccounted"]
                with patch("sys.argv", args), patch("rydberg_mnist.hardware.performance_energy.run", side_effect=AssertionError("no training")):
                    reaccount_main()
                saved_report = json.loads(Path(str(output)+"_reaccounted/report.json").read_text())
                item = saved_report["repeats"][0]["algorithms"]["vqc"]
                self.assertEqual(item["selected_checkpoint"]["training_budget_through_checkpoint"]["calibration_status"], "measured")
                self.assertEqual(item["terminal_checkpoint"]["training_budget_through_checkpoint"]["requested_returned_shots"], train_shots)


if __name__ == "__main__":
    unittest.main()
