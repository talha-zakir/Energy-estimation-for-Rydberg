"""Device validation, measured-energy boundaries and target-selection correctness."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from rydberg_mnist.hardware.calibrated_cost import budget, integrate_power_trace, validate_calibration

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = json.loads((ROOT/"configs/fresnel_beta_calibration.json").read_text())
TARGET = json.loads((ROOT/"configs/fresnel_beta_target.json").read_text())
LITERATURE = json.loads((ROOT/"configs/literature_energy_fresnel_beta.json").read_text())


class CalibratedEnergyTests(unittest.TestCase):
    def test_unknown_energy_does_not_become_zero_or_a_literature_value(self):
        records = [{"shots": 10, "control_us": 1.}]
        value = budget(records, CALIBRATION, TARGET, LITERATURE)
        self.assertIsNone(value["calibrated_shot_execution_energy_j"])
        self.assertIsNone(value["complete_workflow_energy_j"])
        self.assertEqual(value["requested_returned_shots"], 10)
        self.assertGreater(value["literature_scenarios"]["harvard_paper_reference"]["electrical_j"], 0)

    def test_full_measured_cycle_counts_attempts_setup_and_classical_once(self):
        calibration = deepcopy(CALIBRATION)
        calibration.update(calibration_status="measured", electrical_power_w=2., full_cycle_s_per_attempt=3.,
            attempts_per_returned_shot=1.2, setup_energy_j_per_program=.5)
        calibration["provenance"].update(measured_at_utc="2026-11-03T00:00:00Z",
            instrument_or_provider_record="synthetic unit-test fixture", measurement_scope="whole-machine synthetic budget")
        records = [{"shots": 10, "control_us": 1.}, {"shots": 20, "control_us": 2.}]
        value = budget(records, calibration, TARGET, LITERATURE, classical_energy_j=7.)
        self.assertAlmostEqual(value["calibrated_shot_execution_energy_j"], 216.)
        self.assertAlmostEqual(value["calibrated_setup_energy_j"], 1.)
        self.assertAlmostEqual(value["complete_workflow_energy_j"], 224.)
        # Without this workload's external classical meter result, completeness remains unknown.
        self.assertIsNone(budget(records, calibration, TARGET, LITERATURE)["complete_workflow_energy_j"])

    def test_calibration_requires_identity_and_dated_evidence(self):
        wrong = deepcopy(CALIBRATION); wrong["device_sha256"] = "wrong-device"
        with self.assertRaisesRegex(ValueError, "device_sha256"):
            validate_calibration(wrong, TARGET)
        unsupported = deepcopy(CALIBRATION); unsupported["calibration_status"] = "measured"
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_calibration(unsupported, TARGET)
        for key, number in (("electrical_power_w", float("nan")), ("full_cycle_s_per_attempt", -1),
                            ("attempts_per_returned_shot", .9)):
            invalid = deepcopy(CALIBRATION); invalid[key] = number
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_calibration(invalid, TARGET)

    def test_nominal_beta_rate_is_separate_and_does_not_double_count_control(self):
        records = [{"shots": 10, "control_us": 1000.}]
        value = budget(records, CALIBRATION, TARGET, LITERATURE)
        scenario = value["literature_scenarios"]["nominal_beta_throughput__pasqal_historical_power_transfer"]
        self.assertAlmostEqual(scenario["electrical_j"], 2600*4*10)
        self.assertIsNone(value["calibrated_shot_execution_energy_j"])

    def test_power_trace_integrates_physical_units_and_checks_time_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.csv"
            path.write_text("time_s,power_w\n0,0\n1,2\n2,0\n")
            self.assertEqual(integrate_power_trace(path)["energy_j"], 2.)
            path.write_text("time_s,power_w\n0,1\n0,1\n")
            with self.assertRaises(ValueError):
                integrate_power_trace(path)

    def test_metered_program_energy_replaces_modeled_execution_without_double_count(self):
        records = [{"shots": 10, "control_us": 1., "measured_whole_program_electrical_j": 4.,
                    "measurement_source_record": "synthetic meter unit fixture"}]
        value = budget(records, CALIBRATION, TARGET, LITERATURE, classical_energy_j=7.)
        self.assertEqual(value["measured_program_electrical_energy_j"], 4.)
        self.assertEqual(value["complete_measured_workflow_energy_j"], 11.)
        self.assertIsNone(value["complete_workflow_energy_j"])
        self.assertTrue(value["hardware_execution_energy_measured_here"])
        del records[0]["measurement_source_record"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            budget(records, CALIBRATION, TARGET, LITERATURE)

    def test_setup_cannot_be_counted_twice(self):
        calibration = deepcopy(CALIBRATION)
        calibration.update(cycle_includes_program_setup=True, setup_energy_j_per_program=1.)
        with self.assertRaisesRegex(ValueError, "setup"):
            validate_calibration(calibration, TARGET)


@unittest.skipUnless(importlib.util.find_spec("pulser"), "Install hardware extras for finite-shot protocol")
class AccuracyEnergyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rydberg_mnist.hardware.target_profile import AnalogProfile
        cls.config = json.loads((ROOT/"configs/accuracy_energy_fresnel_beta.json").read_text())
        cls.config.update(target=str(ROOT/cls.config["target"]), calibration=str(ROOT/cls.config["calibration"]),
                          literature_config=str(ROOT/cls.config["literature_config"]))
        cls.config["regime"].update(n_qubits=2, input_slots=2)
        cls.profile = AnalogProfile.build(cls.config["target"], cls.config["regime"])

    def test_target_selection_never_uses_test_scores_and_preserves_failure(self):
        from rydberg_mnist.hardware.accuracy_energy import first_target_hit
        checkpoints = [{"validation_accuracy": .6, "test_accuracy": 1., "iteration": 0},
                       {"validation_accuracy": .8, "test_accuracy": 0., "iteration": 4}]
        self.assertEqual(first_target_hit(checkpoints, .75)["iteration"], 4)
        self.assertIsNone(first_target_hit(checkpoints, .9))
        self.assertEqual(first_target_hit(checkpoints, .5)["iteration"], 0)

    def test_both_linear_heads_use_training_only_statistics(self):
        from rydberg_mnist.hardware.accuracy_energy import LinearHead
        train = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        validation = np.array([[100., 100.], [200., 200.]])
        y = np.eye(2)[[0, 0, 1, 1]]
        head = LinearHead.fit(train, validation, y, np.eye(2), [.1, 1.])
        np.testing.assert_array_equal(head.mean, [1.5, 1.5])
        np.testing.assert_array_equal(head.scale, train.std(0))
        head2 = LinearHead.fit(train, validation, y, np.eye(2), [.1, 1.])
        np.testing.assert_array_equal(head.predict(validation), head2.predict(validation))

    def test_balanced_regime_and_qpu_layout_checks(self):
        from pulser import QPUBackend
        from pulser.sampler import sample
        profile = self.profile
        self.assertTrue(profile.register.layout is not None)
        self.assertGreaterEqual(profile.register.layout.number_of_traps, profile.device.min_layout_traps)
        np.testing.assert_allclose(profile.metadata["interaction_to_mixing_actual"], 1., atol=1e-5)
        for value in (-3., 0., 3.):
            sequence = profile.sequence(np.full(4, value), np.full(4, value), 2)
            QPUBackend.validate_sequence(sequence, mimic_qpu=True)
            channel = sequence.declared_channels["reservoir"]
            values = sample(sequence).channel_samples["reservoir"]
            self.assertLessEqual(np.max(values.amp), channel.max_amp)
            self.assertLessEqual(np.max(np.abs(values.det)), channel.max_abs_detuning)
            self.assertEqual(sequence.get_duration()%channel.clock_period, 0)

    def test_future_input_cannot_change_an_early_readout(self):
        from pulser.sampler import sample
        a = np.zeros(4); b = a.copy(); b[[1, 3]] = 3.
        left = sample(self.profile.sequence(a, np.zeros(4), 1)).channel_samples["reservoir"]
        right = sample(self.profile.sequence(b, np.zeros(4), 1)).channel_samples["reservoir"]
        np.testing.assert_array_equal(left.amp, right.amp)
        np.testing.assert_array_equal(left.det, right.det)

    def test_joint_counts_adapter_rejects_invalid_results_and_wrong_target(self):
        from rydberg_mnist.hardware.finite_backend import CallableCountsBackend, FeatureEngine
        backend = CallableCountsBackend(lambda **kw: {"00": kw["shots"]-1}, TARGET["device_sha256"])
        engine = FeatureEngine(self.profile, backend, [1, 2], 64)
        with self.assertRaisesRegex(ValueError, "counts"):
            engine.features(np.zeros((1, 4)), np.zeros(4), [0], algorithm="qrc", stage="train", checkpoint=0, seed=1)
        backend.device_sha256 = "different"
        with self.assertRaisesRegex(ValueError, "snapshot"):
            FeatureEngine(self.profile, backend, [1, 2], 64)

    def test_provider_adapter_preserves_meter_evidence(self):
        from rydberg_mnist.hardware.finite_backend import CallableCountsBackend, ExecutionCounts, FeatureEngine
        backend = CallableCountsBackend(lambda **kw: ExecutionCounts({"00": kw["shots"]},
            attempted_shots=kw["shots"]+2, active_duration_s=3., whole_program_electrical_j=6.,
            source_record="synthetic execution/meter fixture"), TARGET["device_sha256"])
        engine = FeatureEngine(self.profile, backend, [1, 2], 64)
        engine.features(np.zeros((1, 4)), np.zeros(4), [0], algorithm="qrc", stage="train", checkpoint=0, seed=1)
        self.assertEqual(sum(r["measured_whole_program_electrical_j"] for r in engine.records), 12.)
        self.assertTrue(all(r["measured_attempted_shots"] == 34 for r in engine.records))

    def test_feature_engine_accepts_a_non_pulser_compiler_adapter(self):
        from types import SimpleNamespace
        from rydberg_mnist.hardware.finite_backend import CallableCountsBackend, FeatureEngine
        compiler = SimpleNamespace(config={"n_qubits": 2, "input_slots": 2},
            target={"device_sha256": "synthetic-other-device"},
            sequence=lambda x, theta, prefix: {"prefix": prefix},
            account_sequence=lambda program: {"control_us": .2, "serialized_program": json.dumps(program)})
        backend = CallableCountsBackend(lambda **kw: {"00": kw["shots"]}, "synthetic-other-device")
        engine = FeatureEngine(compiler, backend, [1, 2], 64)
        features = engine.features(np.zeros((1, 4)), np.zeros(4), [0], algorithm="qrc", stage="train", checkpoint=0, seed=1)
        np.testing.assert_array_equal(features, np.ones((1, 6)))
        self.assertIsNone(engine.records[0]["omega_squared_integral_rad2_per_us"])

    def test_repeated_pipeline_censors_targets_and_accounts_real_request_counts(self):
        from rydberg_mnist.hardware.accuracy_energy import run, predict_saved
        from rydberg_mnist.hardware.finite_backend import CallableCountsBackend
        rng = np.random.default_rng(10)
        dataset = (rng.integers(0, 256, (16, 784), dtype=np.uint8), np.tile([3, 8], 8),
                   rng.integers(0, 256, (4, 784), dtype=np.uint8), np.array([3, 8, 3, 8]))
        config = deepcopy(self.config)
        config.update(n_train=8, n_val=4, n_test=4, repeat_seeds=[7, 11], vqc_steps=1, minibatch_size=4,
                      checkpoint_every=1, shots_per_image=64, readout_slots=[1, 2],
                      accuracy_targets=[.5, 1.], early_stop_all_targets=False)
        # Synthetic count fixtures test orchestration only, never task accuracy.
        def factory(profile, seed):
            return CallableCountsBackend(lambda **kw: {"00": kw["shots"]//2, "11": kw["shots"]-kw["shots"]//2},
                                         profile.target["device_sha256"])
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), redirect_stdout(io.StringIO()), \
             patch("rydberg_mnist.hardware.accuracy_energy.load_mnist", return_value=dataset):
            output = Path(directory)/"run"
            report = run(config, output, backend_factory=factory)
            self.assertEqual(len(report["repeats"]), 2)
            for repetition in report["repeats"]:
                target = repetition["targets"]["0.5"]
                self.assertTrue(target["paired_target_reached"])
                for algorithm in ("qrc", "vqc"):
                    value = target["algorithms"][algorithm]
                    self.assertEqual(value["checkpoint"]["iteration"], 0)
                    self.assertEqual(value["energy_to_validation_target"]["requested_returned_shots"], 12*64)
                    self.assertIsNone(value["energy_to_validation_target"]["complete_workflow_energy_j"])
                    self.assertEqual(repetition["targets"]["1.0"]["algorithms"][algorithm]["status"], "not_reached_within_budget")
                with (output/f"seed{repetition['seed']}"/"execution_ledger.csv").open() as handle:
                    ledger = list(csv.DictReader(handle))
                self.assertTrue(any(row["stage"] == "spsa_objective" for row in ledger))
                self.assertTrue(all(int(row["shots"]) == 32 for row in ledger))
                self.assertEqual(repetition["feature_count_each"], 6)
                folder = output/f"seed{repetition['seed']}"
                with np.load(folder/"splits_predictions.npz") as splits:
                    predicted = predict_saved(folder/"models_features.npz", dataset[2][splits["test_idx"]],
                                              backend_factory=factory)
                    np.testing.assert_array_equal(predicted, splits["qrc_step0"])
            self.assertEqual(report["target_aggregate"]["1.0"]["paired_validation_target_hits"], 0)
            # Updated measured coefficients can be applied without executing another circuit.
            from rydberg_mnist.hardware.reaccount import recompute
            measured = deepcopy(CALIBRATION)
            measured.update(calibration_status="measured", electrical_power_w=2., full_cycle_s_per_attempt=3.,
                attempts_per_returned_shot=1., setup_energy_j_per_program=0.)
            measured["provenance"].update(measured_at_utc="2026-11-03T00:00:00Z",
                instrument_or_provider_record="synthetic reaccounting test", measurement_scope="whole-machine fixture")
            evidence = {"device_sha256": TARGET["device_sha256"], "workloads": {
                "seed7/qrc/train_to_step0": {"energy_j": 3., "measured_at_utc": "2026-11-03T00:00:00Z",
                    "source_record": "unit fixture", "scope": "external classical training"},
                "seed7/qrc/test_step0": {"energy_j": 2., "measured_at_utc": "2026-11-03T00:00:00Z",
                    "source_record": "unit fixture", "scope": "external classical inference"}}}
            updated = recompute(report, output, measured, evidence)
            value = updated["repeats"][0]["targets"]["0.5"]["algorithms"]["qrc"]
            self.assertEqual(value["training_plus_held_out_inference_energy"]["complete_workflow_energy_j"], 6149.)


if __name__ == "__main__":
    unittest.main()
