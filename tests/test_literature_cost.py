"""Independent published budgets, physical units, and circuit-count checks."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from rydberg_mnist.hardware.literature_cost import (
    Controls, component_power, native_training_count, paper_runtime,
    reproduce_alves, system_scenario, transferred_radiation,
)
from rydberg_mnist.hardware.schedule import Segment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/"configs/literature_energy.json").read_text())


class LiteratureEnergyTests(unittest.TestCase):
    def test_reproduces_published_rydberg_radiation_budget(self):
        value = reproduce_alves(CONFIG["alves_reference"])
        self.assertAlmostEqual(value["computation_j_per_execution"]*1000, .6623104426)
        self.assertAlmostEqual(value["trapping_j_per_execution"], .098441)
        # Independent rounded results in Alves et al. IV.1–IV.2.
        self.assertAlmostEqual(value["computation_j_per_execution"]*1000, .662, delta=.0005)
        self.assertAlmostEqual(value["total_radiation_j"], 69.5, delta=.05)

    def test_full_electrical_inventory_excludes_optional_reload_hardware(self):
        self.assertEqual(component_power(CONFIG["carrasco_reference"]), 22205)

    def test_equation18_retains_transport_and_amortized_reload(self):
        ref = CONFIG["carrasco_reference"]
        # 20 ms overhead + D*(.5 ms clock + .125 ms transport + .5/2400 s reload).
        for depth, expected in ((10, .0283333333333), (100, .1033333333333), (1000, .8533333333333)):
            self.assertAlmostEqual(paper_runtime(depth, ref), expected)
        without_transport = deepcopy(ref); without_transport["transport_per_layer"] = 0
        self.assertAlmostEqual(paper_runtime(100, ref)-paper_runtime(100, without_transport), .0125)

    def test_electrical_units_shots_and_tariff(self):
        value = system_scenario(1e-6, 2600, CONFIG["carrasco_reference"],
                    CONFIG["transferred_scenarios"]["schedule_timing"], shots=1000, tariff=.25)
        self.assertAlmostEqual(value["electrical_j_per_shot"], 52.0026)
        self.assertAlmostEqual(value["electrical_j_per_prediction"], 52002.6)
        self.assertAlmostEqual(value["kwh_per_prediction"], .0144451666666667)
        self.assertAlmostEqual(value["electricity_cost_per_prediction"], .00361129166666667)
        self.assertIsNone(value["end_to_end_pipeline_energy_j"])
        self.assertIsNone(value["provider_bill"])

    def test_explicit_motion_reload_added_once(self):
        timing = {"transport_events": 2, "reconfiguration_layers": 3, "reloads_per_shot": .1}
        value = system_scenario(1e-6, 2600, CONFIG["carrasco_reference"], timing, shots=1)
        self.assertAlmostEqual(value["extra_motion_reload_s_per_shot"], .0525)
        self.assertAlmostEqual(value["cycle_s_per_shot"], .072501)
        self.assertIsNone(value["electricity_cost_per_prediction"])

    def test_reference_shared_ir_once_and_assumed_blue_per_atom(self):
        segment = Segment("four-site", "analog", "rydberg_global", "0rx", (0, 1, 2, 3), 1., 1., 1.)
        controls = Controls.from_schedule([segment])
        self.assertEqual(controls.rydberg_blue_target_s, 4e-6)
        self.assertEqual(controls.rydberg_infrared_beam_s, 1e-6)
        source = transferred_radiation(controls, CONFIG["alves_reference"], 49)
        self.assertAlmostEqual(source["control_radiation_j"], 11.4e-6)
        self.assertIsNone(source["wall_plug_energy_j"])
        # Source radiation is not added to an already inclusive electrical envelope.
        wall = system_scenario(1e-6, 2600, CONFIG["carrasco_reference"],
                    CONFIG["transferred_scenarios"]["schedule_timing"], shots=1)
        self.assertAlmostEqual(wall["electrical_j_per_shot"], 52.0026)

    def test_native_feature_budget_agrees_with_recorded_training_and_scoring(self):
        saved = json.loads((ROOT/"hardware_results/pasqal_native_mnist_3_8_seed7/report.json").read_text())
        counts = native_training_count(saved["config"])
        # Archived experiment emulated training plus frozen/clean/noisy test inputs.
        self.assertEqual(sum(counts.values()), 2500)
        self.assertEqual(sum(counts.values())+3*saved["config"]["n_test"], saved["compiled_emulated_sequences"])

    def test_invalid_time_power_and_shots_are_rejected(self):
        def call(**kw):
            values = {"control_s": 1e-6, "power_w": 2600, "shots": 1000}
            values.update(kw)
            return system_scenario(timing_reference=CONFIG["carrasco_reference"],
                timing_assumptions=CONFIG["transferred_scenarios"]["schedule_timing"], **values)
        for values in ({"control_s": -1}, {"control_s": float("inf")}, {"power_w": float("nan")},
                       {"shots": 0}, {"shots": 1.5}, {"shots": True}, {"tariff": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                call(**values)


if __name__ == "__main__":
    unittest.main()
