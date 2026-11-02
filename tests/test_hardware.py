"""Physical pulse checks and energy accounting; optional QuTiP dependency."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from rydberg_mnist.hardware.energy import ledger, workload_budget
from rydberg_mnist.hardware.schedule import Compiler, Segment, load_profile, rotation_segments
from rydberg_mnist.quantum import Physics, RydbergCircuit, digital_entangler, product_states

PROFILE = Path(__file__).resolve().parents[1] / "configs/hardware_research.json"


class AccountingTests(unittest.TestCase):
    def test_ramp_area_including_short_and_negative_rotations(self):
        for angle in (.001, -.001, 1., -4.):
            segments = rotation_segments(angle, 2., .01, "01x", [0], "test", "raman_local", "rotation")
            area = sum(s.duration_us*(s.amplitude_start+s.amplitude_stop)/2 for s in segments)
            self.assertAlmostEqual(area, angle)
            for s in segments:
                self.assertLessEqual(max(abs(s.amplitude_start), abs(s.amplitude_stop)), 2.)
                self.assertLessEqual(abs(s.amplitude_stop-s.amplitude_start)/s.duration_us, 200.+1e-10)

    def test_global_beam_energy_not_multiplied_by_atom_count(self):
        p = load_profile(PROFILE)
        p["power_calibration"]["rydberg_global"] = {"bias_w": 2., "omega_squared_w": 3., "wall_efficiency": .5}
        p["overhead"] = {"preparation_us": 1., "readout_us": 2., "reset_us": 3.,
                         "background_power_w": 4., "fixed_energy_j_per_shot": 5e-6}
        s = Segment("test ramp", "analog", "rydberg_global", "0rx", (0, 1, 2, 3), 3., 0., 2.)
        report = ledger([s], p, shots=10)
        self.assertAlmostEqual(report["control_optical_j_per_shot"], 18e-6)
        self.assertAlmostEqual(report["full_quantum_cycle_j_per_shot"], 77e-6)
        self.assertAlmostEqual(report["full_quantum_cycle_j_for_requested_shots"], 770e-6)
        self.assertIsNone(report["end_to_end_pipeline_energy_j"])
        self.assertEqual(workload_budget(report, 4, 10)["physical_shot_executions"], 40)

    def test_missing_calibration_stays_unknown(self):
        p = load_profile(PROFILE)
        s = Segment("pulse", "analog", "rydberg_global", "0rx", (0, 1), 1., 2., 2.)
        report = ledger([s], p, shots=1000)
        self.assertIsNone(report["control_optical_j_per_shot"])
        self.assertIsNone(report["full_quantum_cycle_j_for_requested_shots"])
        self.assertEqual(report["channels"]["rydberg_global"]["omega_squared_integral_rad2_per_us"], 4.)

    def test_nonfinite_energy_calibration_is_rejected(self):
        p = load_profile(PROFILE)
        p["power_calibration"]["rydberg_global"] = {"bias_w": 0., "omega_squared_w": float("nan"), "wall_efficiency": 1.}
        s = Segment("pulse", "analog", "rydberg_global", "0rx", (0, 1), 1., 2., 2.)
        with self.assertRaises(ValueError):
            ledger([s], p)


@unittest.skipUnless(importlib.util.find_spec("qutip"), "Install hardware extras for pulse simulations")
class PulsePhysicsTests(unittest.TestCase):
    def dense_propagator(self, sim, segments):
        import qutip as qt
        unitary = qt.qeye([3]*sim.n)
        for s in segments:
            self.assertEqual(s.amplitude_start, s.amplitude_stop)
            h = sim.drift+s.amplitude_start*sim.operator(s.operator, s.targets)
            h -= s.detuning_start*sum(sim.nr[q] for q in s.targets)
            unitary = (-1j*s.duration_us*h).expm()*unitary
        return unitary

    def test_cnot_finite_blockade_and_phase_corrections(self):
        from rydberg_mnist.hardware.simulate import PulseSimulator, computational_indices
        p = load_profile(PROFILE); p["rise_us"] = 0.
        compiler, sim = Compiler(Physics(n_qubits=2, layers=1), p), PulseSimulator(2, p)
        u = self.dense_propagator(sim, compiler.cnot(0, 1)).full()
        idx = computational_indices(2)
        projected = u[np.ix_(idx, idx)]
        fidelity = abs(np.trace(digital_entangler(2).conj().T@projected)/4)**2
        self.assertGreater(fidelity, .99)
        self.assertLess(1-np.trace(projected.conj().T@projected).real/4, .002)
        # Removing the physical interaction must destroy the controlled operation.
        p = deepcopy(p); p["c6_rad_us_um6"] = 0.
        u = self.dense_propagator(PulseSimulator(2, p), compiler.cnot(0, 1)).full()[np.ix_(idx, idx)]
        self.assertLess(abs(np.trace(digital_entangler(2).conj().T@u)/4)**2, .5)

    def test_solver_agrees_with_exact_piecewise_unitaries(self):
        import qutip as qt
        from rydberg_mnist.hardware.simulate import PulseSimulator
        p = load_profile(PROFILE); p["rise_us"] = 0.
        sim = PulseSimulator(2, p)
        schedule = Compiler(Physics(n_qubits=2, layers=1), p).cnot(0, 1)
        initial = qt.tensor((qt.basis(3, 0)+qt.basis(3, 1)).unit(), qt.basis(3, 1))
        expected = self.dense_propagator(sim, schedule)*initial
        actual, report = sim.evolve(schedule, initial)
        self.assertGreater(abs(expected.overlap(actual))**2, 1-1e-7)
        self.assertLess(report["max_normalization_error"], 1e-7)

    def test_hybrid_ideal_fast_transfer_limit(self):
        from rydberg_mnist.hardware.simulate import PulseSimulator, computational_indices
        p = load_profile(PROFILE)
        p.update(rise_us=0., analog_rise_us=0., transfer_omega_rad_us=1e7)
        physics = Physics(n_qubits=2, layers=1)
        angles = np.array([.4, 1.1, .6, .2]); parameters = np.array([.2, -.4, .6, .7, -.2, 1.])
        sim = PulseSimulator(2, p)
        schedule = Compiler(physics, p).compile(angles, parameters, "digital_analog")
        actual = (self.dense_propagator(sim, schedule)*sim.initial()).full().ravel()[computational_indices(2)]
        ideal = RydbergCircuit(physics).forward(product_states(angles[None, :]), parameters)[0]
        self.assertGreater(abs(np.vdot(ideal, actual))**2, 1-1e-6)

    def test_lindblad_rydberg_decay_population(self):
        import qutip as qt
        from rydberg_mnist.hardware.simulate import PulseSimulator
        p = load_profile(PROFILE)
        sim = PulseSimulator(2, p, noisy=True)
        initial = qt.tensor(qt.basis(3, 2), qt.basis(3, 0))
        wait = Segment("wait", "test", "rydberg_local", "1rx", (0,), 2., 0., 0.)
        state, result = sim.evolve([wait], initial)
        expected = np.exp(-2/p["noise"]["rydberg_lifetime_us"])
        self.assertAlmostEqual(qt.expect(sim.nr[0], state), expected, places=6)
        self.assertLess(result["max_normalization_error"], 1e-7)

    def test_rydberg_erasure_is_not_postselected_away(self):
        from rydberg_mnist.hardware.simulate import PulseSimulator
        sim = PulseSimulator(2, load_profile(PROFILE))
        probability = np.zeros(9); probability[6] = 1.  # |r0>
        np.testing.assert_array_equal(sim.features(probability), [0., 1., 0.])


if __name__ == "__main__":
    unittest.main()
