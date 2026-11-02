import unittest
from unittest.mock import patch
from itertools import combinations

import numpy as np
from scipy.linalg import expm

from rydberg_mnist.data import split_indices
from rydberg_mnist.quantum import (Physics, RydbergCircuit, encode_angles,
                                  product_states, hamiltonian, apply_gate, rotation,
                                  digital_entangler)
from rydberg_mnist.readout import fit_readout, projected_loss


class PhysicsTests(unittest.TestCase):
    def test_hamiltonian_independent_kronecker_construction(self):
        p = Physics(n_qubits=3)
        x, number, identity = np.array([[0, 1], [1, 0]]), np.diag([0, 1]), np.eye(2)
        def op(operators):
            out = np.ones((1, 1))
            for a in operators:
                out = np.kron(out, a)
            return out
        h = np.zeros((8, 8))
        for q in range(3):
            h += p.omega_rad_us / 2 * op([x if j == q else identity for j in range(3)])
            h -= p.omega_rad_us * p.delta_over_omega * op([number if j == q else identity for j in range(3)])
        for a, b in combinations(range(3), 2):
            h += p.c6_rad_us_um6 / (p.spacing_um * (b-a))**6 * op([
                number if j in (a, b) else identity for j in range(3)])
        np.testing.assert_allclose(hamiltonian(p), h, atol=1e-12)

    def test_single_qubit_rabi_limit(self):
        # Very large spacing -> no interactions; Delta=0 -> independent Rabi oscillations.
        p = Physics(n_qubits=2, layers=1, delta_over_omega=0, rb_over_a=1e-3,
                    evolution_us=0.03125)
        circuit = RydbergCircuit(p)
        initial = np.array([[1, 0, 0, 0]], dtype=complex)
        states = circuit.forward(initial, np.zeros(circuit.n_params))
        expected = np.cos(p.omega_rad_us * p.evolution_us)
        np.testing.assert_allclose(circuit.features(states)[0, :2], expected, atol=1e-12)

    def test_unitarity_and_norm(self):
        c = RydbergCircuit(Physics(n_qubits=3, layers=2))
        np.testing.assert_allclose(c.unitary.conj().T @ c.unitary, np.eye(8), atol=1e-12)
        rng = np.random.default_rng(3)
        inputs = product_states(rng.uniform(0, np.pi, (5, 6)))
        result = c.forward(inputs, rng.normal(size=c.n_params))
        np.testing.assert_allclose(np.sum(abs(result)**2, axis=1), 1, atol=1e-12)

    def test_gate_endianness(self):
        state = np.array([[1, 0, 0, 0]], dtype=complex)
        np.testing.assert_allclose(abs(apply_gate(state, rotation(np.pi, "x"), 0, 2)),
                                   [[0, 0, 1, 0]], atol=1e-12)

    def test_shared_shot_correlations(self):
        c = RydbergCircuit(Physics(n_qubits=2))
        bell = np.array([[1, 0, 0, 1]], dtype=complex) / np.sqrt(2)
        features = c.features(bell, shots=20000, rng=np.random.default_rng(5))
        self.assertLess(abs(features[0, 0]), .03)
        self.assertEqual(features[0, 0], features[0, 1])
        self.assertAlmostEqual(features[0, 2], 1)

    def test_encoding_and_constant_guard(self):
        values = np.array([[1, 2, 3, 4], [1, 1, 2, 2]], dtype=float)
        angles = encode_angles(values)
        np.testing.assert_allclose(angles, [[0, np.pi, 0, np.pi], [0, 0, 0, 0]])
        np.testing.assert_allclose(abs(product_states(angles)[0]), [0, 1, 0, 0], atol=1e-12)


class LearningTests(unittest.TestCase):
    def test_ordinary_least_squares(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(40, 5))
        w, b = rng.normal(size=(5, 3)), rng.normal(size=3)
        fitted_w, fitted_b = fit_readout(x, x @ w + b, alpha=0)
        np.testing.assert_allclose(fitted_w, w, atol=1e-12)
        np.testing.assert_allclose(fitted_b, b, atol=1e-12)

    def test_projected_gradient_finite_difference(self):
        for architecture in ("digital_analog", "digital"):
            with self.subTest(architecture=architecture):
                self.check_projected_gradient(architecture)

    def check_projected_gradient(self, architecture):
        rng = np.random.default_rng(1)
        c = RydbergCircuit(Physics(n_qubits=3, layers=2), architecture)
        states = product_states(rng.uniform(0, np.pi, (13, 6)))
        labels = np.eye(3)[rng.integers(0, 3, 13)]
        theta = rng.normal(size=c.n_params)
        final, tape = c.forward(states, theta, tape=True)
        _, df, _, _ = projected_loss(c.features(final), labels, alpha=.1)
        analytic = c.backward(final, tape, df)
        numeric = np.zeros_like(theta)
        for k in range(len(theta)):
            shift = np.zeros_like(theta); shift[k] = 1e-6
            plus = projected_loss(c.features(c.forward(states, theta + shift)), labels, .1)[0]
            minus = projected_loss(c.features(c.forward(states, theta - shift)), labels, .1)[0]
            numeric[k] = (plus-minus) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-5)

    def test_split_reproducibility_and_no_leakage(self):
        labels = np.tile(np.arange(10), 100)
        a = split_indices(labels, labels, [3, 8], 80, 20, 40, 7)
        b = split_indices(labels, labels, [3, 8], 80, 20, 40, 7)
        self.assertFalse(set(a[0]) & set(a[1]))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(set(labels[x]), {3, 8})


class DigitalTests(unittest.TestCase):
    def test_cnot_truth_table_and_direction(self):
        # 0->1 followed by 1->2, with q0 the most significant bit.
        expected_destination = [0, 1, 3, 2, 7, 6, 4, 5]
        u = digital_entangler(3)
        np.testing.assert_array_equal(u.argmax(axis=0), expected_destination)
        np.testing.assert_array_equal(u.conj().T @ u, np.eye(8))

    def test_even_then_odd_bond_order(self):
        # (0->1, 2->3) then (1->2): |1000> becomes |1110>, not |1111>.
        u = digital_entangler(4)
        self.assertEqual(np.argmax(u[:, 8]), 14)
        # An entangler with overlapping CNOTs is generally NOT its own inverse.
        self.assertFalse(np.array_equal(u, u.conj().T))

    def test_digital_does_not_build_analog_evolution(self):
        with patch("rydberg_mnist.quantum.hamiltonian", side_effect=AssertionError("analog H used")), \
             patch("rydberg_mnist.quantum.expm", side_effect=AssertionError("analog evolution used")):
            circuit = RydbergCircuit(Physics(), "digital")
        counts = circuit.describe()
        self.assertEqual(counts["rotation_gates_per_circuit"], 48)
        self.assertEqual(counts["cnot_gates_per_circuit"], 14)
        self.assertEqual(counts["analog_blocks_per_circuit"], 0)
        self.assertEqual(counts["analog_evolution_us_per_circuit"], 0)
        self.assertIsNone(counts["hardware_execution_time_us"])

    def test_digital_norm_and_adjoint(self):
        rng = np.random.default_rng(12)
        circuit = RydbergCircuit(Physics(n_qubits=4, layers=2), "digital")
        states = product_states(rng.uniform(0, np.pi, (7, 8)))
        parameters = rng.normal(size=circuit.n_params)
        result = circuit.forward(states, parameters)
        np.testing.assert_allclose(np.sum(abs(result)**2, axis=1), 1, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
