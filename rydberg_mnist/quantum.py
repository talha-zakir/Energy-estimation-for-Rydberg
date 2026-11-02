"""Exact batched statevectors, big-endian qubit indexing, rad/us and micrometres.

Physics and RX-RZ-RX layout follow Lu et al., arXiv:2401.02940v2 and
https://github.com/jz-lu/daqml. This is an independent Python implementation.
"""
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class Physics:
    n_qubits: int = 8
    layers: int = 2
    omega_rad_us: float = 2 * np.pi * 4
    delta_over_omega: float = 0.8
    rb_over_a: float = 0.87
    c6_rad_us_um6: float = 2 * np.pi * 862690
    evolution_us: float = 0.25

    def __post_init__(self):
        if not 2 <= self.n_qubits <= 10 or self.layers < 1:
            raise ValueError("CPU simulator supports 2..10 qubits and >=1 layers")
        if min(self.omega_rad_us, self.c6_rad_us_um6, self.rb_over_a) <= 0:
            raise ValueError("Omega, C6 and Rb/a must be positive")
        if self.evolution_us < 0:
            raise ValueError("Evolution time must be nonnegative")

    @property
    def spacing_um(self):
        return (self.c6_rad_us_um6 / self.omega_rad_us) ** (1 / 6) / self.rb_over_a

    def describe(self):
        return {**asdict(self), "spacing_um": self.spacing_um,
                "analog_evolution_us_per_circuit": self.layers * self.evolution_us}


def bit_table(n):
    return ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)


def hamiltonian(physics):
    """H/hbar = Omega/2 sum X - Delta sum n + sum C6/r^6 n_i n_j."""
    n = physics.n_qubits
    bits = bit_table(n)
    diagonal = -physics.delta_over_omega * physics.omega_rad_us * bits.sum(axis=1)
    for i, j in combinations(range(n), 2):
        interaction = physics.c6_rad_us_um6 / (physics.spacing_um * (j - i))**6
        diagonal += interaction * bits[:, i] * bits[:, j]
    h = np.diag(diagonal)
    rows = np.arange(2**n)
    for q in range(n):
        h[rows, rows ^ (1 << (n - 1 - q))] += physics.omega_rad_us / 2
    return h


def encode_angles(pca_values):
    """Match daqml prep_full.ipynb: each image's theta/phi halves -> [0,pi]."""
    if pca_values.ndim != 2 or pca_values.shape[1] % 2:
        raise ValueError("Expected an even number of PCA features")
    halves = np.split(pca_values, 2, axis=1)
    normalized = []
    for half in halves:
        shifted = half - half.min(axis=1, keepdims=True)
        width = shifted.max(axis=1, keepdims=True)
        normalized.append(np.pi * shifted / np.where(width > 0, width, 1))
    return np.concatenate(normalized, axis=1)


def product_states(angles):
    n = angles.shape[1] // 2
    states = np.ones((len(angles), 1), dtype=complex)
    for q in range(n):
        local = np.column_stack((np.cos(angles[:, q] / 2),
                   np.exp(1j * angles[:, n + q]) * np.sin(angles[:, q] / 2)))
        states = (states[:, :, None] * local[:, None, :]).reshape(len(angles), -1)
    return states


def apply_gate(states, gate, qubit, n):
    blocks = states.reshape(len(states), 2**qubit, 2, 2**(n - qubit - 1))
    return np.einsum("ab,nibj->niaj", gate, blocks).reshape(states.shape)


def pauli_action(states, axis, q, n):
    if axis == "x":
        return states[:, np.arange(2**n) ^ (1 << (n - 1 - q))]
    return states * (1 - 2 * bit_table(n)[:, q])


def rotation(angle, axis):
    if axis == "z":
        return np.diag(np.exp(-0.5j * angle * np.array([1, -1])))
    return np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                     [-1j * np.sin(angle / 2), np.cos(angle / 2)]])


class RydbergCircuit:
    def __init__(self, physics):
        self.physics = physics
        self.n = physics.n_qubits
        self.unitary = expm(-1j * physics.evolution_us * hamiltonian(physics))
        z = 1 - 2 * bit_table(self.n)
        self.observables = np.column_stack([z] + [z[:, i] * z[:, j]
                     for i, j in combinations(range(self.n), 2)])
        self.n_params = 3 * self.n * physics.layers
        self.forward_calls = 0
        self.sample_circuit_evaluations = 0

    def forward(self, initial, parameters, tape=False):
        if np.size(parameters) != self.n_params:
            raise ValueError("Wrong number of rotation parameters")
        self.forward_calls += 1
        self.sample_circuit_evaluations += len(initial)
        states = initial.copy()
        operations = []
        index = 0
        for layer in range(self.physics.layers):
            for axis in ("x", "z", "x"):
                for q in range(self.n):
                    gate = rotation(parameters[index], axis)
                    states = apply_gate(states, gate, q, self.n)
                    if tape:
                        operations.append((index, axis, q, gate, states))
                    index += 1
            states = states @ self.unitary.T
            if tape:
                operations.append(None)
        return (states, operations) if tape else states

    def features(self, states, shots=0, rng=None):
        probabilities = np.abs(states)**2
        if shots:
            if shots < 1 or rng is None:
                raise ValueError("Positive shots and an RNG are required")
            # Shared bitstrings estimate Z and ZZ together; no independent fake noise.
            probabilities = np.array([rng.multinomial(shots, p / p.sum()) / shots
                                      for p in probabilities])
        return probabilities @ self.observables

    def backward(self, final, tape, feature_gradient):
        """Adjoint gradient; dL = 2 Re <adjoint|dstate>, simulation only."""
        adjoint = final * (feature_gradient @ self.observables.T)
        gradient = np.zeros(self.n_params)
        for operation in reversed(tape):
            if operation is None:
                adjoint = adjoint @ self.unitary.conj()
                continue
            index, axis, q, gate, output = operation
            derivative = -0.5j * pauli_action(output, axis, q, self.n)
            gradient[index] = 2 * np.vdot(adjoint, derivative).real
            adjoint = apply_gate(adjoint, gate.conj().T, q, self.n)
        return gradient
