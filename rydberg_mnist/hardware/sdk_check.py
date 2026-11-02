"""Independent analog-block check using real QoolQit/Pulser device compilation."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import qoolqit as qq
from pulser.backend import EmulationConfig, default_observables
from pulser.sampler import sample
from pulser_simulation import QutipBackendV2
from scipy.linalg import expm

from ..quantum import Physics, bit_table, hamiltonian


def sdk_check(physics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    nn_interaction = physics.c6_rad_us_um6/physics.spacing_um**6
    # Normalize nearest-neighbor J to one, including the required time scaling.
    coords = [(float(q-(physics.n_qubits-1)/2), 0.) for q in range(physics.n_qubits)]
    duration = nn_interaction*physics.evolution_us
    register = qq.Register.from_coordinates(coords)
    drive = qq.Drive(amplitude=qq.ConstantWaveform(duration, physics.omega_rad_us/nn_interaction),
                    detuning=qq.ConstantWaveform(duration, physics.omega_rad_us*physics.delta_over_omega/nn_interaction))
    program = qq.QuantumProgram(register, drive)
    # This is the SDK's constrained AnalogDevice, not its unconstrained MockDevice.
    program.compile_to(qq.AnalogDevice(), profile="max_energy")
    sequence = program.compiled_sequence
    backend = QutipBackendV2(sequence, config=EmulationConfig(
        observables=[default_observables.StateResult()], prefer_device_noise_model=False))
    result = backend.run()
    state = result.get_result("state", 1.).to_qobj().full().ravel()
    # Pulser orders (r,g); our ideal reference orders (g,r).
    state = state[::-1]
    initial = np.zeros(2**physics.n_qubits); initial[0] = 1.
    expected = expm(-1j*physics.evolution_us*hamiltonian(physics)) @ initial
    fidelity = float(abs(np.vdot(expected, state))**2)
    samples = sample(sequence)
    pulse = samples.channel_samples["rydberg"]
    # Check the compiler separately from the sampled waveform emulator. Pulser
    # appends a zero endpoint on its 1 ns grid, so its interpolated endpoint is
    # not exactly the mathematical rectangle used by the original baseline.
    bits = bit_table(physics.n_qubits)
    coords_physical = np.array(list(sequence.register.qubits.values()))
    omega, delta = float(np.asarray(pulse.amp).max()), float(np.asarray(pulse.det).max())
    compiled_h = np.diag(-delta*bits.sum(axis=1))
    indices = np.arange(2**physics.n_qubits)
    for i in range(physics.n_qubits):
        compiled_h[indices, indices ^ (1 << (physics.n_qubits-1-i))] += omega/2
        for j in range(i+1, physics.n_qubits):
            v = sequence.device.interaction_coeff/np.linalg.norm(coords_physical[j]-coords_physical[i])**6
            compiled_h += np.diag(v*bits[:, i]*bits[:, j])
    compiled_state = expm(-1j*sequence.get_duration()/1000*compiled_h) @ initial
    compiled_fidelity = float(abs(np.vdot(expected, compiled_state))**2)
    report = {"scope": "Isolated rectangular global analog block from all-ground input; "
                       "not compilation of the full digital-only or hyperfine-storage hybrid circuit",
              "device": sequence.device.name, "constrained_sdk_device": True,
              "input_physics": physics.describe(),
              "compiled_duration_ns": sequence.get_duration(),
              "compiled_coordinates_um": {str(k): np.asarray(v).tolist() for k, v in sequence.register.qubits.items()},
              "compiled_c6_rad_us_um6": sequence.device.interaction_coeff,
              "compiled_peak_omega_rad_us": float(np.asarray(pulse.amp).max()),
              "compiled_peak_detuning_rad_us": float(np.asarray(pulse.det).max()),
              "ideal_state_fidelity": fidelity,
              "compiled_rectangle_state_fidelity": compiled_fidelity,
              "sampled_backend_infidelity_tolerance": 1e-4,
              "passes_fidelity_check": fidelity > 1-1e-4 and compiled_fidelity > 1-1e-8,
              "sampling_note": "Pulser's 1 ns waveform grid includes a zero-valued end sample. "
                    "Compiled rectangular Hamiltonian is checked separately at stricter tolerance; "
                    "the sampled backend's small waveform-discretization discrepancy is retained.",
              "calibration_note": "SDK device constraints are checked. This check does not "
                  "supply current device calibration, noise validation or measured power. "
                  "Compilation rescales geometry, amplitude and time; these compiled controls "
                  "are NOT substituted into the separate shared research-profile energy ledger.",
              "versions": {name: importlib.metadata.version(name) for name in
                           ("qoolqit", "pulser-core", "pulser-simulation", "qutip")}}
    (output / "sdk_report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output / "compiled_sequence.json").write_text(sequence.to_abstract_repr()+"\n")
    if not report["passes_fidelity_check"]:
        raise RuntimeError(f"QoolQit/Pulser cross-check failed: fidelity {fidelity}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(sdk_check(Physics(n_qubits=args.qubits), args.output), indent=2))


if __name__ == "__main__":
    main()
