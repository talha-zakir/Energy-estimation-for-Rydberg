"""Pasqal target audit and local QoolQit/Pulser emulation; never submits QPU jobs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from pulser.backend import EmulationConfig, default_observables as obs
from pulser.json.abstract_repr.deserializer import deserialize_device
from pulser.sampler import sample
from pulser_simulation import QutipBackendV2
import qoolqit as qq

from ..quantum import Physics
from .energy import ledger
from .schedule import Segment


class UnsupportedPasqalCircuit(ValueError):
    pass


def load_target(path):
    path = Path(path)
    config = json.loads(path.read_text())
    if config.get("kind") != "pasqal_sdk_target" or config.get("schema_version") != 1:
        raise ValueError("Expected a version-1 Pasqal SDK target configuration")
    source = path.parent/config["device_json"]
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != config["device_sha256"]:
        raise ValueError("Device snapshot SHA-256 mismatch; update the provenance before use")
    device = deserialize_device(data.decode())
    if device.name != config["target_device"]:
        raise ValueError("Target name does not match the device snapshot")
    return config, device


def capabilities(device):
    channels = {name: {"basis": c.basis, "addressing": c.addressing,
                       "max_omega_rad_us": c.max_amp, "max_abs_detuning_rad_us": c.max_abs_detuning,
                       "clock_period_ns": c.clock_period, "min_pulse_duration_ns": c.min_duration,
                       "mod_bandwidth_mhz": c.mod_bandwidth}
                for name, c in device.channels.items()}
    global_rydberg = any(c.basis == "ground-rydberg" and c.addressing == "Global"
                         for c in device.channels.values())
    local_rydberg = any(c.basis == "ground-rydberg" and c.addressing == "Local"
                        for c in device.channels.values())
    local_storage = any(c.basis == "digital" and c.addressing == "Local"
                        for c in device.channels.values())
    missing = []
    if not local_storage:
        missing.append("Individually addressed hyperfine/storage RX-RZ-RX controls are not exposed")
    if not local_rydberg:
        missing.append("Individually addressed Rydberg pulses for selected-pair blockade CNOTs are not exposed")
    # Ground-Rydberg channels do not certify |1>-|r> transfer from a separate
    # hyperfine storage manifold, even if a hypothetical device has local drives.
    transfer = "The storage-to-Rydberg transfer controls of the current hybrid compiler are not certified by this SDK snapshot"
    missing.append(transfer)
    return {"device": device.name, "channels": channels,
            "rydberg_level": device.rydberg_level, "c6_rad_us_um6": device.interaction_coeff,
            "max_atoms": device.max_atom_num, "min_distance_um": device.min_atom_distance,
            "max_radial_distance_um": device.max_radial_distance,
            "max_sequence_duration_ns": device.max_sequence_duration,
            "supported_global_analog_drive": global_rydberg,
            "exposed_local_storage_drive": local_storage, "exposed_local_rydberg_drive": local_rydberg,
            "device_noise_model_available": device.noise_model is not None,
            "noise_types": list(device.noise_model.noise_types) if device.noise_model else [],
            "original_digital_vqc": {"supported_as_written": False, "reasons": missing[:]},
            "original_hybrid_vqc": {"supported_as_written": False,
                                    "reasons": [r for r in missing if "CNOT" not in r]},
            "submission_ready": False,
            "submission_note": "Local simulation only. Account access, live calibration and any "
                               "cloud register/layout validation are not established."}


def require_supported(device, architecture):
    audit = capabilities(device)
    if architecture in ("digital", "digital_analog"):
        key = "original_digital_vqc" if architecture == "digital" else "original_hybrid_vqc"
        raise UnsupportedPasqalCircuit(f"{architecture} is not certified for {device.name}: "
                                       + "; ".join(audit[key]["reasons"]))
    if architecture not in ("analog_block", "native_analog") or not audit["supported_global_analog_drive"]:
        raise UnsupportedPasqalCircuit("This path supports only a global analog block on this target")


def compile_analog_block(device, physics):
    require_supported(device, "analog_block")
    if physics.n_qubits > device.max_atom_num:
        raise ValueError("Register exceeds target atom count")
    interaction = physics.c6_rad_us_um6/physics.spacing_um**6
    coords = [(float(q-(physics.n_qubits-1)/2), 0.) for q in range(physics.n_qubits)]
    duration = interaction*physics.evolution_us
    program = qq.QuantumProgram(qq.Register.from_coordinates(coords), qq.Drive(
        amplitude=qq.ConstantWaveform(duration, physics.omega_rad_us/interaction),
        detuning=qq.ConstantWaveform(duration, physics.omega_rad_us*physics.delta_over_omega/interaction)))
    program.compile_to(qq.Device(pulser_device=device), profile="max_energy")
    return program.compiled_sequence


def backend_config(device, *, noisy, modulation, shots=1000, trajectories=8):
    if shots < 1 or trajectories < 1:
        raise ValueError("Shots and trajectories must be positive")
    if noisy and device.noise_model is None:
        raise ValueError("Target has no supplied noise model; refusing to silently run noiselessly")
    return EmulationConfig(observables=[obs.Occupation(one_state="r"), obs.BitStrings(num_shots=shots, one_state="r")],
        prefer_device_noise_model=noisy, with_modulation=modulation,
        n_trajectories=trajectories if noisy else 1, default_num_shots=shots)


def sequence_ledger(sequence, target_config, *, modulation, shots):
    """Integrate the SDK's global drive samples; no invented storage controls."""
    samples = sample(sequence, modulation=modulation)
    segments, traces = [], {}
    for name, values in samples.channel_samples.items():
        if sequence.declared_channels[name].addressing != "Global":
            raise ValueError("This energy adapter currently handles global channels only")
        amplitude = np.append(np.asarray(values.amp, dtype=float), 0.)
        detuning = np.append(np.asarray(values.det, dtype=float), 0.)
        traces[name] = {"amplitude_rad_us": amplitude, "detuning_rad_us": detuning}
        for t in range(len(amplitude)-1):
            segments.append(Segment("SDK global waveform", "pasqal_analog_block", "rydberg_global", "0rx",
                    tuple(range(len(sequence.register.qubit_ids))), .001,
                    float(amplitude[t]), float(amplitude[t+1]), float(detuning[t]), float(detuning[t+1])))
    if len(traces) != 1:
        raise ValueError("Expected a single global channel for this target experiment")
    profile = {"calibration_status": target_config["calibration_status"],
               "power_calibration": target_config["power_calibration"], "overhead": target_config["overhead"]}
    report = ledger(segments, profile, shots)
    report["waveform_accounting"] = "Linear interpolation of 1 ns SDK amplitude samples including a final zero endpoint"
    report["modulation_included"] = modulation
    return report, traces


def run(target_path, output, n=8, shots=1000, trajectories=8, seed=7, architecture="analog_block"):
    config, device = load_target(target_path)
    audit = capabilities(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"compatibility.json").write_text(json.dumps(audit, indent=2)+"\n")
    require_supported(device, architecture)
    sequence = compile_analog_block(device, Physics(n_qubits=n))
    (output/"compiled_sequence.json").write_text(sequence.to_abstract_repr()+"\n")
    # Preserve the vendor-supplied noise with the public snapshot; don't relabel
    # the illustrative three-level research profile as Pasqal calibration.
    public_noise = json.loads(device.to_abstract_repr()).get("default_noise_model")
    report = {"status": "Local simulation of a supported global analog block, not a full VQC",
              "target_config": config, "target_capabilities": audit,
              "vendor_snapshot_noise_model": public_noise,
              "input": "all ground states; no MNIST encoding, local gates or hyperfine transfers",
              "readout": "1 denotes Rydberg |r>. Occupation is a state expectation; "
                         "sampled bitstrings include the SDK detector/SPAM model in the noisy condition.",
              "compiled_duration_ns": sequence.get_duration(),
              "compiled_coordinates_um": {str(k): np.asarray(v).tolist() for k, v in sequence.register.qubits.items()},
              "compiled_c6_rad_us_um6": sequence.device.interaction_coeff,
              "shots_per_condition": shots, "requested_noise_trajectories": trajectories,
              "seed": seed, "conditions": {},
              "noise_provenance": "Noise supplied with the public FRESNEL snapshot. "
                   "Its recency and applicability to a future booked run are not independently verified.",
              "reproducibility": "NumPy sampling RNG reset per condition; exact stochastic reproducibility "
                   "also depends on the SDK/solver versions and backend RNG behavior.",
              "versions": {name: importlib.metadata.version(name) for name in
                           ("qoolqit", "pulser-core", "pulser-simulation", "qutip", "pasqal-cloud")}}
    for label, noisy, modulation in (("ideal", False, False), ("bandwidth", False, True),
                                      ("device_noise_and_bandwidth", True, True)):
        np.random.seed(seed)
        result = QutipBackendV2(sequence, config=backend_config(device, noisy=noisy,
                    modulation=modulation, shots=shots, trajectories=trajectories)).run()
        occupation = np.asarray(result.get_result("occupation", 1.), dtype=float)
        bitstrings = dict(result.get_result("bitstrings", 1.))
        counts = {str(k): int(v) for k, v in bitstrings.items()}
        if sum(counts.values()) != shots:
            # Multi-trajectory observables may pool shots across trajectories.
            # Report what was returned; do not silently relabel the count.
            shot_note = "SDK pooled or redistributed trajectory samples; see returned_sample_count"
        else:
            shot_note = "Returned sample count matches requested count"
        energy, traces = sequence_ledger(sequence, config, modulation=modulation, shots=shots)
        np.savez_compressed(output/(label+"_waveforms.npz"), **{
            channel+"_"+key: values for channel, signals in traces.items() for key, values in signals.items()})
        report["conditions"][label] = {"device_noise_enabled": noisy, "modulation_enabled": modulation,
                "occupation": occupation.tolist(), "bitstrings": counts,
                "returned_sample_count": sum(counts.values()), "sampling_note": shot_note, "energy": energy}
        print(label, "mean occupation", float(occupation.mean()), "samples", sum(counts.values()), flush=True)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="configs/pasqal_target.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--architecture", choices=("analog_block", "digital", "digital_analog"), default="analog_block")
    args = parser.parse_args()
    run(args.target, args.output, args.qubits, args.shots, args.trajectories, architecture=args.architecture)


if __name__ == "__main__":
    main()
