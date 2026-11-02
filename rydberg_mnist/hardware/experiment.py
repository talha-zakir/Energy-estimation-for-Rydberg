"""Small real-MNIST probe of finite pulses and noise; not an accuracy benchmark."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import qutip as qt
from threadpoolctl import threadpool_limits

from ..data import load_mnist
from ..experiment import run
from ..quantum import Physics, RydbergCircuit, encode_angles, product_states
from .energy import ledger
from .schedule import Compiler, load_profile, schedule_json
from .simulate import PulseSimulator, computational_indices


def model_inputs(model_path, test_images):
    with np.load(model_path, allow_pickle=False) as model:
        components = (test_images/255.-model["pca_mean"]) @ model["pca_components"].T
        angles = encode_angles(components)
        architecture = str(model["architecture"]) if "architecture" in model else "digital_analog"
        prefix = "digital" if architecture == "digital" else "rydberg"
        return (Physics(**json.loads(str(model["physics_json"]))), architecture,
                angles, model["parameters"].copy(), model[f"trained_{prefix}_linear_weights"].copy(),
                model[f"trained_{prefix}_linear_bias"].copy(), model["digits"].copy())


def probe(output, profile, cache, n=4, examples=4, realizations=2, shots=1000, resume=False):
    if not 2 <= n <= 4 or examples < 2 or examples % 2 or realizations < 1 or shots < 1:
        raise ValueError("Probe needs 2..4 atoms, an even positive number of examples, and positive draws/shots")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    started = time.perf_counter()
    p = Physics(n_qubits=n)
    # Preserve the requested common geometry if the profile is changed.
    rb = (profile["c6_rad_us_um6"]/p.omega_rad_us)**(1/6)
    p = Physics(**{**asdict(p), "rb_over_a": rb/profile["spacing_um"],
                   "c6_rad_us_um6": profile["c6_rad_us_um6"]})
    config = {"digits": [3, 8], "seed": 7, "n_train": 200, "n_val": 40,
              "n_test": max(40, examples), "maxiter": 20, "shots": shots,
              "threads": 4, "alpha_grid": [0., .001, .01, .1, 1., 10.], "physics": asdict(p)}
    _, _, test_images, _ = load_mnist(cache)
    summary = {"status": "Illustrative hardware-informed research simulation; "
                         "not a device-calibrated prediction or hardware accuracy benchmark",
               "profile": profile, "examples": examples, "static_noise_realizations": realizations,
               "shots_per_input": shots, "training_origin": "ideal CPU simulator",
               "readout_protocol": "Ideal-trained linear weights frozen; no retraining on pulse/noisy features",
               "noise_protocol": "Exact Lindblad density evolution per static control draw. "
                     "Same position/detuning/amplitude draw paired across architectures; "
                     "draws are shared over inputs within a realization. Detector shots sample "
                     "the equal mixture of these states, not a single fixed-drift realization.",
               "readout_erasure_policy": "Residual |r> is flagged with Z=0. No postselection. "
                     "0/1 detection flips are included; erasure-resolving detection is an explicit assumption.",
               "versions": {name: importlib.metadata.version(name) for name in
                            ("qutip", "numpy", "scipy", "scikit-learn")}, "architectures": {}}
    chosen_indices = None
    for architecture in ("digital_analog", "digital"):
        config["architecture"] = architecture
        ideal_dir = output / ("ideal_"+architecture)
        if resume and ideal_dir.exists():
            previous = json.loads((ideal_dir / "metrics.json").read_text())
            if previous["config"] != config:
                raise ValueError("Resume config does not match the saved ideal model")
        else:
            run(dict(config), ideal_dir, cache)
        with np.load(ideal_dir / "splits_predictions.npz") as split:
            test_indices, labels = split["test_idx"], split["test_labels"]
            positions = np.concatenate([np.flatnonzero(labels == label)[:examples//2] for label in (3, 8)])
            selected_indices, selected_labels = test_indices[positions], labels[positions]
        if chosen_indices is None:
            chosen_indices = selected_indices
        np.testing.assert_array_equal(chosen_indices, selected_indices)
        physics, _, angles, parameters, w, b, digits = model_inputs(ideal_dir / "model.npz", test_images[selected_indices])
        compiler = Compiler(physics, profile)
        ideal_states = RydbergCircuit(physics, architecture).forward(product_states(angles), parameters)
        ideal_features = RydbergCircuit(physics, architecture).features(ideal_states)
        sim_coherent = PulseSimulator(n, profile)
        sim_noisy = [PulseSimulator(n, profile, noisy=True, seed=1007+k) for k in range(realizations)]
        rows = []
        for index, angle in enumerate(angles):
            schedule = compiler.compile(angle, parameters, architecture)
            if index == 0:
                (output / (architecture+"_schedule.json")).write_text(json.dumps(schedule_json(schedule), indent=2)+"\n")
            coherent, result = sim_coherent.evolve(schedule)
            coherent_features = sim_coherent.features(result["probabilities"])
            desired_vector = np.zeros(3**n, dtype=complex)
            desired_vector[computational_indices(n)] = ideal_states[index]
            desired = qt.Qobj(desired_vector, dims=[[3]*n, [1]*n])
            desired_projector = qt.ket2dm(desired)
            coherent_fidelity = float(abs(desired.overlap(coherent))**2)
            noise_results, noise_features, noise_fidelities, noise_probs = [], [], [], []
            for realization, sim in enumerate(sim_noisy):
                state, diagnostics = sim.evolve(schedule)
                noise_features.append(sim.features(diagnostics["probabilities"]))
                noise_probs.append(diagnostics["probabilities"])
                noise_fidelities.append(float(qt.expect(desired_projector, state)))
                noise_results.append({k: v for k, v in diagnostics.items() if k != "probabilities"})
            mean_features = np.mean(noise_features, axis=0)
            mean_probs = np.mean(noise_probs, axis=0)
            sampled_features = sim_noisy[0].features(mean_probs, shots=shots, seed=2007+index)
            row = {"mnist_official_test_index": int(selected_indices[index]), "label": int(selected_labels[index]),
                   "ideal_prediction": int(digits[np.argmax(ideal_features[index]@w+b)]),
                   "coherent_pulse_prediction": int(digits[np.argmax(coherent_features@w+b)]),
                   "noisy_expected_prediction": int(digits[np.argmax(mean_features@w+b)]),
                   "noisy_sampled_prediction": int(digits[np.argmax(sampled_features@w+b)]),
                   "coherent_pulse_state_fidelity_to_ideal": coherent_fidelity,
                   "noisy_state_fidelity_to_ideal_mean": float(np.mean(noise_fidelities)),
                   "coherent_diagnostics": {k: v for k, v in result.items() if k != "probabilities"},
                   "noise_diagnostics": noise_results,
                   "ideal_features": ideal_features[index].tolist(),
                   "coherent_features": coherent_features.tolist(), "noisy_features_mean": mean_features.tolist(),
                   "energy": ledger(schedule, profile, shots)}
            rows.append(row)
            print(f"Pulse probe {architecture}: {index+1}/{examples}; "
                  f"coherent fidelity {coherent_fidelity:.4f}, noisy fidelity {np.mean(noise_fidelities):.4f}", flush=True)
        summary["architectures"][architecture] = {"n_qubits": n, "physics": physics.describe(), "inputs": rows}
        # Save after each architecture so a long run has recoverable progress.
        (output / "pulse_report.json").write_text(json.dumps(summary, indent=2)+"\n")
    summary["runtime_seconds"] = time.perf_counter()-started
    (output / "pulse_report.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def account_saved_pair(analog_dir, digital_dir, profile, output, cache, shots=1000):
    """Compile existing eight-qubit trained models without large density simulation."""
    from ..compare import validate_pair
    validate_pair(analog_dir, digital_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _, _, test_images, _ = load_mnist(cache)
    with np.load(Path(analog_dir) / "splits_predictions.npz") as split:
        selected = int(split["test_idx"][0])
        label = int(split["test_labels"][0])
    reports = {}
    for directory in (analog_dir, digital_dir):
        p, architecture, angles, theta, _, _, _ = model_inputs(Path(directory)/"model.npz", test_images[[selected]])
        if not np.isclose(p.spacing_um, profile["spacing_um"], atol=1e-8) or not np.isclose(p.c6_rad_us_um6, profile["c6_rad_us_um6"]):
            raise ValueError("Saved ideal model and hardware profile differ in geometry/C6; retrain before comparing")
        schedule = Compiler(p, profile).compile(angles[0], theta, architecture)
        reports[architecture] = {"source_model": str(Path(directory)/"model.npz"),
                                 "n_qubits": p.n_qubits, "segment_count": len(schedule),
                                 "energy": ledger(schedule, profile, shots)}
        (output / (architecture+"_schedule.json")).write_text(json.dumps(schedule_json(schedule), indent=2)+"\n")
    report = {"status": "Eight-qubit schedule compilation/accounting only; no noisy eight-qubit dynamics run",
              "mnist_official_test_index": selected, "label": label,
              "scope": "One paired input, frozen trained parameters; no energy-per-accuracy or training-energy claim",
              "profile": profile, "architectures": reports}
    (output / "accounting_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="configs/hardware_research.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/mnist")
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--realizations", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--account-saved", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse verified ideal training artifacts after a failed probe")
    parser.add_argument("--analog-dir", default="results/mnist_3_8_seed7")
    parser.add_argument("--digital-dir", default="results/mnist_3_8_digital_seed7")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    with threadpool_limits(limits=4):
        if args.account_saved:
            report = account_saved_pair(args.analog_dir, args.digital_dir, profile, args.output, args.cache, args.shots)
            print(json.dumps(report, indent=2))
        else:
            probe(args.output, profile, args.cache, args.qubits, args.examples, args.realizations, args.shots, args.resume)


if __name__ == "__main__":
    main()
