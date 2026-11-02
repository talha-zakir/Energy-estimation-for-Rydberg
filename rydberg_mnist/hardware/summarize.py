"""Generate a cautious, reproducible summary of the initial physical-model runs."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root="hardware_results"):
    root = Path(root)
    probe = json.loads((root/"pulse_probe_4q/pulse_report.json").read_text())
    accounting = json.loads((root/"accounting_8q/accounting_report.json").read_text())
    sdk = json.loads((root/"qoolqit_analog_block_8q/sdk_report.json").read_text())
    lines = ["# Hardware-informed simulation results", "",
             "These are research-model simulations with illustrative control/noise parameters, "
             "not calibrated QPU predictions. No hardware jobs were submitted. The original "
             "ideal eight-qubit accuracy benchmarks are retained separately.", "",
             "## SDK cross-check: isolated eight-atom analog block", "",
             f"QoolQit compiles to constrained `{sdk['device']}` and Pulser/QutipBackendV2 "
             f"simulates the sequence. Compiled duration: {sdk['compiled_duration_ns']} ns. "
             f"Compiled rectangular-Hamiltonian state fidelity to the original ideal block: "
             f"{sdk['compiled_rectangle_state_fidelity']:.12f}. Sampled-backend fidelity: "
             f"{sdk['ideal_state_fidelity']:.9f}.", "",
             "The backend's 1 ns sampled waveform has a zero end sample. Its small discrepancy "
             "from a mathematical rectangle is reported rather than hidden. The SDK's compilation "
             "rescales C6/geometry, amplitude and time. This block validation does not compile the "
             "full digital/hybrid storage controls, and its compiled settings are not used in the "
             "separate research-profile energy ledger.", "",
             "[SDK report](qoolqit_analog_block_8q/sdk_report.json) · "
             "[Compiled Pulser sequence](qoolqit_analog_block_8q/compiled_sequence.json)", "",
             "## Full pulse/noise smoke test: four atoms", "",
             "Four balanced MNIST 3/8 inputs, two paired static-noise realizations, and 1,000 "
             "simulated detector shots per input. Ideal-trained parameters/readouts are frozen. "
             "Fidelity below is unconditional state fidelity to each architecture's own ideal output; "
             "it is not digit classification accuracy and is not a direct similarity between architectures.", "",
             "| Architecture | Mean coherent pulse fidelity | Mean noisy fidelity | Mean residual Rydberg probability (noisy) |",
             "|---|---:|---:|---:|"]
    stats = {}
    for architecture, item in probe["architectures"].items():
        rows = item["inputs"]
        coherent = np.mean([r["coherent_pulse_state_fidelity_to_ideal"] for r in rows])
        noisy = np.mean([r["noisy_state_fidelity_to_ideal_mean"] for r in rows])
        erasure = np.mean([d["residual_rydberg_probability"] for r in rows for d in r["noise_diagnostics"]])
        norm_error = max([r["coherent_diagnostics"]["max_normalization_error"] for r in rows] +
                         [d["max_normalization_error"] for r in rows for d in r["noise_diagnostics"]])
        stats[architecture] = {"mean_coherent_fidelity": float(coherent), "mean_noisy_fidelity": float(noisy),
                               "mean_noisy_residual_rydberg_probability": float(erasure),
                               "max_normalization_error": norm_error}
        lines.append(f"| {architecture} | {coherent:.4f} | {noisy:.4f} | {erasure:.4f} |")
    lines += ["", "**This first pulse compiler is not yet a validated implementation of the "
              "ideal classifiers on hardware.** Finite transfer and always-on interactions "
              "substantially alter the hybrid outputs under this profile. The slow, serialized "
              "blockade gates are also sensitive to the assumed noise. Pulse calibration, phase "
              "compensation or pulse-aware retraining must precede an accuracy-matched energy "
              "comparison. These four examples and two noise draws cannot establish hardware "
              "accuracy or an energy winner.", "",
              "[Per-input outputs and diagnostics](pulse_probe_4q/pulse_report.json) · "
              "[Figure](pulse_probe.png)", "",
              "## Full eight-qubit schedules: accounting only", "",
              f"One paired official MNIST test input (index {accounting['mnist_official_test_index']}). "
              "Uses the existing trained eight-qubit models. This section compiles schedules; "
              "it does not run full noisy eight-atom dynamics.", "",
              "| Architecture | Linear waveform segments | Scheduled control time per shot | Optical joules | Full pipeline joules |",
              "|---|---:|---:|---|---|"]
    for architecture, item in accounting["architectures"].items():
        energy = item["energy"]
        lines.append(f"| {architecture} | {item['segment_count']} | "
                     f"{energy['scheduled_control_window_us_per_shot']:.3f} us | Unknown | Unknown |")
    lines += ["", "These times include encoding, storage rotations, transfer, CNOT basis/phase "
              "corrections and analog evolution. They exclude uncalibrated loading, cooling, "
              "measurement/reset and classical processing. The one-local-beam scheduler "
              "serializes CNOTs; optimized native gates or parallel addressing can change this "
              "comparison substantially. The values are not vendor performance specifications.", "",
              "The hybrid's short but strong transfer pulses have a much larger local-Rydberg "
              "squared-amplitude integral than the slower digital pulses. Therefore even the "
              "shorter control window does not by itself establish lower laser energy. Channel "
              "power calibrations and wall efficiencies are still absent; all joule totals remain "
              "null. Full pipeline energy additionally needs classical training/inference measurements.", "",
              "[Eight-qubit accounting](accounting_8q/accounting_report.json) · "
              "[Hybrid schedule](accounting_8q/digital_analog_schedule.json) · "
              "[Digital schedule](accounting_8q/digital_schedule.json)", "",
              "## Validation and next step", "",
              "Tests cover CNOT blockade dependence and phase corrections, the hybrid's fast-transfer "
              "limit, QuTiP integration against exact constant-segment exponentials, Lindblad decay, "
              "normalization, erasure handling, finite-rise pulse area/slew limits, global-beam power "
              "counting, shot multiplication, and missing/nonfinite energy calibration. Earlier "
              "ideal-circuit and data checks remain in the suite.", "",
              "Next select a device/control specification, calibrate powers and noise, improve or "
              "retrain the pulse implementations, and expand to a meaningful matched-accuracy "
              "classification benchmark. Extending full noisy dynamics beyond four atoms requires "
              "a trajectory/tensor-network implementation or additional compute.", "",
              "[Model, assumptions, equations and reproduction commands](../docs/pulse-model-and-energy.md)", ""]
    (root/"SUMMARY.md").write_text("\n".join(lines))
    (root/"probe_statistics.json").write_text(json.dumps(stats, indent=2)+"\n")
    plot_probe(probe, root)
    print("\n".join(lines))


def plot_probe(probe, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    for ax, (architecture, item) in zip(axes, probe["architectures"].items()):
        rows = item["inputs"]
        x = np.arange(1, len(rows)+1)
        ax.plot(x, [r["coherent_pulse_state_fidelity_to_ideal"] for r in rows], "o-", label="Coherent finite pulses")
        ax.plot(x, [r["noisy_state_fidelity_to_ideal_mean"] for r in rows], "s-", label="With assumed noise")
        ax.set(title=architecture.replace("_", " "), xlabel="Paired input", ylabel="Fidelity to own ideal output",
               ylim=(0, 1.02), xticks=x)
        ax.legend(fontsize=8)
    fig.suptitle("Four-atom research-model smoke test — not calibrated hardware accuracy")
    fig.savefig(root/"pulse_probe.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="hardware_results")
    summarize(parser.parse_args().root)
