"""Summarize the pinned Pasqal target and its separate native classifier."""
import json
from pathlib import Path

import numpy as np


def summarize(root="hardware_results"):
    root = Path(root)
    block = json.loads((root/"pasqal_fresnel_8q/report.json").read_text())
    native = json.loads((root/"pasqal_native_mnist_3_8_seed7/report.json").read_text())
    target = native["target"]
    lines = ["# Pasqal target and native analog classifier", "",
             f"Target: **{target['target_device']}**, pinned public specification, using QoolQit/Pulser. "
             "No authenticated hardware access or live calibration was established. All execution was local.", "",
             "## Compatibility", "",
             "The retrieved interface exposes global ground–Rydberg control. The original digital "
             "and digital–analog circuits are preserved as research baselines and are rejected "
             "by the Pasqal backend because they require unexposed local/storage controls. "
             "The new global-pulse classifier is a separate architecture, not a relabeling of "
             "either earlier circuit.", "",
             "[Target configuration](../configs/pasqal_target.json) · "
             "[Compatibility audit](pasqal_fresnel_8q/compatibility.json)", "",
             "## Eight-atom analog-block check", "",
             f"Compiled instruction duration: {block['compiled_duration_ns']} ns. "
             "This isolated check starts all atoms in the ground state; it is not an MNIST classifier.", "",
             "| Condition | Mean Rydberg occupation | Returned bitstring samples |",
             "|---|---:|---:|"]
    for name, condition in block["conditions"].items():
        lines.append(f"| {name} | {np.mean(condition['occupation']):.5f} | {condition['returned_sample_count']} |")
    lines += ["", "The noisy condition uses the supplied public-device noise model. Occupations "
              "are quantum-state expectations; detector errors affect the sampled bitstrings. "
              "The noise model's recency/applicability to a booked device is not independently verified.", "",
              "[Block simulation report](pasqal_fresnel_8q/report.json)", "",
              "## Pasqal-native MNIST 3/8 classifier", "",
              f"{native['config']['n_qubits']} atoms; {native['config']['input_slots']} global pulse slots; "
              f"{native['parameter_count']} trained pulse offsets; linear regression readout. "
              f"{native['config']['n_train']} training, {native['config']['n_val']} validation, "
              f"{native['config']['n_test']} held-out test images; seed {native['config']['seed']}.", "",
              "Every training evaluation is compiled with QoolQit and emulated with Pulser's "
              "finite-bandwidth response. Geometry and instruction duration are fixed across "
              "inputs/parameters. SPSA trains pulse offsets; the readout is fitted analytically. "
              "Training uses the coherent bandwidth model. Pulse/readout parameters are frozen "
              "before noisy evaluation.", "",
              "| Model/condition | Held-out accuracy | Correct / test images |",
              "|---|---:|---:|"]
    for name, item in native["models"].items():
        metric = item["test"]
        lines.append(f"| {name} | {metric['accuracy']:.1%} | {metric['correct']} / {metric['n']} |")
    lines += ["", f"Fixed SPSA budget: {native['config']['iterations']} iterations; selected "
              f"checkpoint: {native['selected_iteration']} by validation MSE. No convergence "
              f"claim. Compiled/emulated sequences: {native['compiled_emulated_sequences']}. "
              f"Runtime: {native['runtime_seconds']:.1f} s.", "",
              "This is a small single-seed pipeline check. The datasets, architecture, feature "
              "budget and training budget differ from the original eight-qubit baselines. "
              "These scores do not establish a fair cross-architecture advantage or quantum advantage.", "",
              "The device-noise expectation condition includes state dynamics; the shot condition "
              "also includes SDK detector/SPAM errors. Their distinction matters when using a "
              "readout trained without those errors.", "",
              "[Classifier report](pasqal_native_mnist_3_8_seed7/report.json) · "
              "[Training history](pasqal_native_mnist_3_8_seed7/history.json) · "
              "[Compiled input example](pasqal_native_mnist_3_8_seed7/compiled_example.json) · "
              "[Figure](pasqal_native_comparison.png)", "",
              "## Energy status", "",
              f"For the first test input, scheduled modulated control window: "
              f"{native['first_test_input_energy']['scheduled_control_window_us_per_shot']:.3f} us. "
              "Its laser-control exposure is recorded from the actual SDK waveform. Full shot "
              "cycle time and optical/electrical/end-to-end joules remain unknown. No laser "
              "power or wall efficiency is inferred from Rabi frequency alone.", "",
              "The next energy study must establish common task/data/accuracy criteria, compatible "
              "hardware targets for the original baselines, measured power/overhead parameters, "
              "and explicit training versus inference workload counts.", "",
              "[Design, assumptions and reproduction commands](../docs/pasqal-target.md)", ""]
    (root/"PASQAL_SUMMARY.md").write_text("\n".join(lines))
    plot(native, root)
    print("\n".join(lines))


def plot(native, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [name.replace("_", " ") for name in native["models"]]
    values = [item["test"]["accuracy"]*100 for item in native["models"].values()]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.barh(labels, values)
    ax.set(xlim=(0, 100), xlabel="Held-out test accuracy (%)",
           title="Pasqal-native analog classifier — 40 MNIST 3/8 test images, one seed")
    for i, value in enumerate(values):
        ax.text(value-1, i, f"{value:.1f}%", ha="right", va="center", color="white")
    fig.savefig(root/"pasqal_native_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    summarize()
