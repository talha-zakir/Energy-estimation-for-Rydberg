"""Validate and summarize paired digital/analog experiments without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np


def validate_pair(analog_dir, digital_dir):
    """Reject comparisons that changed the data, encoding, initialization or budget."""
    directories = [Path(analog_dir), Path(digital_dir)]
    reports = [json.loads((d / "metrics.json").read_text()) for d in directories]
    if [r.get("architecture", "digital_analog") for r in reports] != ["digital_analog", "digital"]:
        raise ValueError("Expected one digital_analog and one digital run")
    for key in ("digits", "seed", "n_train", "n_val", "n_test", "maxiter", "shots", "alpha_grid"):
        if reports[0]["config"][key] != reports[1]["config"][key]:
            raise ValueError(f"Unmatched comparison setting: {key}")
    for key in ("n_qubits", "layers"):
        if reports[0]["config"]["physics"][key] != reports[1]["config"]["physics"][key]:
            raise ValueError(f"Unmatched circuit size: {key}")
    if reports[0]["optimizer"]["method"] != reports[1]["optimizer"]["method"]:
        raise ValueError("Unmatched optimizers")
    with np.load(directories[0] / "splits_predictions.npz") as a, \
         np.load(directories[1] / "splits_predictions.npz") as d:
        for key in ("train_idx", "validation_idx", "test_idx", "train_labels", "validation_labels", "test_labels"):
            np.testing.assert_array_equal(a[key], d[key], err_msg=f"Unmatched {key}")
        # These controls should be identical when only the entangler changes.
        for model in ("classical_pca_linear", "encoding_only_linear"):
            for split in ("train", "validation", "test"):
                key = f"{model}_{split}"
                np.testing.assert_array_equal(a[key], d[key], err_msg=f"Unmatched control: {key}")
    with np.load(directories[0] / "model.npz") as a, np.load(directories[1] / "model.npz") as d:
        np.testing.assert_array_equal(a["initial_parameters"], d["initial_parameters"])
        for key in ("pca_components", "pca_mean", "pca_std", "digits"):
            np.testing.assert_allclose(a[key], d[key], atol=1e-12, rtol=1e-12,
                                       err_msg=f"Unmatched preprocessing: {key}")
    return reports


def compare(root="results"):
    root = Path(root)
    indexed = {}
    for path in root.glob("*/metrics.json"):
        report = json.loads(path.read_text())
        key = (tuple(report["config"]["digits"]), report["config"]["seed"],
               report.get("architecture", "digital_analog"))
        if key in indexed:
            raise ValueError(f"Ambiguous duplicate experiment: {key}")
        indexed[key] = path.parent
    grouped = {}
    for (digits, seed, architecture), analog_dir in sorted(indexed.items()):
        if architecture != "digital_analog":
            continue
        digital_dir = indexed.get((digits, seed, "digital"))
        if digital_dir is None:
            continue
        analog, digital = validate_pair(analog_dir, digital_dir)
        row = {"seed": seed, "analog_directory": analog_dir.name, "digital_directory": digital_dir.name,
               "pairing_checks_passed": True}
        for condition, suffix in (("exact", ""), ("shots", "_shots")):
            if suffix and not analog["config"]["shots"]:
                continue
            a = analog["models"]["trained_rydberg_linear" + suffix]["test"]["accuracy"]
            d = digital["models"]["trained_digital_linear" + suffix]["test"]["accuracy"]
            row[condition] = {"digital_analog": a, "digital": d, "digital_minus_analog": d - a}
        grouped.setdefault(digits, []).append(row)
    if not grouped:
        raise ValueError("No matched architecture pairs found")

    lines = ["# Digital-only versus digital–analog VQC", "",
             "CPU simulations only. Each architecture uses the same 8 qubits, two layers, "
             "48 trainable RX/RZ angles, 16 PCA inputs, 36 measured Z/ZZ features, and "
             "linear regression readout. All runs have a 40-iteration L-BFGS budget. "
             "Readout alpha is selected separately by the same validation procedure; "
             "checkpoint selection uses validation MSE. Test labels do not tune either architecture.", "",
             "The digital entangler contains nearest-neighbor CNOTs on even bonds followed "
             "by odd bonds, matching Lu et al.'s noiseless digital reference. It is an ideal "
             "gate circuit, not a digital approximation of the analog Hamiltonian.", "",
             "The comparison checks identical split indices/labels, initial angles, fitted "
             "PCA, classical/encoding-only predictions, shot budget, hyperparameter grid, "
             "and training budget before including a pair. Original analog runs are retained.", ""]
    aggregated = {}
    for condition in ("exact", "shots"):
        lines += [f"## {'Exact expectations' if condition == 'exact' else '1,000-shot readout'}", "",
                  "Mean test accuracy ± sample SD across seeds (percentage points). "
                  "The difference is digital minus digital–analog.", "",
                  "| Task | Seeds | Digital–analog | Digital only | Mean difference |",
                  "|---|---|---:|---:|---:|"]
        for digits, rows in grouped.items():
            rows.sort(key=lambda r: r["seed"])
            if condition not in rows[0]:
                continue
            task = "3 vs 8" if digits == (3, 8) else "All ten digits"
            stats = {}
            for name in ("digital_analog", "digital", "digital_minus_analog"):
                values = np.array([row[condition][name] for row in rows])
                stats[name] = {"mean": float(values.mean()),
                               "sample_std": float(values.std(ddof=1)) if len(rows) > 1 else 0.}
            aggregated.setdefault(task, {"pairs": rows})[condition] = stats
            a, d = stats["digital_analog"], stats["digital"]
            seeds = ", ".join(str(row["seed"]) for row in rows)
            lines.append(f"| {task} | {seeds} | {100*a['mean']:.2f}% ± {100*a['sample_std']:.2f} pp "
                         f"| {100*d['mean']:.2f}% ± {100*d['sample_std']:.2f} pp "
                         f"| {100*stats['digital_minus_analog']['mean']:+.2f} pp |")
        lines.append("")
    lines += ["## Resource interpretation", "",
              "| Per ansatz circuit | Digital–analog | Digital only |",
              "|---|---:|---:|", "| Trainable single-qubit rotations | 48 | 48 |",
              "| Logical CNOT gates | 0 | 14 |", "| CNOT parallel sublayers | 0 | 4 |",
              "| Global analog evolution blocks | 2 | 0 |",
              "| Global analog evolution duration | 0.5 us | Not applicable |", "",
              "Counts exclude input preparation/readout. Matching layer and parameter counts "
              "does not match physical time, fidelity, laser energy or native gate count. "
              "The digital model does not simulate Rydberg pulses or CNOT errors. Native "
              "CNOT decomposition and its duration remain device-dependent. These results "
              "cannot establish which architecture uses less hardware energy.", "",
              "## Limits", "",
              "Only three seeds and small held-out MNIST subsets are used. Seeds change "
              "both data selection and initialization; test subsets overlap between seeds. "
              "The initial analog test results were already inspected before adding the digital "
              "arm, so this is an exploratory paired comparison, not a fresh confirmatory test. "
              "Quantum training is noiseless; the shot condition only resamples final "
              "features and refits the linear head. Optimization can stop at its budget limit. "
              "No quantum advantage or hardware superiority is established.", "",
              "## Paired artifacts", ""]
    for digits, rows in grouped.items():
        for row in rows:
            lines.append(f"- Digits {list(digits)}, seed {row['seed']}: "
                         f"[digital–analog]({row['analog_directory']}/metrics.json), "
                         f"[digital]({row['digital_directory']}/metrics.json). Pairing checks passed.")
    lines += ["", "Regenerate: `python -m rydberg_mnist.compare`.", ""]
    (root / "ARCHITECTURE_COMPARISON.md").write_text("\n".join(lines))
    (root / "architecture_comparison.json").write_text(json.dumps(aggregated, indent=2) + "\n")
    plot_comparison(aggregated, root)
    print("\n".join(lines))


def plot_comparison(aggregated, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, (task, values) in zip(axes, aggregated.items()):
        for name, label, color in (("digital_analog", "Digital–analog", "tab:blue"),
                                   ("digital", "Digital only", "tab:orange")):
            a = [row["exact"][name] * 100 for row in values["pairs"]]
            seeds = [str(row["seed"]) for row in values["pairs"]]
            ax.plot(seeds, a, "o-", label=label, color=color)
        ax.set(title=task, xlabel="Paired seed", ylabel="Test accuracy (%)", ylim=(0, 100))
        ax.legend()
    fig.suptitle("Ideal CPU simulation — matched data and parameter budget")
    fig.savefig(root / "architecture_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results", help="Directory with one run per task/seed/architecture")
    compare(parser.parse_args().root)
