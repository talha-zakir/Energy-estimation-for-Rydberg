"""Aggregate committed experiments: python -m rydberg_mnist.summarize."""
import json
from pathlib import Path

import numpy as np


def summarize(root="results"):
    root = Path(root)
    groups = {}
    for path in sorted(root.glob("*/metrics.json")):
        report = json.loads(path.read_text())
        key = tuple(report["config"]["digits"])
        groups.setdefault(key, []).append((path.parent, report))
    lines = ["# Initial MNIST results", "", "CPU statevector simulations, not QPU results. "
             "Each run uses 1,000 training examples and 200 validation examples. "
             "Binary tests use 400 official-test examples; ten-class tests use 500. "
             "Physics and two-layer architecture are fixed across tasks.", ""]
    aggregates = {}
    for digits, runs in groups.items():
        label = "MNIST 3 vs 8" if digits == (3, 8) else "MNIST all ten digits"
        seeds = [r["config"]["seed"] for _, r in runs]
        lines += [f"## {label}", "", f"Seeds: {seeds}. Values are mean test accuracy ± sample "
                  "standard deviation across runs (percentage points). Seeds vary both the "
                  "stratified subset and parameter initialization; this is not an isolated "
                  "initialization study or a statistical significance test.", "",
                  "| Model | Mean ± SD | Individual runs |", "|---|---:|---|"]
        task = {}
        for name in runs[0][1]["models"]:
            values = np.array([r["models"][name]["test"]["accuracy"] for _, r in runs])
            mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.
            task[name] = {"mean_accuracy": mean, "sample_std": sd, "values": values.tolist()}
            values_text = ", ".join(f"{x:.2%}" for x in values)
            lines.append(f"| {name.replace('_', ' ')} | {100*mean:.2f}% ± {100*sd:.2f} pp | {values_text} |")
        aggregates[label] = {"seeds": seeds, "models": task}
        lines += ["", "Run artifacts:", ""]
        for directory, report in runs:
            link = directory.name
            status = report["optimizer"]
            lines.append(f"- Seed {report['config']['seed']}: [metrics]({link}/metrics.json), "
                         f"[figure]({link}/comparison.png), [history]({link}/history.json), "
                         f"[model]({link}/model.npz). Runtime {report['runtime_seconds']:.1f} s; "
                         f"selected iteration {status['selected_iteration']}; "
                         f"optimizer converged: {status['converged']}.")
        lines.append("")
    lines += ["## Interpretation", "",
              "Training improves on the frozen random Rydberg circuit in these runs. "
              "The encoding-only control is competitive, and these experiments do not "
              "establish a benefit attributable to Rydberg interactions or quantum advantage. "
              "The classical comparison uses only 16 PCA features; it is not a state-of-the-art "
              "MNIST benchmark. The held-out subsets are small and overlap across seeds.", "",
              "The shot condition uses 1,000 sampled bitstrings per example with a refitted "
              "linear readout. It does not model control noise, atom loss, decoherence or "
              "readout errors. Shot counts are simulated; no hardware jobs were submitted.", "",
              "The maximum optimizer budget is 40 iterations. A budget-limit exit is explicitly "
              "recorded, not reported as convergence. The selected model minimizes validation "
              "MSE over checkpoints; test labels do not select models. Per-run Wilson intervals "
              "are conditional on that fitted model and do not include training variation.", "",
              "Regenerate with `python -m rydberg_mnist.summarize`. Physics, data "
              "checksums, split indices, package versions and all deviations from the "
              "paper are recorded in this repository.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    (root / "aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    summarize()
