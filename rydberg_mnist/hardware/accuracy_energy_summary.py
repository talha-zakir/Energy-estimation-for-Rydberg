"""Summarize censored accuracy targets without asserting a hardware-energy winner."""
import json

import numpy as np


def summarize(report, output):
    config = report["config"]
    aggregates = {}
    for target in config["accuracy_targets"]:
        rows = [repeat["targets"][str(target)] for repeat in report["repeats"]]
        paired = [row for row in rows if row["paired_target_reached"]]
        compatible = [row for row in paired if row["held_out_accuracies_within_declared_tolerance"]]
        aggregates[str(target)] = {"repeats": len(rows), "paired_validation_target_hits": len(paired),
            "paired_held_out_compatibility_hits": len(compatible),
            "qrc_target_hits": sum(row["algorithms"]["qrc"]["status"] == "validation_target_reached" for row in rows),
            "vqc_target_hits": sum(row["algorithms"]["vqc"]["status"] == "validation_target_reached" for row in rows),
            "selection": "All repetitions retained; no accuracy/cost extrapolation for unreached targets"}
    report["target_aggregate"] = aggregates
    report["accuracy_aggregate"] = {}
    for algorithm in ("qrc", "vqc", "classical"):
        values = [r["classical_linear_test"]["accuracy"] if algorithm == "classical" else
                  r["validation_selected_fallback"][algorithm]["held_out_test"]["accuracy"] for r in report["repeats"]]
        report["accuracy_aggregate"][algorithm] = {"mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "repeats": len(values), "scope": "Validation-selected fallback models; overlapping test subsets across seeds"}
    profile = report["device_profile"]
    lines = ["# Finite-shot energy-to-accuracy study: FRESNEL beta profile", "",
        "Both methods use the same device, input features, pulse encoding, temporal Z/ZZ observables, "
        "finite-shot allocation, detector/noise model and classical linear-regression procedure. "
        "QRC fixes its pulse offsets. VQC updates them using sampled minibatch SPSA and refits the same linear head.", "",
        "## Physical profile and hardware path", "",
        f"Target identifier: **{profile['target_device']}**. The account-specific beta generation is unverified. "
        f"{config['regime']['n_qubits']} atoms; {config['regime']['input_slots']} slots; "
        f"mixing {profile['mixing_rad_us']:.4f} rad/us; nearest-neighbor interaction/mixing "
        f"{profile['interaction_to_mixing_actual']}; encoding/mixing {profile['encoding_to_mixing']:.3f}. "
        f"Slot {profile['slot_duration_ns']} ns with {profile['rise_ns']} ns ramps. "
        "The comparable-scale regime follows [Kornjača et al., Supplementary II.2](https://arxiv.org/html/2407.02553v1). "
        "Device limits, layout size/filling, clock and pulse bounds are validated locally by Pulser.", "",
        f"Layout has {profile['layout_traps']} traps; pre-calibrated: {profile['layout_precalibrated']}. "
        "The default custom layout requires provider calibration. Serialized example sequences include the layout "
        "and measurement instruction and pass local QPU checks. Account access, live calibration and server-side "
        "validation are still required. This run submitted no hardware jobs.", "",
        "## Paired finite-shot protocol", "",
        f"MNIST {config['digits']}; {config['n_train']} train / {config['n_val']} validation / {config['n_test']} "
        f"official-test images per repeat; seeds {config['repeat_seeds']}. "
        f"Each method uses {config['shots_per_image']:,} total shots per image, split across "
        f"{config['readout_slots']} prefix readouts. Both train and validate on sampled bitstrings with SPAM; "
        "neither is trained on exact state expectations. PCA and feature scaling fit training data only. "
        "The ridge/ordinary linear regression implementation and alpha grid are identical. "
        f"VQC has at most {config['vqc_steps']} SPSA steps; no convergence claim. "
        f"Simulation uses {config['noise_trajectories']} noise trajectories, which are not physical shot multipliers.", "",
        "Targets are declared in the configuration and selected using validation accuracy only. Test images "
        "are evaluated after all target/checkpoint decisions are frozen. Reaching the same validation target "
        "does not guarantee equal held-out accuracy; both test scores and their tolerance flag are reported. "
        "Unreached targets retain their consumed budget and are explicitly censored.", "",
        "The shared frozen reservoir is also the zero-update member of the VQC family. If it already meets a "
        "target, the minimum VQC budget can equal QRC's. Forcing extra optimization would artificially inflate "
        "the VQC energy-to-target. This study tests the energy-efficiency hypothesis and does not assume it.", "",
        "## Observed held-out accuracy", "",
        "Models below are selected by minimum validation MSE; they are descriptive fallbacks when a target "
        "is unavailable. They are not used to declare an unreachable target reached.", "",
        "| Seed | QRC validation | QRC test | VQC validation | VQC test | VQC selected step | Classical linear test |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for repeat in report["repeats"]:
        a, b = [repeat["validation_selected_fallback"][name] for name in ("qrc", "vqc")]
        lines.append(f"| {repeat['seed']} | {a['checkpoint']['validation_accuracy']:.1%} | {a['held_out_test']['accuracy']:.1%} | "
                     f"{b['checkpoint']['validation_accuracy']:.1%} | {b['held_out_test']['accuracy']:.1%} | "
                     f"{b['checkpoint']['iteration']} | {repeat['classical_linear_test']['accuracy']:.1%} |")
    lines += ["", "## Energy needed to reach each validation target", "",
        "Requested physical-shot budgets include all earlier training, validation and objective evaluations. "
        "Unsuccessful optimization attempts are retained. The readout reuses stored features for its alpha grid, "
        "so a classical refit does not trigger extra quantum shots.", "",
        "| Target | Seed | QRC status / training shots | VQC status / training shots | Held-out QRC / VQC |",
        "|---|---:|---|---|---|"]
    for target in config["accuracy_targets"]:
        for repeat in report["repeats"]:
            row = repeat["targets"][str(target)]
            cells, tests = [], []
            for name in ("qrc", "vqc"):
                item = row["algorithms"][name]
                hit = item["status"] == "validation_target_reached"
                value = item["energy_to_validation_target"] if hit else item["training_budget_spent"]
                cells.append(("reached" if hit else "unreached")+f" / {value['requested_returned_shots']:,}")
                tests.append(f"{item['held_out_test']['accuracy']:.1%}" if hit else "—")
            lines.append(f"| {target:.0%} | {repeat['seed']} | {cells[0]} | {cells[1]} | {' / '.join(tests)} |")
    lines += ["", "## Electrical calibration and scenario boundaries", "",
        f"Calibration status: **{report['calibration']['calibration_status']}**. "
        "FRESNEL-specific power, complete cycle time, attempts/loss factor, setup energy and external classical "
        "energy are required for complete calibrated totals. Missing values remain null. "
        "A measured full cycle already includes pulse execution and is never charged twice.", "",
        "The JSON separately retains the earlier paper-power / paper-overhead scenarios and a sensitivity "
        "using the [provider's published beta throughput](https://www.scaleway.com/en/docs/quantum-computing/additional-content/pasqal-qpus/), "
        "0.25 shots/s, with the same conditional power envelopes. That nominal 4 s/shot cycle is 200 times "
        "the previous 20 ms transferred overhead; it is not a measurement of this booked device. "
        "The provider page labels the table Beta while its prose mentions Gamma, so the actual account/device "
        "generation must be confirmed. These scenarios do not replace electrical metering, include full "
        "classical energy, or infer provider charges.", "",
        "| Target | Seed | Method | Target status | Calibrated training J | Paper-reference training kWh | Nominal beta rate + historical 2.6 kW: training kWh |",
        "|---|---:|---|---|---:|---:|---:|"]
    for target in config["accuracy_targets"]:
        for repetition in report["repeats"]:
            for algorithm, item in repetition["targets"][str(target)]["algorithms"].items():
                hit = item["status"] == "validation_target_reached"
                value = item["energy_to_validation_target"] if hit else item["training_budget_spent"]
                calibrated = value["complete_workflow_energy_j"]
                scenarios = value["literature_scenarios"]
                paper = scenarios["harvard_paper_reference"]["kwh"]
                nominal = scenarios.get("nominal_beta_throughput__pasqal_historical_power_transfer", {}).get("kwh")
                calibrated_text = "unknown" if calibrated is None else f"{calibrated:.3f}"
                nominal_text = "unknown" if nominal is None else f"{nominal:.3f}"
                lines.append(f"| {target:.0%} | {repetition['seed']} | {algorithm} | "
                             f"{'reached' if hit else 'unreached; spent budget'} | {calibrated_text} | {paper:.3f} | {nominal_text} |")
    lines += ["",
        "For unreached targets, the table shows the budget already spent, not an estimate of energy that "
        "would eventually reach the target. Both power/rate-transfer columns remain conditional scenarios.", "",
        "![Validation learning curves and target reachability](accuracy_energy.png)", "",
        "## Limits and reuse", "",
        "This is a small exploratory sample with three paired repeats by default, not a fully converged VQC "
        "study or a replication of the paper's complete resampling analysis. Held-out images can overlap "
        "between repeat seeds and with earlier studies. Snapshot noise models and derived pulse scales "
        "require validation against live hardware. Accuracy or energy advantage is not presumed.", "",
        "The learning loop accepts a count-backend adapter; the default is local Pulser simulation. "
        "Another Pulser-compatible analog device can use a new pinned target profile. Other hardware "
        "requires an appropriate sequence compiler and execution adapter. See "
        "[the protocol and calibration guide](../../docs/accuracy-energy-protocol.md).", "",
        "[Machine-readable results](report.json) · [Configuration](../../configs/accuracy_energy_fresnel_beta.json)", ""]
    (output/"SUMMARY.md").write_text("\n".join(lines))
    plot(report, output)


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for repeat in report["repeats"]:
        for algorithm, style in (("qrc", "o"), ("vqc", ".-")):
            points = repeat["checkpoints"][algorithm]
            axes[0].plot([p["cumulative_requested_shots"] for p in points],
                         [p["validation_accuracy"]*100 for p in points], style,
                         label=f"{algorithm.upper()}, seed {repeat['seed']}")
    for target in report["config"]["accuracy_targets"]:
        axes[0].axhline(target*100, color="gray", linestyle=":", alpha=.3)
    axes[0].set(xlabel="Cumulative requested training + validation shots", ylabel="Validation accuracy (%)",
                ylim=(0, 105), title="Same finite-shot features and linear head")
    axes[0].legend(fontsize=7)
    targets = list(report["target_aggregate"])
    x = np.arange(len(targets))
    for offset, field, label in ((-.18, "qrc_target_hits", "QRC"), (.18, "vqc_target_hits", "VQC")):
        values = [report["target_aggregate"][t][field] for t in targets]
        axes[1].bar(x+offset, values, width=.35, label=label)
    axes[1].set(xticks=x, xticklabels=[f"{float(t):.0%}" for t in targets], ylabel="Repeats reaching validation target",
                ylim=(0, len(report["repeats"])+.5), title="Unreached targets remain censored")
    axes[1].legend()
    fig.suptitle("Energy-to-accuracy protocol — local FRESNEL-profile finite-shot simulation")
    fig.supxlabel("Energy accounting uses explicit measurement provenance; missing components remain unknown.", fontsize=9)
    fig.savefig(output/"accuracy_energy.png", dpi=180)
    fig.savefig(output/"accuracy_energy.svg")
    p = output/"accuracy_energy.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
    plt.close(fig)
