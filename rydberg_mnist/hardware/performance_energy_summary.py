"""Full-training performance/resource summaries; calibrated energy may be unknown."""
import numpy as np


def summarize(report, output):
    aggregates = {}
    for algorithm in ("qrc", "vqc", "classical"):
        aggregates[algorithm] = {}
        for label in ("selected", "terminal"):
            values = [r["classical_linear_test"]["accuracy"] if algorithm == "classical" else
                      r["algorithms"][algorithm][label+"_test"]["accuracy"] for r in report["repeats"]]
            aggregates[algorithm][label] = {"mean": float(np.mean(values)),
                "sample_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "repeats": len(values), "scope": "Paired exploratory seeds; overlapping test subsets, not independent population replicates"}
    report["accuracy_aggregate"] = aggregates
    config, profile = report["config"], report["device_profile"]
    lines = ["# QRC/VQC full-training performance and resources", "",
        "Accuracy is an observed outcome. Both methods use identical atoms, unique data splits, training-only PCA, "
        "pulse encoder, Z/ZZ temporal observables, shots per image, device/noise specification and linear regression procedure. "
        "QRC keeps pulse offsets fixed; VQC uses minibatch SPSA and periodically refits the same readout.", "",
        f"MNIST {config['digits']}; {config['n_train']} train / {config['n_val']} validation / {config['n_test']} test; "
        f"seeds {config['repeat_seeds']}; {config['regime']['n_qubits']} atoms; {len(config['readout_slots'])} prefixes; "
        f"{config['shots_per_image']:,} total shots/image. Validation uses {config['validation_repetitions']} independent "
        "finite-shot feature measurements of the same validation images for both methods. The readout and alpha are "
        "selected by mean validation MSE. Test data is evaluated after checkpoint selection.", "",
        f"VQC stops after {config['plateau_patience']} checkpoint checks without at least "
        f"{config['plateau_relative_improvement']:.0%} relative MSE improvement, or {config['vqc_steps']} SPSA steps. "
        f"Checks occur every {config['checkpoint_every']} steps and at the cap. A plateau is an operational stopping rule; "
        "a budget-limited run has not met it. Neither establishes global convergence. The absolute best validation loss "
        "selects the model even if its improvement is smaller than the plateau threshold.", "",
        "## Held-out results and all consumed training", "",
        "Training costs below include **all** training, validation repetitions, optimizer evaluations and discarded "
        "checkpoints. Selecting step zero never removes later work from these costs. Terminal results show the last "
        "checkpoint separately. Test shots are separate; identical selected/terminal checkpoints reuse one test evaluation.", "",
        "| Seed | Method | Selected step | Selected test | Terminal step | Terminal test | Stop reason | All training shots |",
        "|---|---|---:|---:|---:|---:|---|---:|"]
    for repeat in report["repeats"]:
        for algorithm, item in repeat["algorithms"].items():
            lines.append(f"| {repeat['seed']} | {algorithm.upper()} | {item['selected_checkpoint']['iteration']} | "
                f"{item['selected_test']['accuracy']:.1%} | {item['terminal_checkpoint']['iteration']} | "
                f"{item['terminal_test']['accuracy']:.1%} | {item['completion']['reason']} | "
                f"{item['training_budget']['requested_returned_shots']:,} |")
    lines += ["", "| Method | Selected test mean ± sample SD | Terminal test mean ± sample SD |", "|---|---:|---:|"]
    for algorithm, groups in aggregates.items():
        values = [f"{groups[label]['mean']:.1%}"+(f" ± {groups[label]['sample_standard_deviation']:.1%}"
                  if groups[label]['sample_standard_deviation'] is not None else "") for label in ("selected", "terminal")]
        lines.append(f"| {algorithm.upper()} | {values[0]} | {values[1]} |")
    lines += ["", "## Device provenance and energy", "",
        f"Compiled target: **{profile['target_device']}**, snapshot `{report['target']['device_sha256']}`. "
        "The public FRESNEL snapshot is provisional for the user's FRESNEL beta account. Local Pulser geometry, "
        "layout, global-channel, timing and shot-cap validation does not establish account-specific server acceptance. "
        f"Layout pre-calibrated: **{profile['layout_precalibrated']}**. Provider layout calibration and a matching "
        "account device export remain required. This default run submitted no QPU jobs.", "",
        "The comparable interaction/drive/input regime follows "
        "[Kornjača et al., Supplementary II.2](https://arxiv.org/html/2407.02553v1#A2.SS2), adapted to this snapshot's "
        "constraints. This is a native analog VQC versus fixed analog reservoir comparison. Earlier digital/hybrid "
        "research architectures are separate and are not certified for these exposed global controls.", "",
        f"Electrical calibration status: **{report['calibration']['calibration_status']}**. "
        "Calibrated joules remain unknown without target-specific measured whole-machine power, full cycle, attempts, "
        "setup and allocated external classical energy. The JSON reports those components separately. Each ledger row "
        "identifies the exact compiled program hash, actual shot request and any supplied meter/retry evidence. "
        "The full cycle includes control, loading/reset and readout; control duration is not added twice. CPU simulation "
        "runtime is not QPU runtime or electrical energy.", "",
        "| Seed | Method | All training shots | Inference shots: selected / terminal | All-study jobs | Calibrated training J |",
        "|---|---|---:|---:|---:|---:|"]
    for repeat in report["repeats"]:
        for algorithm, item in repeat["algorithms"].items():
            energy = item["training_budget"]["complete_workflow_energy_j"]
            lines.append(f"| {repeat['seed']} | {algorithm.upper()} | {item['training_budget']['requested_returned_shots']:,} | "
                f"{item['selected_inference_budget']['requested_returned_shots']:,} / "
                f"{item['terminal_inference_budget']['requested_returned_shots']:,} | "
                f"{item['complete_study_execution_budget']['submitted_jobs_including_reported_retries']:,} | "
                f"{'unknown' if energy is None else f'{energy:.4f}'} |")
    lines += ["", "## Interpretation limits", "",
        "This is a small exploratory simulator study; earlier work inspected overlapping MNIST test images. "
        "Validation repetitions reduce measurement noise, not finite-dataset uncertainty or validation selection bias. "
        "The default two Monte Carlo noise trajectories are a computational approximation; a trajectory-convergence "
        "study and fresh held-out data are needed for publication claims. Repeated seed SDs are descriptive, not "
        "independent-test confidence intervals. The symmetric four-atom chain has only six symmetry-distinct ideal "
        "Z/ZZ channels per prefix, despite 40 nominal features across four prefixes; sampling can inflate numerical rank. "
        "The reservoir resets per image, so no streaming memory capacity is established. No quantum or electrical-energy "
        "advantage is inferred from a shot ratio.", "",
        "## Appendix: conditional literature sensitivity", "",
        "JSON `literature_scenarios` retains Carrasco-Codina electrical inventory/timing transfers and historical "
        "Pasqal power envelopes, plus nominal provider beta throughput. These are other-device or historical sensitivity "
        "assumptions, not measured FRESNEL beta values. They apply to the actual compiled controls and consumed shots. "
        "They do not determine the main results, provider bills, or calibrated energy totals. See "
        "[the protocol](../../docs/performance-energy-protocol.md) for reaccounting and literature boundaries.", ""]
    (output/"SUMMARY.md").write_text("\n".join(lines))
    plot(report, output)


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, repeat in enumerate(report["repeats"]):
        color = colors[index % len(colors)]
        for algorithm, style in (("qrc", "o"), ("vqc", "o-")):
            points = repeat["checkpoints"][algorithm]
            axes[0].plot([p["cumulative_requested_shots"] for p in points],
                         [p["validation_accuracy"]*100 for p in points], style,
                         label=f"{algorithm.upper()}, seed {repeat['seed']}", color=color,
                         markersize=9 if algorithm == "qrc" else 4,
                         markerfacecolor="none" if algorithm == "qrc" else color,
                         markeredgewidth=1.5 if algorithm == "qrc" else 1.,
                         zorder=5 if algorithm == "qrc" else 3)
    axes[0].set(xlabel="All consumed training + validation shots", ylabel="Mean validation accuracy (%)", ylim=(0, 105))
    axes[0].legend(fontsize=7)
    for offset, label in ((-.18, "selected"), (.18, "terminal")):
        groups = [report["accuracy_aggregate"][a][label] for a in ("qrc", "vqc", "classical")]
        axes[1].bar(np.arange(3)+offset, [g["mean"]*100 for g in groups], width=.35, label=label,
                    yerr=[(g["sample_standard_deviation"] or 0)*100 for g in groups], capsize=3)
    axes[1].set(xticks=np.arange(3), xticklabels=["QRC", "VQC", "PCA linear"], ylabel="Held-out accuracy (%)", ylim=(0, 105),
                title=f"Mean ± sample SD ({len(report['repeats'])} exploratory seeds)")
    axes[1].legend()
    fig.suptitle("Paired local FRESNEL-profile simulation: accuracy is an outcome")
    fig.supxlabel("Full consumed resources are retained; calibrated QPU electrical energy remains unknown.", fontsize=9)
    fig.savefig(output/"performance_energy.png", dpi=180)
    fig.savefig(output/"performance_energy.svg", metadata={"Date": None})
    path = output/"performance_energy.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)
