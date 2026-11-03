"""Matched-task QRC/VQC comparisons with explicit destructive-readout costs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .literature_cost import (Controls, component_power, native_training_count,
                             positive_integer, summary_stats, system_scenario, transferred_radiation)


def composite_energy(control_us, allocation, literature):
    """Charge reset, control and readout for EVERY measured pulse prefix."""
    if len(control_us) != len(allocation) or not len(allocation):
        raise ValueError("One shot allocation is required per prefix")
    for count in allocation:
        positive_integer(count, "shots per prefix")
    transferred = literature["transferred_scenarios"]
    ref = literature["carrasco_reference"]
    result = {"physical_shots": sum(allocation), "destructive_prefix_experiments": len(allocation),
              "system_scenarios": {}, "radiation_source_transfer_j": 0.,
              "measured_device_energy_j": None, "end_to_end_energy_j": None, "provider_bill": None}
    for scenario in transferred["system_power_scenarios"]:
        power = component_power(ref) if scenario.get("power_from") == "carrasco_components" else scenario["power_w"]
        nodes = [system_scenario(float(duration)*1e-6, power, ref, transferred["schedule_timing"],
                     shots=count, tariff=transferred["electricity_tariff_per_kwh"])
                 for duration, count in zip(control_us, allocation)]
        result["system_scenarios"][scenario["id"]] = {
            "power_w": power, "allocated_hardware_s": sum(v["cycle_s_per_shot"]*s for v, s in zip(nodes, allocation)),
            "electrical_j": sum(v["electrical_j_per_prediction"] for v in nodes),
            "kwh": sum(v["kwh_per_prediction"] for v in nodes),
            "electricity_cost": None if transferred["electricity_tariff_per_kwh"] is None else
                                sum(v["electricity_cost_per_prediction"] for v in nodes),
            "currency": transferred["currency"]}
    for duration, count in zip(control_us, allocation):
        control = Controls(float(duration)*1e-6, 0., 0., transferred.get("n_atoms", 4)*float(duration)*1e-6,
                           float(duration)*1e-6)
        source = transferred_radiation(control, literature["alves_reference"], transferred["radiation_source"]["trap_count"])
        result["radiation_source_transfer_j"] += count*source["total_radiation_j"]
    return result


def workload(rows, allocation, literature):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mnist_index"], []).append(row)
    values = [composite_energy([r["scheduled_control_us"] for r in sorted(group, key=lambda r: r["prefix"])],
                              allocation, literature) for group in grouped.values()]
    result = {"images": len(values), "shots_per_image": sum(allocation),
              "prefix_experiments_per_image": len(allocation),
              "physical_shots": len(values)*sum(allocation), "system_scenarios": {},
              "source_radiation_j_per_image": summary_stats([v["radiation_source_transfer_j"] for v in values]),
              "measured_energy_j": None, "end_to_end_pipeline_energy_j": None, "provider_bill": None}
    for name in values[0]["system_scenarios"]:
        nodes = [v["system_scenarios"][name] for v in values]
        result["system_scenarios"][name] = {
            "mean_kwh_per_image": float(np.mean([v["kwh"] for v in nodes])),
            "total_kwh": sum(v["kwh"] for v in nodes),
            "total_allocated_hardware_s": sum(v["allocated_hardware_s"] for v in nodes),
            "total_electricity_cost": None if nodes[0]["electricity_cost"] is None else
                                     sum(v["electricity_cost"] for v in nodes), "currency": nodes[0]["currency"]}
    return result


def paired_result(vqc, qrc, labels):
    vqc, qrc, labels = np.asarray(vqc), np.asarray(qrc), np.asarray(labels)
    if vqc.shape != qrc.shape or vqc.shape != labels.shape or not labels.size:
        raise ValueError("Paired predictions and labels must have the same nonempty shape")
    a, b = vqc == labels, qrc == labels
    vqc_only, qrc_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return {"qrc_minus_vqc_accuracy": float(np.mean(b)-np.mean(a)),
            "vqc_correct_qrc_wrong": vqc_only, "qrc_correct_vqc_wrong": qrc_only,
            "mcnemar_exact_two_sided_p": 1. if not vqc_only+qrc_only else
                                        float(binomtest(qrc_only, vqc_only+qrc_only, p=.5).pvalue),
            "scope": "Exploratory paired test; one reused test subset and sampling seed, multiple comparisons unadjusted"}


def finish_report(report, qrc_rows, vqc_rows, predictions, labels, output):
    config, native = report["config"], report["native_config"]
    path = Path(config["literature_config"])
    literature = json.loads(path.read_text())
    # Explicit atom count matters only to the counterfactual blue-source scaling.
    literature["transferred_scenarios"]["n_atoms"] = native["n_qubits"]
    allocation = report["protocol"]["equal_total_shots_allocation"]
    node_allocation = report["protocol"]["equal_shots_per_node_allocation"]
    shots = report["protocol"]["vqc_and_endpoint_shots_per_image"]
    test = [r for r in qrc_rows if r["split"] == "test"]
    train = [r for r in qrc_rows if r["split"] in ("train", "validation")]
    final = native["input_slots"]
    energy = {"status": "Conditional literature transfer; no measured Pasqal energy or provider pricing",
              "configuration": literature, "source_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "inference": {
                "vqc": workload(vqc_rows, [shots], literature),
                "qrc_endpoint": workload([r for r in test if r["prefix"] == final], [shots], literature),
                "qrc_temporal_equal_total_shots": workload(test, allocation, literature),
                "qrc_temporal_equal_shots_per_node": workload(test, node_allocation, literature)},
              "training_projection": {
                "qrc_endpoint": workload([r for r in train if r["prefix"] == final], [shots], literature),
                "qrc_temporal_equal_total_shots": workload(train, allocation, literature),
                "qrc_temporal_equal_shots_per_node": workload(train, node_allocation, literature)}}
    counts = native_training_count(native)
    count = sum(counts.values())
    reference = energy["inference"]["vqc"]
    energy["training_projection"]["vqc"] = {"circuit_feature_evaluation_breakdown": counts,
        "circuit_feature_evaluations": count, "shots_per_evaluation": shots, "physical_shots": count*shots,
        "system_scenarios": {name: {"total_kwh": item["mean_kwh_per_image"]*count,
            "total_electricity_cost": None if item["total_electricity_cost"] is None else
                                      item["total_electricity_cost"]/reference["images"]*count,
            "total_allocated_hardware_s": item["total_allocated_hardware_s"]/reference["images"]*count,
            "currency": item["currency"]} for name, item in reference["system_scenarios"].items()},
        "measured_energy_j": None,
        "duration_assumption": "Selected VQC's representative compiled duration; optimization pulse traces are unavailable"}
    for name, item in energy["training_projection"].items():
        if name != "vqc":
            item["circuit_feature_evaluations"] = item["images"]*item["prefix_experiments_per_image"]
            item["feature_cache_reused_for_all_linear_fits"] = True
    energy["training_scope"] = ("Hypothetical finite-shot execution of the recorded training recipes. "
        "Training actually used exact coherent expectations on CPU; noisy finite-shot hardware training "
        "would change the learned readouts and SPSA trajectory. External classical energy, loading/retries/idle "
        "allocation remain unmeasured. A 20 ms reset/readout assumption is transferred, with zero extra motion/reloads.")
    report["energy"] = energy
    report["paired_comparisons"] = {
        name+"__"+condition: paired_result(predictions["vqc__"+condition], predictions[name+"__"+condition], labels)
        for name in ("qrc_endpoint", "qrc_temporal")
        for condition in ("coherent", "device_noise_expectations", "equal_total_shots")}
    report["pulse_accounting"] = {"qrc": {str(prefix): summary_stats([r["scheduled_control_us"] for r in test
                                                                                    if r["prefix"] == prefix])
                                          for prefix in config["readout_slots"]},
                                 "vqc": summary_stats([r["scheduled_control_us"] for r in vqc_rows])}
    with (output/"control_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *qrc_rows[0].keys()], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"model": "qrc", **row} for row in qrc_rows)
        writer.writerows({"model": "vqc", **row} for row in vqc_rows)
    write_summary(report, output)
    plot(report, output)


def write_summary(report, output):
    native, config = report["native_config"], report["config"]
    energy = report["energy"]
    literature = energy["configuration"]
    transferred = literature["transferred_scenarios"]
    timing = literature["carrasco_reference"]
    power_kw = component_power(timing)/1000
    lines = ["# Pulser quantum reservoir versus variational classifier", "",
        "Local simulation on the same pinned **Pasqal FRESNEL** public device specification. "
        "This implements fixed-control QRC with a trained linear regression readout. No jobs were sent to a QPU.", "",
        "## Paired protocol", "",
        f"MNIST digits {native['digits']}; identical {native['n_train']} train / {native['n_val']} validation / "
        f"{native['n_test']} test images and archived training-only PCA; {native['n_qubits']} atoms; "
        f"{native['input_slots']} input pulse slots; seed {native['seed']}. "
        "QoolQit compiles the same global amplitude/detuning encoding; Pulser QutipBackendV2 simulates finite bandwidth. "
        "QRC fixes the VQC's archived initial pulse offsets, with no reservoir search. Only its classical head is fitted.", "",
        "The endpoint QRC has the same feature count as VQC and reproduces the existing frozen-pulse ablation. "
        "Temporal QRC concatenates Z and ZZ features from separately executed pulse prefixes. "
        "Every prefix starts from the ground state, ends with its own physical modulation tail, and requires "
        "destructive measurement. It does not obtain several hardware readouts from one shot. "
        "Reservoir memory persists between input slots within each prefix; images do not share quantum state.", "",
        "All heads are fitted on coherent training features and frozen for noisy inference. "
        "Their regularization is chosen by validation MSE from the same grid. The VQC uses its saved selected head. "
        f"Its fixed {native['iterations']}-update SPSA budget has no convergence claim. "
        f"Noisy reevaluation uses {config['noise_trajectories']} SDK noise trajectories and one sampling seed, "
        "with identical device/detector settings. Expectation results precede detector errors; sampled results include SPAM.", "",
        "The earlier VQC archive used different noise-sampling settings. Its coherent predictions are reproduced "
        "exactly here; fresh noisy scores are reported for this common protocol rather than mixed with older scores.", "",
        "## Held-out task accuracy", "",
        f"Equal total budget: **{config['total_shots_per_prediction']:,} shots per image**, allocated across "
        f"QRC temporal nodes as {report['protocol']['equal_total_shots_allocation']}. "
        "The expectation columns describe exact-readout simulations and have no physical finite-shot accuracy assigned.", "",
        "| Model | Features | Trained quantum parameters | Coherent accuracy | Noisy expectations | Noisy finite-shot accuracy (Wilson 95%) |",
        "|---|---:|---:|---:|---:|---|"]
    for name in ("vqc", "qrc_endpoint", "qrc_temporal"):
        item = report["models"][name]; c = item["conditions"]; shot = c["equal_total_shots"]
        low, high = shot["accuracy_wilson95"]
        lines.append(f"| {name} | {item['feature_count']} | {item['quantum_parameters_trained']} | "
                     f"{c['coherent']['accuracy']:.1%} | {c['device_noise_expectations']['accuracy']:.1%} | "
                     f"{shot['accuracy']:.1%} ({low:.1%}–{high:.1%}) |")
    classical = report["models"]["classical_pca_linear"]["conditions"]["coherent"]["accuracy"]
    extra = report["models"]["qrc_temporal"]["conditions"]["equal_shots_per_node"]["accuracy"]
    lines += ["", f"Classical PCA + linear regression: **{classical:.1%}**. "
        f"Temporal QRC at {config['shots_per_readout']:,} shots **per node** "
        f"({sum(report['protocol']['equal_shots_per_node_allocation']):,} total per image): **{extra:.1%}**. "
        "That additional condition spends a different hardware-shot budget. Its counts and the smaller-budget "
        "condition are correlated because the latter is a random subset of the larger sample.", "",
        "## Conditional hardware energy and electricity cost", "",
        "The same versioned [Rydberg energy papers and transfer assumptions](../../docs/literature-energy-cost.md) "
        f"are used for every model. The main table allocates the full **{power_kw:g} kW** reference machine, "
        f"{timing['reset_s']*1000:g} ms reset + {timing['measurement_s']*1000:g} ms readout per physical shot, "
        f"with the explicitly configured motion/reload counts and the illustrative "
        f"**{transferred['electricity_tariff_per_kwh']} {transferred['currency']}/kWh** tariff. "
        "The report also includes the historical vendor-power transfer. "
        "These are scenarios, not measured Pasqal consumption or service charges.", "",
        "| Workload | Shots/image | kWh/image | Test-set kWh | Test-set electricity cost |",
        "|---|---:|---:|---:|---:|"]
    for name, item in energy["inference"].items():
        s = item["system_scenarios"]["harvard_paper_reference"]
        cost = "unknown" if s["total_electricity_cost"] is None else f"{s['total_electricity_cost']:.4f} {s['currency']}"
        lines.append(f"| {name} | {item['shots_per_image']:,} | {s['mean_kwh_per_image']:.6f} | {s['total_kwh']:.6f} | {cost} |")
    lines += ["", "Every prefix pays its own reset and readout overhead. All Z/ZZ features at one node share "
        "the same measured bitstrings; there is no extra shot multiplier for feature count. "
        "Source-radiation transfers are recorded separately and never added to whole-machine electrical energy.", "",
        "## Hypothetical hardware training projection", "",
        f"| Recipe | Circuit-feature evaluations | Hypothetical physical shots | kWh, {power_kw:g} kW scenario |",
        "|---|---:|---:|---:|"]
    for name, item in energy["training_projection"].items():
        lines.append(f"| {name} | {item['circuit_feature_evaluations']:,} | {item['physical_shots']:,} | "
                     f"{item['system_scenarios']['harvard_paper_reference']['total_kwh']:.3f} |")
    lines += ["", energy["training_scope"], "",
        "QRC extracts training/validation features once and caches them for every linear fit. "
        "VQC's recorded SPSA recipe repeatedly evaluates its quantum model; the original optimizer, "
        "regularization selection and validation checkpoints are included in the count. "
        "Training-energy projections do not describe the electricity used by our CPU simulations.", "",
        "![Accuracy and projected hardware energy](qrc_vqc_comparison.png)", "",
        "## Interpretation limits", "",
        "This is an exploratory single-seed check on a small, previously inspected test subset. "
        "Wilson intervals are conditional on this run; paired discordance counts and exact McNemar p-values "
        "are included in JSON, without a multiple-comparison correction. They do not quantify reservoir-seed "
        "variation or all hardware-noise uncertainty. Temporal QRC has a larger linear head than VQC. "
        "Neither quantum method is guaranteed to match the other's accuracy. No quantum or energy advantage "
        "is established. The preserved eight-atom digital/hybrid research models remain a different architecture "
        "and are not supported on this selected global-only interface.", "",
        "This is an independent adaptation of fixed-reservoir learning, not a reproduction of published QRC "
        "accuracies. See [method and papers](../../docs/qrc-vqc-comparison.md). "
        "Actual device and end-to-end energy, external classical training/readout energy, retries, loading "
        "and provider prices remain unknown.", "",
        "[Full report and both power scenarios](report.json) · [Control ledger](control_ledger.csv) · "
        "[Compiled pulse prefixes](compiled_examples) · [Predictions](splits_predictions.npz)", "",
        "Reproduce: `.venv-hardware/bin/python -m rydberg_mnist.hardware.qrc "
        "--output hardware_results/new_qrc_comparison`.", ""]
    (output/"SUMMARY.md").write_text("\n".join(lines))


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["vqc", "qrc_endpoint", "qrc_temporal"]
    labels = ["VQC", "QRC endpoint", "QRC temporal"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    x = np.arange(3)
    for offset, condition, label in ((-.18, "coherent", "Coherent readout"),
                                    (.18, "equal_total_shots", "Device noise + matched shots")):
        metrics = [report["models"][name]["conditions"][condition] for name in names]
        values = np.array([m["accuracy"] for m in metrics])*100
        errors = np.array([[m["accuracy"]-m["accuracy_wilson95"][0],
                            m["accuracy_wilson95"][1]-m["accuracy"]] for m in metrics]).T*100
        axes[0].bar(x+offset, values, width=.34, yerr=errors, capsize=3, label=label)
    classical = report["models"]["classical_pca_linear"]["conditions"]["coherent"]["accuracy"]*100
    axes[0].axhline(classical, color="black", linestyle="--", linewidth=1, label="Classical linear")
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 105), ylabel="Held-out accuracy (%)",
                title=f"Same {report['native_config']['n_test']} MNIST 3/8 test images")
    axes[0].legend(fontsize=8, loc="lower left")
    cost_names = ["vqc", "qrc_endpoint", "qrc_temporal_equal_total_shots"]
    for offset, phase, label in ((-.18, "training_projection", "Projected training"),
                                (.18, "inference", "Test-set inference")):
        values = [report["energy"][phase][name]["system_scenarios"]["harvard_paper_reference"]["total_kwh"]
                  for name in cost_names]
        axes[1].bar(x+offset, values, width=.34, label=label)
    timing = report["energy"]["configuration"]["carrasco_reference"]
    axes[1].set(xticks=x, xticklabels=labels, yscale="log", ylabel="Conditional allocated hardware energy (kWh)",
                title=f"{component_power(timing)/1000:g} kW; "
                      f"{1000*(timing['reset_s']+timing['measurement_s']):g} ms reset + readout / shot")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fixed Rydberg reservoir versus saved variational pulses — local Pulser simulation")
    fig.supxlabel(f"{report['config']['total_shots_per_prediction']:,} total shots / image; "
                  "temporal QRC uses a larger readout; energy is a literature transfer, not a device measurement.", fontsize=9)
    fig.savefig(output/"qrc_vqc_comparison.png", dpi=180)
    fig.savefig(output/"qrc_vqc_comparison.svg")
    path = output/"qrc_vqc_comparison.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)
