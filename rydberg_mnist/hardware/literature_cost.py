"""Paper-anchored energy scenarios, separate from unknown device calibrations."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from math import isfinite
from pathlib import Path

import numpy as np

from ..data import load_mnist
from .schedule import Compiler


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def component_power(reference):
    return sum(positive_integer(c["count"], c["name"])*nonnegative(c["unit_w"], c["name"])
               for c in reference["components"])


def paper_runtime(depth, reference):
    """Carrasco-Codina et al. Eq.18 per shot, with explicit transport term."""
    positive_integer(depth, "post-compilation depth")
    period = positive_integer(reference["layers_per_reload"], "layers_per_reload")
    return (nonnegative(reference["reset_s"], "reset") + nonnegative(reference["measurement_s"], "readout")
            + depth*nonnegative(reference["clock_s"], "clock")
            + depth*nonnegative(reference["transport_per_layer"], "transport ratio")*
                    nonnegative(reference["transport_s"], "transport")
            + depth/period*nonnegative(reference["reload_s"], "reload"))


def reproduce_alves(reference):
    r = reference["reproduction"]
    computation = 1e-6*(reference["microwave_radiation_w"]*r["microwave_us"]
                 + reference["blue_source_w"]*r["blue_us"]
                 + reference["infrared_global_source_w"]*r["infrared_us"])
    preparation = (reference["cooling_source_w"]*reference["cooling_s"]
                   + reference["pumping_source_w"]*reference["pumping_s"])
    measurement = reference["measurement_beams"]*reference["measurement_source_w_per_beam"]*reference["measurement_s"]
    trapping = reference["reference_traps"]*reference["trap_source_w_per_trap"]*r["trap_on_s"]
    return {"computation_j_per_execution": computation, "trapping_j_per_execution": trapping,
            "preparation_j_per_execution": preparation, "measurement_j_per_execution": measurement,
            "radiation_j_per_execution": computation+trapping+preparation+measurement,
            "total_radiation_j": r["repetitions"]*(computation+trapping+preparation+measurement),
            "repetitions": r["repetitions"],
            "scope": "Reproduction of the paper's source-power/time tables, not wall-plug energy"}


@dataclass(frozen=True)
class Controls:
    duration_s: float
    xy_target_s: float
    stark_target_s: float
    rydberg_blue_target_s: float
    rydberg_infrared_beam_s: float

    @classmethod
    def from_schedule(cls, schedule):
        duration = xy = stark = blue = infrared = 0.
        for segment in schedule:
            dt = segment.duration_us*1e-6
            duration += dt
            if segment.channel == "raman_local":
                xy += len(segment.targets)*dt
            elif segment.channel == "stark_local":
                stark += len(segment.targets)*dt
            elif segment.channel in ("rydberg_local", "rydberg_global"):
                blue += len(segment.targets)*dt
                infrared += dt  # One shared IR beam, even when multiple atoms are illuminated.
            else:
                raise ValueError(f"No literature mapping for {segment.channel}")
        return cls(duration, xy, stark, blue, infrared)

    @classmethod
    def from_native_ledger(cls, energy, n):
        duration = energy["scheduled_control_window_us_per_shot"]*1e-6
        return cls(duration, 0., 0., n*duration, duration)


def transferred_radiation(controls, reference, trap_count):
    """Fixed reference on-powers; explicitly NOT calibration from effective Rabi rate."""
    positive_integer(trap_count, "trap_count")
    gate = (reference["microwave_radiation_w"]*controls.xy_target_s
            + reference["blue_source_w"]*(controls.stark_target_s+controls.rydberg_blue_target_s)
            + reference["infrared_global_source_w"]*controls.rydberg_infrared_beam_s)
    overhead = reference["cooling_s"]+reference["pumping_s"]+reference["measurement_s"]
    traps = trap_count*reference["trap_source_w_per_trap"]*(overhead+controls.duration_s)
    preparation = reference["cooling_source_w"]*reference["cooling_s"] + reference["pumping_source_w"]*reference["pumping_s"]
    readout = reference["measurement_beams"]*reference["measurement_source_w_per_beam"]*reference["measurement_s"]
    return {"control_radiation_j": gate, "traps_radiation_j": traps,
            "preparation_radiation_j": preparation, "readout_radiation_j": readout,
            "total_radiation_j": gate+traps+preparation+readout,
            "cycle_s": overhead+controls.duration_s,
            "wall_plug_energy_j": None,
            "scope": "Counterfactual transfer of constant paper source powers; "
                     "Raman/microwave equivalence, global blue scaling and fixed on-power are assumptions"}


def system_scenario(control_s, power_w, timing_reference, timing_assumptions, *, shots, tariff=None):
    """Constant whole-machine power × transferred overhead + compiled control time.

    No radiation energy is added: source power is already represented within
    whole-machine electrical consumption. The gate-clock D is not guessed from
    VQC ansatz layers; optional reconfiguration/transport counts are explicit.
    """
    positive_integer(shots, "shots")
    control_s, power_w = nonnegative(control_s, "control time"), nonnegative(power_w, "power")
    extra = sum(nonnegative(timing_assumptions[key], key)*nonnegative(timing_reference[ref], ref)
                for key, ref in (("transport_events", "transport_s"),
                                 ("reconfiguration_layers", "clock_s"), ("reloads_per_shot", "reload_s")))
    cycle = (control_s+nonnegative(timing_reference["reset_s"], "reset")
             + nonnegative(timing_reference["measurement_s"], "readout")+extra)
    energy = power_w*cycle
    kwh = energy*shots/3.6e6
    return {"power_w": power_w, "cycle_s_per_shot": cycle,
            "scheduled_control_s_per_shot": control_s, "extra_motion_reload_s_per_shot": extra,
            "electrical_j_per_shot": energy, "shots_per_prediction": shots,
            "electrical_j_per_prediction": energy*shots, "kwh_per_prediction": kwh,
            "electricity_cost_per_prediction": None if tariff is None else kwh*nonnegative(tariff, "tariff"),
            "end_to_end_pipeline_energy_j": None, "provider_bill": None}


def summary_stats(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def collect_inputs(root, cache):
    """Compile existing held-out inputs; no classification training or simulation."""
    from .experiment import model_inputs
    from .pasqal import load_target, sequence_ledger
    from .pasqal_native import make_sequence
    from .schedule import load_profile
    from ..compare import validate_pair
    root = Path(root)
    _, _, images, _ = load_mnist(cache)
    analog_dir, digital_dir = root/"results/mnist_3_8_seed7", root/"results/mnist_3_8_digital_seed7"
    validate_pair(analog_dir, digital_dir)
    profile = load_profile(root/"configs/hardware_research.json")
    groups = {}
    for directory in (analog_dir, digital_dir):
        with np.load(directory/"splits_predictions.npz") as split:
            indices = split["test_idx"]
        physics, architecture, angles, theta, _, _, _ = model_inputs(directory/"model.npz", images[indices])
        compiler = Compiler(physics, profile)
        rows = [{"mnist_test_index": int(index), "controls": Controls.from_schedule(compiler.compile(angle, theta, architecture))}
                for index, angle in zip(indices, angles)]
        groups[architecture] = {"n_qubits": physics.n_qubits, "inputs": rows,
             "source_model": str(directory/"model.npz"),
             "source_model_sha256": hashlib.sha256((directory/"model.npz").read_bytes()).hexdigest(),
             "status": "Research gate/transfer recipe; not supported on selected Pasqal interface; unequal physical fidelity"}
        print("Compiled cost inputs", architecture, len(rows), flush=True)
    directory = root/"hardware_results/pasqal_native_mnist_3_8_seed7"
    with np.load(directory/"model.npz", allow_pickle=False) as model, np.load(directory/"splits_predictions.npz") as split:
        config = json.loads(str(model["config_json"]))
        indices = split["test_idx"]
        values = (images[indices]/255.-model["pca_mean"])@model["pca_components"].T
        values = np.clip(values/model["pca_scale"], -3, 3)
        target, device = load_target(root/config["target"])
        theta = model["parameters"]
        rows = []
        for index, features in zip(indices, values):
            sequence = make_sequence(device, config, features, theta)
            energy, _ = sequence_ledger(sequence, target, modulation=True, shots=1)
            rows.append({"mnist_test_index": int(index),
                         "controls": Controls.from_native_ledger(energy, config["n_qubits"])})
        groups["pasqal_native_analog"] = {"n_qubits": config["n_qubits"], "inputs": rows,
                  "source_model": str(directory/"model.npz"),
                  "source_model_sha256": hashlib.sha256((directory/"model.npz").read_bytes()).hexdigest(),
                  "target_device_sha256": target["device_sha256"],
                  "status": "Supported SDK global-pulse model; different data/architecture budget from original pair",
                  "native_config": config}
        print("Compiled cost inputs pasqal_native_analog", len(rows), flush=True)
    return groups


def native_training_count(config):
    """Projected circuit-feature evaluations of this exact SPSA training recipe."""
    iterations, interval = config["iterations"], config["checkpoint_every"]
    checkpoints = 1+len([i for i in range(1, iterations+1) if i%interval == 0 or i == iterations])
    return {"initial_alpha_selection": config["n_train"]+config["n_val"],
            "spsa_objective_inputs": 2*iterations*config["n_train"],
            "validation_checkpoint_inputs": checkpoints*(config["n_train"]+config["n_val"]),
            "final_readout_fit_inputs": config["n_train"]}


def estimate(config_path, output, root=".", cache="data/mnist"):
    path = Path(config_path)
    config = json.loads(path.read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    groups = collect_inputs(root, cache)
    ref, wall = config["alves_reference"], config["carrasco_reference"]
    transferred = config["transferred_scenarios"]
    scenarios = {s["id"]: {"power_w": component_power(wall) if s.get("power_from") == "carrasco_components" else s["power_w"],
                           "source": s["source"]} for s in transferred["system_power_scenarios"]}
    shots, tariff = transferred["shots_per_prediction"], transferred["electricity_tariff_per_kwh"]
    report = {"status": "Literature-anchored scenarios, not measured/calibrated Pasqal costs or energy advantage",
              "configuration": config, "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "reproduction_alves": reproduce_alves(ref),
              "carrasco_table8_power_w": component_power(wall),
              "carrasco_equation18_reference_times_s": {str(d): paper_runtime(d, wall) for d in (10, 100, 1000)},
              "system_scenarios": scenarios, "architectures": {},
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "qoolqit", "pulser-core", "matplotlib")},
              "research_profile_sha256": hashlib.sha256((Path(root)/"configs/hardware_research.json").read_bytes()).hexdigest(),
              "training_hardware_projection": {}, "sensitivity": []}
    flat_rows = []
    for architecture, group in groups.items():
        radiation, system = [], {key: [] for key in scenarios}
        for row in group["inputs"]:
            controls = row["controls"]
            source = transferred_radiation(controls, ref, transferred["radiation_source"]["trap_count"])
            radiation.append(source)
            record = {"architecture": architecture, "mnist_test_index": row["mnist_test_index"],
                      "control_us": controls.duration_s*1e6, "radiation_scenario_j_per_shot": source["total_radiation_j"]}
            for name, scenario in scenarios.items():
                value = system_scenario(controls.duration_s, scenario["power_w"], wall,
                        transferred["schedule_timing"], shots=shots, tariff=tariff)
                system[name].append(value)
                record[name+"_j_per_prediction"] = value["electrical_j_per_prediction"]
            flat_rows.append(record)
        item = {"n_qubits": group["n_qubits"], "held_out_inputs": len(group["inputs"]),
                "source_model": group["source_model"], "implementation_status": group["status"],
                "source_model_sha256": group["source_model_sha256"],
                "scheduled_control_us": summary_stats([x["controls"].duration_s*1e6 for x in group["inputs"]]),
                "radiation_scenario_j_per_shot": summary_stats([x["total_radiation_j"] for x in radiation]),
                "control_radiation_j_per_shot": summary_stats([x["control_radiation_j"] for x in radiation]),
                "system_scenarios": {}, "measured_energy_j": None, "provider_bill": None,
                "hardware_training_energy_j": None}
        for name, values in system.items():
            mean = {key: summary_stats([v[key] for v in values]) for key in
                    ("cycle_s_per_shot", "electrical_j_per_shot", "electrical_j_per_prediction", "kwh_per_prediction")}
            dataset_kwh = sum(v["kwh_per_prediction"] for v in values)
            item["system_scenarios"][name] = {"statistics": mean, "held_out_dataset_kwh": dataset_kwh,
                "held_out_dataset_electricity_cost": None if tariff is None else dataset_kwh*tariff,
                "currency": transferred["currency"], "electricity_tariff_per_kwh": tariff}
        report["architectures"][architecture] = item
        if architecture == "pasqal_native_analog":
            item["target_device_sha256"] = group["target_device_sha256"]
            counts = native_training_count(group["native_config"])
            executions = sum(counts.values())
            for name, values in system.items():
                report["training_hardware_projection"][name] = {
                    "circuit_feature_evaluation_breakdown": counts, "total_circuit_feature_evaluations": executions,
                    "assumed_shots_per_feature_evaluation": shots, "physical_shots_if_run": executions*shots,
                    "estimated_allocated_hardware_hours": executions*shots*np.mean([v["cycle_s_per_shot"] for v in values])/3600,
                    "estimated_hardware_kwh": executions*np.mean([v["kwh_per_prediction"] for v in values]),
                    "estimated_electricity_cost": None if tariff is None else executions*np.mean([v["kwh_per_prediction"] for v in values])*tariff,
                    "actual_hardware_energy_j": None,
                    "scope": "Hypothetical finite-shot replay budget, using selected model's representative duration. "
                             "Actual training was a CPU simulation with exact expectations; "
                             "SPSA trajectory/accuracy would change on noisy finite-shot hardware. "
                             "Original ideal-adjoint training has no inferred hardware gradient budget."}
            duration = item["scheduled_control_us"]["mean"]*1e-6
            for name, scenario in scenarios.items():
                for overhead in transferred["reset_plus_readout_sensitivity_s"]:
                    for number_shots in transferred["shot_sensitivity"]:
                        energy = scenario["power_w"]*(duration+overhead)*number_shots
                        report["sensitivity"].append({"scenario": name, "assumed_reset_plus_readout_s": overhead,
                             "shots": number_shots, "electrical_j_per_prediction": energy,
                             "electricity_cost": None if tariff is None else energy/3.6e6*tariff})
    (output/"estimates.json").write_text(json.dumps(report, indent=2, default=float)+"\n")
    with (output/"per_input_estimates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat_rows)
    plot_scenarios(report, output)
    write_summary(report, output)
    return report


def plot_scenarios(report, output):
    """Standalone figures keep accounting boundaries and assumptions explicit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = report["architectures"]
    labels = ["Hybrid research\n8 atoms", "Digital research\n8 atoms", "Pasqal-native\n4 atoms"]
    scenarios = report["system_scenarios"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    x = np.arange(len(groups))
    for index, (name, scenario) in enumerate(scenarios.items()):
        power_label = ("Paper machine: 22.205 kW" if name == "harvard_paper_reference"
                       else "Historical vendor transfer: 2.6 kW")
        values = [g["system_scenarios"][name]["statistics"]["electrical_j_per_shot"]["mean"] for g in groups.values()]
        axes[0].bar(x+(index-.5)*.34, values, width=.34, label=power_label)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Whole-machine electrical J / shot",
                title="20 ms transferred reset + readout")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.2)
    for index, (name, scenario) in enumerate(scenarios.items()):
        values = sorted((s for s in report["sensitivity"] if s["scenario"] == name and s["shots"] == 1000),
                        key=lambda s: s["assumed_reset_plus_readout_s"])
        axes[1].loglog([v["assumed_reset_plus_readout_s"]*1000 for v in values],
                       [v["electrical_j_per_prediction"]/3.6e6 for v in values],
                       marker="o", label=f'{scenario["power_w"]/1000:g} kW')
    axes[1].set(xlabel="Assumed reset + readout (ms)", ylabel="Whole-machine kWh / prediction",
                title="Native model; 1,000 shots / prediction")
    axes[1].legend()
    axes[1].grid(which="both", alpha=.2)
    fig.suptitle("Conditional literature transfers: uncalibrated power and shot overhead")
    fig.supxlabel("Research recipes are unsupported on the selected Pasqal interface; fidelity and accuracy are unmatched.",
                  fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output/f"electrical_scenarios.{extension}", dpi=180)
    svg = output/"electrical_scenarios.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


def write_summary(report, output):
    config = report["configuration"]
    transferred = config["transferred_scenarios"]
    reproduction = report["reproduction_alves"]
    lines = ["# Literature-based Rydberg energy/cost scenarios", "",
             "These estimates transfer published Rydberg-hardware parameters to our compiled "
             "schedules. They are conditional scenarios, not measurements or calibration of Pasqal "
             "FRESNEL. Radiation-source and whole-machine electrical energy are different boundaries; "
             "do not add them together.", "",
             "## Sources and reproducibility", "",
             "- [Alves, Pezzutto and Omar, arXiv:2601.03141v2](https://arxiv.org/html/2601.03141v2), "
             "Sections IV.1–IV.2: source powers, timing and trapping/preparation/readout budget. "
             "Cesium experimental reference; excludes wall-plug conversion and classical/environmental costs.",
             "- [Carrasco-Codina et al., arXiv:2605.15090v2](https://arxiv.org/html/2605.15090v2), "
             "Section 6.2, Eq.18 and Tables 8–9: full electrical component inventory and operation times. "
             "Harvard rubidium reference, not a Pasqal measurement.",
             "- [Pasqal, 5 October 2023](https://www.pasqal.com/blog/quantum-computing-rethinking-energy-consumption/): "
             "historical approximately 2.6 kW vendor statement, used only as a power-transfer sensitivity scenario.", "",
             f"Reproducing Alves' tabulated arithmetic gives {reproduction['computation_j_per_execution']*1000:.6f} mJ "
             f"of computation radiation per execution and {reproduction['total_radiation_j']:.6f} J for "
             "700 complete executions, matching its rounded 0.662 mJ and 69.5 J. "
             f"Summing Carrasco-Codina Table 8 baseline components gives {report['carrasco_table8_power_w']:.0f} W "
             "(the paper rounds this to approximately 22 kW); continuous-reload extras are not added.", "",
             "The paper's Eq.18 is also implemented exactly. Its transport term is kept explicit; "
             "its approximate prose examples do not exactly reproduce Eq.18 with beta=0.25. "
             "We do not identify a two-layer VQC with depth D=2 in that equation. "
             "Our fixed-register schedule scenario instead uses actual compiled control time, "
             "10 ms reset + 10 ms readout transferred from Table 9, and explicitly assumes "
             "zero extra motion/reconfiguration/reloading. Those omitted overheads need calibration.", "",
             "## Whole-machine electrical scenario", "",
             f"Assume {transferred['shots_per_prediction']} shots per prediction. "
             f"Illustrative electricity tariff: {transferred['electricity_tariff_per_kwh']} "
             f"{transferred['currency']}/kWh. This is not a market quote or a cloud-provider price. "
             "The full reference machine's power is allocated to the workload; it is not divided "
             "by qubit count or added to the radiation estimate.", "",
             "| Implementation | Qubits | Test inputs | Mean control window | 22.205 kW reference: J/shot | kWh/prediction |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["architectures"].items():
        s = item["system_scenarios"]["harvard_paper_reference"]["statistics"]
        lines.append(f"| {name} | {item['n_qubits']} | {item['held_out_inputs']} | "
                     f"{item['scheduled_control_us']['mean']:.3f} us | "
                     f"{s['electrical_j_per_shot']['mean']:.3f} | {s['kwh_per_prediction']['mean']:.6f} |")
    native = report["architectures"]["pasqal_native_analog"]
    lines += ["", "For the Pasqal-native classifier's 40 held-out inputs:", "",
              "| Assumed power envelope | Mean J/shot | kWh for 40 inputs | Electricity cost for 40 inputs |",
              "|---|---:|---:|---:|"]
    for name, scenario in native["system_scenarios"].items():
        cost = scenario["held_out_dataset_electricity_cost"]
        price = "Unknown" if cost is None else f"{cost:.4f} {scenario['currency']}"
        lines.append(f"| {name} | {scenario['statistics']['electrical_j_per_shot']['mean']:.3f} | "
                     f"{scenario['held_out_dataset_kwh']:.6f} | {price} |")
    lines += ["", "These scenarios mix a specified power envelope with transferred shot overheads; "
              "they are not confidence intervals, bounds, or current Pasqal energy measurements. "
              "Preparation/readout dominate at the assumed rates, so pulse-duration ratios "
              "must not be presented as full-machine energy-saving ratios.", "",
              "![Electrical scenarios and shot-overhead sensitivity](electrical_scenarios.png)", "",
              "## Radiation-source transfer scenario", "",
              "This optional counterfactual holds the paper's source on-powers fixed, charges "
              "Raman control at its derived microwave radiation power, assumes one blue reference "
              "power per illuminated atom, and counts shared IR once. Trap/preparation/readout "
              "use the paper's 49-trap reference. It is **not** a calibration from our effective "
              "Rabi rates, wavelengths, beam geometry or atomic species. Its architecture ranking "
              "can change with those assumptions. Electrical energy remains unknown at this boundary.", "",
              "| Implementation | Mean source-radiation J/shot | Control-only source-radiation J/shot |",
              "|---|---:|---:|"]
    for name, item in report["architectures"].items():
        lines.append(f"| {name} | {item['radiation_scenario_j_per_shot']['mean']:.9f} | "
                     f"{item['control_radiation_j_per_shot']['mean']:.9f} |")
    lines += ["", "## Hypothetical hardware training budget", "",
              "The native SPSA recipe can be projected into circuit-feature evaluations, but "
              "it was executed on the CPU with exact expectation values. Assuming 1,000 shots "
              "per feature evaluation would change the optimization trajectory and attainable accuracy.", ""]
    for name, projection in report["training_hardware_projection"].items():
        lines.append(f"- {name}: {projection['total_circuit_feature_evaluations']} circuit-feature evaluations, "
                     f"{projection['physical_shots_if_run']:,} hypothetical physical shots, "
                     f"{projection['estimated_allocated_hardware_hours']:.3f} allocated hardware hours, "
                     f"{projection['estimated_hardware_kwh']:.3f} kWh. This is not the CPU simulation's energy.")
    lines += ["", "No hardware-training budget is inferred for the original exact-adjoint optimizers. "
              "Their future hardware gradient method and sampling plan are unspecified.", "",
              "## Limits", "",
              "The original digital/hybrid recipes cannot run as written on the selected global-only "
              "Pasqal interface. The native analog model uses a different architecture/data budget. "
              "Physical fidelities and accuracies are not matched; these tables are accounting "
              "scenarios, not an energy-advantage benchmark. Real wall power, readout/reset/loading "
              "times, failed-shot handling, idle allocation and external classical preprocessing/"
              "training/inference energy remain unmeasured. Manufacturing and embodied energy "
              "are outside scope. No cloud billing, tariffs or provider charges are inferred.", "",
              "[Parameter provenance](../../configs/literature_energy.json) · "
              "[Machine-readable estimates](estimates.json) · [Per-input estimates](per_input_estimates.csv)", "",
              "Reproduce: `.venv-hardware/bin/python -m rydberg_mnist.hardware.literature_cost "
              "--output estimates/new_literature_run`.", ""]
    (output/"SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/literature_energy.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--cache", default="data/mnist")
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        estimate(args.config, args.output, args.root, args.cache)


if __name__ == "__main__":
    main()
