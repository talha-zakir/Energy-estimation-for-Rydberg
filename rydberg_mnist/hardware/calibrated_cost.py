"""Traceable device measurements, complete-workflow budgets, and separate scenarios."""
import argparse
import csv
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .literature_cost import component_power


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def validate_calibration(calibration, target):
    if calibration.get("schema_version") != 1 or calibration.get("boundary") != "whole_machine_electrical":
        raise ValueError("Expected a version-1 whole-machine electrical calibration")
    for key in ("target_device", "device_sha256"):
        if calibration[key] != target[key]:
            raise ValueError(f"Calibration {key} does not match the selected device")
    fields = ("electrical_power_w", "full_cycle_s_per_attempt", "attempts_per_returned_shot",
              "setup_energy_j_per_program", "external_classical_energy_j")
    for key in fields:
        if calibration[key] is not None:
            finite_nonnegative(calibration[key], key)
    if calibration["attempts_per_returned_shot"] is not None and calibration["attempts_per_returned_shot"] < 1:
        raise ValueError("Attempts per returned shot must be at least one")
    if calibration["full_cycle_s_per_attempt"] == 0 or calibration["electrical_power_w"] == 0:
        raise ValueError("Measured power and cycle time must be positive")
    if not calibration["cycle_includes_control_and_loading_readout_reset"]:
        raise ValueError("Supply a complete measured cycle; control time must not be counted twice")
    if calibration.get("cycle_includes_program_setup", False) and calibration["setup_energy_j_per_program"] not in (None, 0):
        raise ValueError("Program setup is already in the full cycle; do not charge it again")
    if calibration["calibration_status"] == "measured":
        provenance = calibration["provenance"]
        if not all(provenance.get(key) for key in ("measured_at_utc", "instrument_or_provider_record", "measurement_scope")):
            raise ValueError("Measured calibrations require dated, scoped measurement provenance")
        if any(calibration[key] is None for key in fields[:3]):
            raise ValueError("Measured shot-energy calibration needs power, full cycle and attempt factor")
    elif calibration["calibration_status"] != "unavailable":
        raise ValueError("Calibration status must be measured or unavailable")


def budget(records, calibration, target, literature, classical_energy_j=None):
    """Counts all requested shots, including search, failed target attempts and validation."""
    validate_calibration(calibration, target)
    for record in records:
        if isinstance(record["shots"], bool) or not isinstance(record["shots"], int) or record["shots"] < 1:
            raise ValueError("Execution records require positive integer shot counts")
        finite_nonnegative(record["control_us"], "compiled control duration")
    shots = sum(r["shots"] for r in records)
    control_s = sum(r["control_us"]*1e-6*r["shots"] for r in records)
    power, cycle, attempts = (calibration[key] for key in
                             ("electrical_power_w", "full_cycle_s_per_attempt", "attempts_per_returned_shot"))
    measured = calibration["calibration_status"] == "measured"
    shot_energy = power*cycle*attempts*shots if measured else None
    setup = calibration["setup_energy_j_per_program"]
    setup_included = calibration.get("cycle_includes_program_setup", False)
    setup_energy = (0. if setup_included else len(records)*setup) if measured and (setup_included or setup is not None) else None
    # External classical measurements must belong to THIS workload, not a
    # calibration run. The supplied per-run value is never scaled by shot count.
    classical = classical_energy_j
    if classical is not None:
        finite_nonnegative(classical, "classical energy")
    total = shot_energy+setup_energy+classical if all(v is not None for v in (shot_energy, setup_energy, classical)) else None
    # A per-program wall-meter measurement includes that program's setup and
    # attempts already. Never add a modeled setup/cycle estimate on top of it.
    measured_programs = [r.get("measured_whole_program_electrical_j") for r in records]
    program_meter_total = (sum(finite_nonnegative(v, "metered program energy") for v in measured_programs)
                           if records and all(v is not None for v in measured_programs) else None)
    if program_meter_total is not None and not all(r.get("measurement_source_record") for r in records):
        raise ValueError("Program meter totals require measurement provenance for every request")
    measured_total = program_meter_total+classical if program_meter_total is not None and classical is not None else None
    transferred = literature["transferred_scenarios"]
    reference = literature["carrasco_reference"]
    scenarios = {}
    for scenario in transferred["system_power_scenarios"]:
        p = component_power(reference) if scenario.get("power_from") == "carrasco_components" else scenario["power_w"]
        # Full-machine reference transfer retains the explicitly configured
        # reset/readout, transport, reconfiguration and reload overheads.
        timing = transferred["schedule_timing"]
        overhead = (reference["reset_s"]+reference["measurement_s"]+timing["transport_events"]*reference["transport_s"]
                    + timing["reconfiguration_layers"]*reference["clock_s"]+timing["reloads_per_shot"]*reference["reload_s"])
        energy = p*(shots*overhead+control_s)
        scenarios[scenario["id"]] = {"electrical_j": energy, "kwh": energy/3.6e6,
            "scope": "Transferred literature power and operation times; assumes successful shots without retry charge"}
    nominal = literature.get("fresnel_beta_nominal_throughput")
    if nominal:
        # A published throughput is not a measured cycle calibration. It already
        # includes the control slot; do not add pulse duration to it again.
        for scenario in transferred["system_power_scenarios"]:
            p = component_power(reference) if scenario.get("power_from") == "carrasco_components" else scenario["power_w"]
            energy = p*shots/nominal["shots_per_s"]
            scenarios["nominal_beta_throughput__"+scenario["id"]] = {
                "electrical_j": energy, "kwh": energy/3.6e6,
                "scope": "Conditional power transfer times published beta throughput; neither is account-specific metering"}
    tariff = transferred["electricity_tariff_per_kwh"]
    for item in scenarios.values():
        item["electricity_cost"] = None if tariff is None else item["kwh"]*tariff
        item["currency"] = transferred["currency"]
    return {"requested_returned_shots": shots, "pulse_program_requests": len(records),
        "compiled_control_s_for_requested_shots": control_s,
        "calibrated_shot_execution_energy_j": shot_energy, "calibrated_setup_energy_j": setup_energy,
        "measured_external_classical_energy_j": classical, "complete_workflow_energy_j": total,
        "measured_program_electrical_energy_j": program_meter_total,
        "complete_measured_workflow_energy_j": measured_total,
        "complete_workflow_kwh": None if total is None else total/3.6e6,
        "calibration_status": calibration["calibration_status"], "literature_scenarios": scenarios,
        "missing_measurements": [key for key in ("electrical_power_w", "full_cycle_s_per_attempt",
            "attempts_per_returned_shot", "setup_energy_j_per_program") if calibration[key] is None
            and not (key == "setup_energy_j_per_program" and setup_included)]
            + (["external_classical_energy_for_this_workload_j"] if classical is None else []),
        "hardware_execution_energy_measured_here": program_meter_total is not None, "provider_bill": None}


def integrate_power_trace(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("Power trace needs at least two time/power samples")
    times = np.array([float(row["time_s"]) for row in rows])
    powers = np.array([float(row["power_w"]) for row in rows])
    if not np.all(np.isfinite([times, powers])) or np.any(np.diff(times) <= 0) or np.any(powers < 0):
        raise ValueError("Trace times must increase and powers must be finite and nonnegative")
    return {"energy_j": float(np.trapezoid(powers, times)), "duration_s": float(times[-1]-times[0]),
            "mean_power_w": float(np.trapezoid(powers, times)/(times[-1]-times[0])),
            "scope": "Integrated meter trace only; device identity, workload scope and attribution require metadata"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(integrate_power_trace(args.power_trace), indent=2)+"\n")


if __name__ == "__main__":
    main()
