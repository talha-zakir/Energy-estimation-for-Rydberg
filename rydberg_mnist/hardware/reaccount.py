"""Recompute saved workload energy from updated measurements without retraining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .calibrated_cost import budget, finite_nonnegative, validate_calibration


def read_ledger(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("request_id", "checkpoint", "source_index", "prefix", "shots", "noise_seed"):
            row[key] = int(row[key])
        for key in ("control_us", "omega_squared_integral_rad2_per_us"):
            row[key] = float(row[key]) if row[key] else None
        for key in ("measured_attempted_shots", "measured_active_duration_s", "measured_whole_program_electrical_j"):
            value = row.get(key)
            row[key] = None if not value else (int(value) if key == "measured_attempted_shots" else float(value))
    return rows


def classical_measurement(evidence, key, target):
    if evidence is None or key not in evidence.get("workloads", {}):
        return None
    if evidence.get("device_sha256") != target["device_sha256"]:
        raise ValueError("Classical measurement allocation refers to a different device study")
    item = evidence["workloads"][key]
    if not all(item.get(k) for k in ("measured_at_utc", "source_record", "scope")):
        raise ValueError("Classical energy requires a dated, scoped meter source record")
    return finite_nonnegative(item["energy_j"], "classical workload energy")


def recompute(report, source, calibration, classical=None):
    target = report["target"]
    validate_calibration(calibration, target)
    literature = report["literature_configuration"]
    for repetition in report["repeats"]:
        seed = repetition["seed"]
        records = read_ledger(Path(source)/f"seed{seed}"/"execution_ledger.csv")
        def energy(rows, key, components=()):
            value = classical_measurement(classical, key, target)
            if value is None and components:
                values = [classical_measurement(classical, part, target) for part in components]
                value = sum(values) if all(v is not None for v in values) else None
            return budget(rows, calibration, target, literature, value)
        for algorithm in ("qrc", "vqc"):
            own = [r for r in records if r["algorithm"] == algorithm]
            repetition["complete_study_execution_budget"][algorithm] = energy(own, f"seed{seed}/{algorithm}/full_study")
        for row in repetition["targets"].values():
            for algorithm, item in row["algorithms"].items():
                training = [r for r in records if r["algorithm"] == algorithm and r["stage"] != "test"]
                if item["status"] == "not_reached_within_budget":
                    item["training_budget_spent"] = energy(training, f"seed{seed}/{algorithm}/all_training")
                    continue
                checkpoint = item["checkpoint"]
                training = [r for r in records[:checkpoint["ledger_end"]]
                            if r["algorithm"] == algorithm and r["stage"] != "test"]
                inference = [r for r in records if r["algorithm"] == algorithm and r["stage"] == "test"
                             and r["checkpoint"] == checkpoint["iteration"]]
                root = f"seed{seed}/{algorithm}"
                train_key, test_key = f"{root}/train_to_step{checkpoint['iteration']}", f"{root}/test_step{checkpoint['iteration']}"
                item["energy_to_validation_target"] = energy(training, train_key)
                item["held_out_inference_energy"] = energy(inference, test_key)
                item["training_plus_held_out_inference_energy"] = energy(training+inference,
                    f"{root}/train_and_test_step{checkpoint['iteration']}", components=(train_key, test_key))
    report["calibration"] = calibration
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed accuracy_energy run directory")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--classical-measurements")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    report_path = source/"report.json"
    report = json.loads(report_path.read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    classical = None if not args.classical_measurements else json.loads(Path(args.classical_measurements).read_text())
    recompute(report, source, calibration, classical)
    report["reaccounting"] = {"source_directory": str(source),
        "original_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest(),
        "physics_or_training_rerun": False}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    from .accuracy_energy_summary import summarize
    summarize(report, output)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
