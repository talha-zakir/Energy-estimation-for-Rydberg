"""Control exposure and calibrated energy accounting; missing data stays unknown."""
from collections import defaultdict
from math import isfinite


def ledger(segments, profile, shots=1):
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    channels = defaultdict(lambda: {"beam_time_us": 0., "omega_squared_integral_rad2_per_us": 0.})
    stages = defaultdict(float)
    duration = 0.
    for s in segments:
        # Global amplitude refers to the entire beam, not one laser per atom.
        beams = 1 if s.channel.endswith("_global") else len(s.targets)
        a, b, dt = s.amplitude_start, s.amplitude_stop, s.duration_us
        channels[s.channel]["beam_time_us"] += beams*dt
        channels[s.channel]["omega_squared_integral_rad2_per_us"] += beams*dt*(a*a+a*b+b*b)/3
        stages[s.stage] += dt
        duration += dt
    optical, electrical, missing = 0., 0., []
    electrical_known = True
    for channel, item in channels.items():
        calibration = profile.get("power_calibration", {}).get(channel, {})
        bias, coefficient = calibration.get("bias_w"), calibration.get("omega_squared_w")
        if bias is None or coefficient is None:
            item["optical_j_per_shot"] = None
            item["electrical_j_per_shot"] = None
            missing.append(f"{channel}: beam-power calibration")
            electrical_known = False
            continue
        if not isfinite(bias) or not isfinite(coefficient) or bias < 0 or coefficient < 0:
            raise ValueError("Power calibrations must be nonnegative")
        energy = 1e-6*(bias*item["beam_time_us"] + coefficient*item["omega_squared_integral_rad2_per_us"])
        item["optical_j_per_shot"] = energy
        optical += energy
        eta = calibration.get("wall_efficiency")
        if eta is None:
            electrical_known = False
            item["electrical_j_per_shot"] = None
            missing.append(f"{channel}: wall efficiency")
        else:
            if not isfinite(eta) or not 0 < eta <= 1:
                raise ValueError("Wall efficiency must lie in (0,1]")
            item["electrical_j_per_shot"] = energy/eta
            electrical += energy/eta
    optical_known = all(v["optical_j_per_shot"] is not None for v in channels.values())
    overhead = profile.get("overhead", {})
    timing = [overhead.get(k) for k in ("preparation_us", "readout_us", "reset_us")]
    total_time = duration+sum(timing) if all(t is not None for t in timing) else None
    background, fixed = overhead.get("background_power_w"), overhead.get("fixed_energy_j_per_shot")
    for k, v in overhead.items():
        if v is not None and (not isfinite(v) or v < 0):
            raise ValueError(f"Negative overhead {k}")
    complete = electrical_known and total_time is not None and background is not None and fixed is not None
    full_j = electrical + background*total_time*1e-6 + fixed if complete else None
    if not complete:
        missing.append("complete cycle timing/background and fixed overhead energy")
    return {"calibration_status": profile["calibration_status"],
            "scheduled_control_window_us_per_shot": duration,
            "full_cycle_us_per_shot": total_time, "stage_duration_us": dict(stages),
            "channels": dict(channels), "shots": shots,
            "control_optical_j_per_shot": optical if optical_known else None,
            "control_electrical_j_per_shot": electrical if electrical_known else None,
            "full_quantum_cycle_j_per_shot": full_j,
            "full_quantum_cycle_j_for_requested_shots": full_j*shots if complete else None,
            "classical_preprocessing_training_readout_energy_j": None,
            "end_to_end_pipeline_energy_j": None,
            "missing_calibrations": missing,
            "scope": "Laser coefficients use P(t)=bias+k*Omega_eff(t)^2 in W. "
                     "A two-photon implementation requires calibration including BOTH optical legs. "
                     "Exposure is not energy. Training/CPU/GPU energy is not inferred from runtime."}


def workload_budget(per_shot, examples, shots, quantum_circuit_evaluations_per_example=1):
    """Explicit execution multiplier, not an estimate of hardware training gradients."""
    counts = (examples, shots, quantum_circuit_evaluations_per_example)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in counts):
        raise ValueError("Workload counts must be positive integers")
    executions = examples*shots*quantum_circuit_evaluations_per_example
    energy = per_shot["full_quantum_cycle_j_per_shot"]
    return {"physical_shot_executions": executions,
            "scheduled_control_time_us": executions*per_shot["scheduled_control_window_us_per_shot"],
            "quantum_cycle_energy_j": None if energy is None else executions*energy,
            "end_to_end_energy_j": None}
