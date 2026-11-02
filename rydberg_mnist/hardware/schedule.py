"""Compile logical circuits into finite-duration controls on |0>, |1>, |r>."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ..quantum import cnot_pairs


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    if profile.get("kind") == "pasqal_sdk_target":
        raise ValueError("Pasqal target configs require the QoolQit/Pulser backend; "
                         "they do not support the three-level research compiler's local gates")
    for key in ("spacing_um", "c6_rad_us_um6", "raman_omega_rad_us", "stark_rate_rad_us",
                "gate_omega_rad_us", "transfer_omega_rad_us"):
        if not np.isfinite(profile[key]) or profile[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("rise_us", "analog_rise_us"):
        if not np.isfinite(profile[key]) or profile[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    noise = profile["noise"]
    for key, value in noise.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid noise field {key}")
    for key in ("readout_false_positive", "readout_false_negative", "rydberg_decay_to_zero_fraction"):
        if not 0 <= noise[key] <= 1:
            raise ValueError(f"Invalid probability {key}")
    for key in ("rydberg_lifetime_us", "rydberg_pure_dephasing_us", "storage_pure_dephasing_us"):
        if noise[key] <= 0:
            raise ValueError(f"{key} must be positive; disable noise via the simulation flag")
    return profile


@dataclass(frozen=True)
class Segment:
    label: str
    stage: str
    channel: str
    operator: str
    targets: tuple
    duration_us: float
    amplitude_start: float
    amplitude_stop: float
    detuning_start: float = 0.0
    detuning_stop: float = 0.0

    def __post_init__(self):
        if self.duration_us <= 0 or not np.all(np.isfinite([
                self.duration_us, self.amplitude_start, self.amplitude_stop,
                self.detuning_start, self.detuning_stop])):
            raise ValueError("Segments require positive duration and finite controls")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("Targets must be nonempty and distinct")


def rotation_segments(angle, rate, rise_us, operator, targets, stage, channel, label):
    """Trapezoid/triangle of exactly the requested area; finite amplitude and slew."""
    area = abs(float(angle))
    if area < 1e-14:
        return []
    sign = np.sign(angle)
    if rise_us == 0:
        return [Segment(label, stage, channel, operator, tuple(targets), area / rate, sign*rate, sign*rate)]
    if area < rate * rise_us:
        ramp = np.sqrt(area * rise_us / rate)
        peak, hold = rate * ramp / rise_us, 0.0
    else:
        ramp, peak, hold = rise_us, rate, area / rate - rise_us
    pulses = [Segment(label, stage, channel, operator, tuple(targets), ramp, 0, sign*peak)]
    if hold > 1e-14:
        pulses.append(Segment(label, stage, channel, operator, tuple(targets), hold, sign*peak, sign*peak))
    pulses.append(Segment(label, stage, channel, operator, tuple(targets), ramp, sign*peak, 0))
    return pulses


class Compiler:
    """One locally addressed beam; Rydberg interactions remain on throughout."""
    def __init__(self, physics, profile):
        self.physics, self.profile = physics, profile
        self.n = physics.n_qubits

    def local(self, angle, axis, q, stage="ansatz", label=None):
        z = axis == "z"
        return rotation_segments(angle, self.profile["stark_rate_rad_us" if z else "raman_omega_rad_us"],
                    self.profile["rise_us"], "01"+axis, [q], stage,
                    "stark_local" if z else "raman_local", label or f"R{axis} q{q}")

    def excitation(self, angle, q, stage="digital_entangler", operator="1rx", label=None):
        transfer = stage == "transfer"
        return rotation_segments(angle, self.profile["transfer_omega_rad_us" if transfer else "gate_omega_rad_us"],
                   self.profile["rise_us"], operator, [q], stage, "rydberg_local",
                   label or f"Rydberg q{q}")

    def hadamard(self, q):
        # RY(pi/2) RZ(pi) = -i H; each H is physically implemented, not free.
        return (self.local(np.pi, "z", q, "digital_entangler", "H: Z") +
                self.local(np.pi/2, "y", q, "digital_entangler", "H: Y"))

    def cnot(self, control, target):
        # pi-2pi-pi produces diag(1,-1,-1,-1) in the blockade limit.
        # Physical Z corrections on BOTH qubits give CZ up to a global phase.
        return (self.hadamard(target) +
                self.excitation(np.pi, control, label="CZ: control pi") +
                self.excitation(2*np.pi, target, label="CZ: target 2pi") +
                self.excitation(np.pi, control, label="CZ: control pi") +
                self.local(np.pi, "z", control, "digital_entangler", "CZ phase correction") +
                self.local(np.pi, "z", target, "digital_entangler", "CZ phase correction") +
                self.hadamard(target))

    def analog(self):
        p = self.physics
        duration = p.evolution_us
        if duration == 0:
            return []
        rise = min(self.profile["analog_rise_us"], duration/2)
        omega, delta = p.omega_rad_us, p.omega_rad_us*p.delta_over_omega
        args = ("global Rydberg", "analog", "rydberg_global", "0rx", tuple(range(self.n)))
        if rise == 0:
            return [Segment(*args, duration, omega, omega, delta, delta)]
        output = [Segment(*args, rise, 0, omega, 0, delta)]
        if duration > 2*rise:
            output.append(Segment(*args, duration-2*rise, omega, omega, delta, delta))
        output.append(Segment(*args, rise, omega, 0, delta, 0))
        return output

    def compile(self, angles, parameters, architecture):
        if architecture not in ("digital", "digital_analog"):
            raise ValueError("Unknown architecture")
        if len(angles) != 2*self.n or len(parameters) != 3*self.n*self.physics.layers:
            raise ValueError("Wrong encoding or parameter dimension")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(parameters)):
            raise ValueError("Nonfinite angles")
        segments = []
        for q in range(self.n):
            segments += self.local(angles[q], "y", q, "encoding")
            segments += self.local(angles[self.n+q], "z", q, "encoding")
        index = 0
        for layer in range(self.physics.layers):
            for axis in ("x", "z", "x"):
                for q in range(self.n):
                    segments += self.local(parameters[index], axis, q)
                    index += 1
            if architecture == "digital":
                for control, target in cnot_pairs(self.n):
                    segments += self.cnot(control, target)
            else:
                # +Y(pi) maps |1> -> |r> with no -i transfer phase.
                for q in range(self.n):
                    segments += self.excitation(np.pi, q, "transfer", "1ry", "storage -> Rydberg")
                segments += self.analog()
                for q in reversed(range(self.n)):
                    segments += self.excitation(-np.pi, q, "transfer", "1ry", "Rydberg -> storage")
        return segments


def schedule_json(segments):
    elapsed = 0.0
    result = []
    for segment in segments:
        result.append({"start_us": elapsed, **asdict(segment)})
        elapsed += segment.duration_us
    return result
