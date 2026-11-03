"""Constraint-derived analog reservoir programs on an interchangeable Pulser device."""
from dataclasses import dataclass
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np
from pulser import Pulse, QPUBackend, Register, Sequence
from pulser.waveforms import CompositeWaveform, ConstantWaveform, RampWaveform
from pulser.sampler import sample

from .pasqal import load_target


@dataclass
class AnalogProfile:
    target: dict
    device: object
    config: dict
    register: object
    channel_id: str
    omega: float
    delta: float
    slot_ns: int
    rise_ns: int
    metadata: dict

    @classmethod
    def build(cls, target_path, config):
        target, device = load_target(target_path)
        channels = [(key, channel) for key, channel in device.channels.items()
                    if channel.basis == "ground-rydberg" and channel.addressing == "Global"]
        if not channels:
            raise ValueError("This compiler requires a global ground–Rydberg channel")
        channel_id, channel = channels[0]
        n = config["n_qubits"]
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= device.max_atom_num:
            raise ValueError("Invalid atom count")
        for key in ("mixing_fraction_of_max", "interaction_to_mixing", "encoding_to_mixing", "phase_per_slot"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        omega = config["mixing_fraction_of_max"]*channel.max_amp
        delta = config["encoding_to_mixing"]*omega
        if 1.4*omega > channel.max_amp or delta > channel.max_abs_detuning:
            raise ValueError("Requested mixing/encoding scales exceed the channel limits")
        spacing = (device.interaction_coeff/(config["interaction_to_mixing"]*omega))**(1/6)
        if spacing < device.min_atom_distance:
            raise ValueError("Published-regime spacing is below this device's minimum")
        clock = channel.clock_period
        slot = int(ceil(config["phase_per_slot"]/omega*1000/clock)*clock)
        rise = max(clock, int(ceil(.2*slot/clock)*clock))
        if slot < channel.min_duration or slot > channel.max_duration or 2*rise >= slot:
            raise ValueError("Pulse slot violates device duration limits")
        if slot*config["input_slots"] > device.max_sequence_duration:
            raise ValueError("Input encoding exceeds maximum sequence duration")
        # A real QPU needs a register derived from a layout. Pulser's automatic
        # layout respects min/max trap count and filling, even for small registers.
        coordinates = {f"q{i}": (float((i-(n-1)/2)*spacing), 0.) for i in range(n)}
        if config.get("layout_slug"):
            candidates = [layout for layout in device.pre_calibrated_layouts if layout.slug == config["layout_slug"]]
            if len(candidates) != 1:
                raise ValueError("Requested calibrated layout is unavailable")
            traps = config.get("trap_ids", [])
            if len(traps) != n:
                raise ValueError("Provide one calibrated-layout trap ID per atom")
            register = candidates[0].define_register(*traps, qubit_ids=list(coordinates))
        else:
            register = Register(coordinates).with_automatic_layout(device)
        device.validate_register(register)
        actual = np.array(list(register.qubits.values()))
        distances = np.linalg.norm(np.diff(actual, axis=0), axis=1)
        interactions = device.interaction_coeff/distances**6
        calibrated = device.register_is_from_calibrated_layout(register)
        metadata = {"target_device": device.name, "device_sha256": target["device_sha256"],
            "compiler": "Pulser global analog pulses, explicit layout, SDK QPU validation",
            "atomic_coordinates_um": actual.tolist(), "nearest_neighbor_spacing_um": distances.tolist(),
            "mixing_rad_us": omega, "encoding_rad_us": delta,
            "interaction_to_mixing_actual": (interactions/omega).tolist(),
            "encoding_to_mixing": delta/omega, "slot_duration_ns": slot, "rise_ns": rise,
            "mixing_phase_per_slot_actual": slot*.001*omega,
            "layout_traps": register.layout.number_of_traps, "layout_precalibrated": calibrated,
            "layout_calibration_required": not calibrated,
            "hardware_submission_verified": False,
            "published_regime_source": "https://arxiv.org/html/2407.02553v1#A2.SS2",
            "scope": "Comparable driving/interaction/encoding/probing scales adapted within target constraints; "
                     "not a reproduction or a provider-confirmed operating optimum"}
        return cls(target, device, dict(config), register, channel_id, omega, delta, slot, rise, metadata)

    def sequence(self, features, parameters, prefix):
        slots = self.config["input_slots"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or not 1 <= prefix <= slots:
            raise ValueError("Invalid readout prefix")
        features, parameters = np.asarray(features), np.asarray(parameters)
        if features.shape != (2*slots,) or parameters.shape != (2*slots,) or not np.all(np.isfinite([features, parameters])):
            raise ValueError("Expected finite amplitude/detuning inputs and offsets")
        amplitudes = self.omega*(1+.4*np.tanh(features[:slots]+parameters[:slots]))
        detunings = self.delta*np.tanh(features[slots:]+parameters[slots:])
        sequence = Sequence(self.register, self.device)
        sequence.declare_channel("reservoir", self.channel_id)
        def waveform(value):
            return CompositeWaveform(RampWaveform(self.rise_ns, 0., float(value)),
                ConstantWaveform(self.slot_ns-2*self.rise_ns, float(value)),
                RampWaveform(self.rise_ns, float(value), 0.))
        for amplitude, detuning in zip(amplitudes[:prefix], detunings[:prefix]):
            sequence.add(Pulse(waveform(amplitude), waveform(detuning), 0.), "reservoir")
        sequence.measure("ground-rydberg")
        self.device.validate_register(sequence.register)
        QPUBackend.validate_sequence(sequence, mimic_qpu=True)
        return sequence

    def export_example(self, sequence, path, shots):
        """Offline export only; account/layout/server validation is still required."""
        path = Path(path)
        payload = sequence.to_abstract_repr()
        path.write_text(payload+"\n")
        return {"sequence_file": path.name, "sequence_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "device_sha256": self.target["device_sha256"], "runs": shots,
                "local_qpu_validation_passed": True, "server_validation_passed": False,
                "layout_calibration_required": self.metadata["layout_calibration_required"]}

    def account_sequence(self, sequence):
        """Compiler-owned accounting keeps the learning loop independent of SDK objects."""
        waveform = sample(sequence, modulation=True)
        values = next(iter(waveform.channel_samples.values()))
        return {"control_us": len(values.amp)*.001,
                "omega_squared_integral_rad2_per_us": float(np.sum(np.asarray(values.amp)**2)*.001),
                "serialized_program": sequence.to_abstract_repr()}
