"""Swappable count-execution adapters and an auditable finite-shot feature engine."""
import gzip
import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .qrc import allocate_shots, emulate_counts
from .pasqal_native import sampled_features


@dataclass
class ExecutionCounts:
    """Optional hardware measurements supplied by a provider/meter adapter."""
    counts: dict
    attempted_shots: int | None = None
    active_duration_s: float | None = None
    whole_program_electrical_j: float | None = None
    source_record: str | None = None
    submitted_jobs: int = 1


class CountsBackend(Protocol):
    """A provider adapter returns actual joint bitstring counts, never moments."""
    execution_kind: str

    def sample_counts(self, sequence, shots: int, seed: int) -> dict:
        ...


class PulserCountsBackend:
    execution_kind = "local_noisy_finite_shot_simulation"

    def __init__(self, device, trajectories):
        if trajectories < 1 or device.noise_model is None:
            raise ValueError("A supplied noise model and positive trajectory count are required")
        self.device, self.trajectories = device, trajectories

    def sample_counts(self, sequence, shots, seed):
        _, counts = emulate_counts(sequence, self.device, noisy=True, shots=shots,
                                   seed=seed, trajectories=self.trajectories)
        return counts


class CallableCountsBackend:
    """Inject a provider's blocking execution function after configuring its connection.

    The application owns authentication, submission, retries and job-result retrieval.
    Constructing this adapter never connects to or submits a job to a QPU.
    """
    execution_kind = "external_provider_callback"

    def __init__(self, execute_counts, device_sha256):
        self.execute_counts = execute_counts
        self.device_sha256 = device_sha256

    def sample_counts(self, sequence, shots, seed):
        return self.execute_counts(sequence=sequence, shots=shots, seed=seed)


class FeatureEngine:
    def __init__(self, profile, backend, readout_slots, total_shots):
        self.profile, self.backend = profile, backend
        slots = profile.config["input_slots"]
        if not readout_slots or readout_slots != sorted(set(readout_slots)) or readout_slots[-1] != slots or readout_slots[0] < 1:
            raise ValueError("Readout prefixes must be sorted, unique, and include the last slot")
        if getattr(backend, "device_sha256", profile.target["device_sha256"]) != profile.target["device_sha256"]:
            raise ValueError("Execution adapter and selected device snapshot do not match")
        self.readout_slots = readout_slots
        self.allocation = allocate_shots(total_shots, len(readout_slots))
        self.records, self.counts = [], []
        limits = [value for value in (getattr(profile, "max_shots_per_job", None),
                  getattr(getattr(profile, "device", None), "max_runs", None)) if value is not None]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in limits):
            raise ValueError("Device per-job shot limit must be a positive integer")
        self.max_runs = min(limits) if limits else None

    def job_shots(self, shots):
        """Split a logical prefix readout into valid provider jobs."""
        limit = self.max_runs or shots
        return [min(limit, shots-start) for start in range(0, shots, limit)]

    def features(self, data, parameters, indices, *, algorithm, stage, checkpoint, seed):
        if len(data) != len(indices):
            raise ValueError("One source index is required per input")
        result = []
        n = self.profile.config["n_qubits"]
        for i, (features, index) in enumerate(zip(data, indices)):
            nodes = []
            for node, (prefix, total) in enumerate(zip(self.readout_slots, self.allocation)):
                sequence = self.profile.sequence(features, parameters, prefix)
                accounting = self.profile.account_sequence(sequence)
                aggregate = {}
                chunks = self.job_shots(total)
                logical_id = len(self.records)
                for job, shots in enumerate(chunks):
                    self._execute(sequence, shots, seed+1009*i+node+job*1000003,
                        algorithm, stage, checkpoint, index, prefix, logical_id, job, len(chunks), accounting, aggregate)
                nodes.append(sampled_features(aggregate, n))
            result.append(np.concatenate(nodes))
        return np.asarray(result)

    def _execute(self, sequence, shots, seed, algorithm, stage, checkpoint, index,
                 prefix, logical_id, job, job_count, accounting, aggregate):
        n = self.profile.config["n_qubits"]
        execution_seed = int(seed % (2**32-1))
        returned = self.backend.sample_counts(sequence, shots, execution_seed)
        evidence = returned if isinstance(returned, ExecutionCounts) else ExecutionCounts(returned)
        counts = evidence.counts
        if (not counts or any(not isinstance(k, str) or len(k) != n or set(k)-{"0", "1"} for k in counts)
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in counts.values())
                or sum(counts.values()) != shots):
            raise ValueError("Backend returned invalid joint counts or the wrong requested sample count")
        if evidence.attempted_shots is not None and (isinstance(evidence.attempted_shots, bool)
                or not isinstance(evidence.attempted_shots, int) or evidence.attempted_shots < shots):
            raise ValueError("Measured attempt count must include every returned shot")
        if isinstance(evidence.submitted_jobs, bool) or not isinstance(evidence.submitted_jobs, int) or evidence.submitted_jobs < 1:
            raise ValueError("Submitted job count must include the successful job and retries")
        for value in (evidence.active_duration_s, evidence.whole_program_electrical_j):
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
                raise ValueError("Invalid measured execution time/energy")
        if any(v is not None for v in (evidence.attempted_shots, evidence.active_duration_s,
                                        evidence.whole_program_electrical_j)) and not evidence.source_record:
            raise ValueError("Hardware measurements require a provider/meter source record")
        if evidence.submitted_jobs > 1 and not evidence.source_record:
            raise ValueError("Provider retries require a source record")
        for bitstring, value in counts.items():
            aggregate[bitstring] = aggregate.get(bitstring, 0)+int(value)
        payload = accounting["serialized_program"]
        record = {"request_id": len(self.records), "algorithm": algorithm, "stage": stage,
            "checkpoint": int(checkpoint), "source_index": int(index), "prefix": int(prefix),
            "shots": int(shots), "noise_seed": execution_seed,
            "logical_request_id": logical_id, "job_index": job, "job_count": job_count,
            "submitted_jobs": evidence.submitted_jobs,
            "control_us": accounting["control_us"],
            "omega_squared_integral_rad2_per_us": accounting.get("omega_squared_integral_rad2_per_us"),
            "program_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "execution_kind": self.backend.execution_kind,
            "measured_attempted_shots": evidence.attempted_shots,
            "measured_active_duration_s": evidence.active_duration_s,
            "measured_whole_program_electrical_j": evidence.whole_program_electrical_j,
            "measurement_source_record": evidence.source_record}
        self.records.append(record)
        self.counts.append({"request_id": record["request_id"], "counts": {k: int(v) for k, v in counts.items()}})

    def save_counts(self, path):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in self.counts:
                handle.write(json.dumps(row, separators=(",", ":"))+"\n")
