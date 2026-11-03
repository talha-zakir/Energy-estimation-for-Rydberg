# Energy needed to reach an accuracy target

> Optional legacy accuracy-target study. The primary study is now the [full-training performance/resource protocol](performance-energy-protocol.md), which treats accuracy as an outcome and charges all consumed work regardless of checkpoint selection. Existing reports and commands remain supported.


The [finite-shot sample](../hardware_results/accuracy_energy_fresnel_beta_sample/SUMMARY.md) replaces a fixed-iteration cost ratio with a measurable question: **what energy and shot budget does each learning method consume before meeting a predeclared validation accuracy, and how does that model perform on held-out images?** A large energy difference is a hypothesis to test. It is not imposed on the experiment.

## Common learning system

Both QRC and VQC use the same atoms, compressed inputs, global-pulse encoding, temporal readout prefixes, joint Z/ZZ features, noisy shot counts, regression implementation and alpha grid. The four-atom default has forty features for both methods. The readout is multi-output linear least squares on one-hot targets, with optional ridge regularization; its argmax determines the digit class. Training-only feature standardization is identical for both methods. PCA is fitted only to training images.

QRC fixes eight initial pulse offsets and extracts its training/validation features once. VQC begins at the same offsets, then alternates finite-shot minibatch SPSA pulse updates with fresh sampled feature extraction and analytic linear-head refits. A fixed head is used inside each minibatch SPSA objective. Both training and validation use the supplied device noise, detector/SPAM model and finite shots. Neither method trains from exact state expectations in this protocol.

The default sample has 80 training, 20 validation and 40 official-test MNIST 3/8 images per repetition, with paired seeds 7, 11 and 19. It uses 1,000 total shots per image, divided across four destructive prefix measurements, and two noise trajectories for simulation. Trajectories are numerical noise realizations, not extra physical shots. This three-repeat sample is smaller than the complete uncertainty protocol in the reference literature. Test subsets can overlap across repetitions and with previous studies.

## Accuracy targets and stopping

The configuration declares 70%, 80% and 90% validation targets. For each method, the first eligible checkpoint meeting each target is selected. Choices are frozen before test scoring. The report gives both independent held-out accuracies and a predeclared tolerance flag; matching validation targets does not guarantee matching true or test accuracy. Test labels never select targets, parameters, checkpoints or regularization.

If a target is missed, its status is `not_reached_within_budget`, and its spent budget is retained. No interpolation fabricates a successful high-accuracy model. Eight SPSA steps in the sample are a demonstration budget, not a convergence claim. Increase the budget for a substantive trained-model study.

The fixed QRC reservoir is also the zero-update member of this VQC family. Checkpoint zero remains eligible: if it already attains the target, both methods can have the same minimum energy-to-target. Requiring unnecessary VQC updates would manufacture a cost disadvantage. Positive-update VQC checkpoints are retained in the learning curves, so their benefit and resource use can be inspected separately. Historical “25x” training projections from the earlier unequal-readout experiment should not be treated as evidence for equal-accuracy energy advantage.

## FRESNEL beta physical profile

The new [target snapshot](../configs/fresnel_beta_target.json) pins the publicly returned `FRESNEL` device. A provider table labels `QPU-FRESNEL-100PQ` as Orion Beta, while its prose mentions Gamma. The user has FRESNEL beta access, but this session has not authenticated to that account or confirmed its booked-device generation. A different account-specific specification must be pinned and revalidated before hardware execution.

The reservoir is constructed from the comparable-scale regime in [Kornjača et al., Supplementary II.2](https://arxiv.org/html/2407.02553v1): driving, interactions, encoding and probing should have comparable scales. Our design targets nearest-neighbor `V/Omega=1`, encoding `Delta_scale/Omega=1` and `Omega * slot_time` approximately 2. These are adaptations within the device's constraints, not an assertion of a universal optimum for our classifier.

For the pinned default, reference mixing/encoding is approximately 5.655 rad/us and spacing is approximately 7.314 um. Slots are 356 ns, with 72 ns ramps, aligned to the device's 4 ns clock. Bounded input/parameter maps keep amplitude and detuning below the exposed limits. Reported scale ratios refer to design reference values; actual amplitudes vary with inputs and the supplied bandwidth response modifies pulse shapes.

Registers derive from real `RegisterLayout` objects. The default automatic layout has sixty traps and respects Pulser's geometry, layout-size/filling and duration rules. Every sequence includes a measurement instruction and passes local `QPUBackend.validate_sequence(..., mimic_qpu=True)`. The default layout is custom and requires provider calibration. To reuse an existing calibrated layout, set `layout_slug` and `trap_ids`; the report then calculates the actual interaction ratios of that geometry rather than assuming the design spacing was realized. Provider/server acceptance and live calibration remain additional checks.

## Execution records and complete energy accounting

Every prefix request records algorithm, stage, checkpoint, source-image index, requested/returned shot count, random seed, program hash, compiled modulated duration and control exposure. Raw joint counts are saved in compressed JSON lines. Each prefix has its own preparation, execution and destructive readout. All Z/ZZ features at one node reuse the same bitstrings. Optimization, validation and failed target attempts remain in cumulative training costs. Classical alpha selection reuses cached features and triggers no extra quantum requests.

Three accounting boundaries are separate:

| Quantity | Required evidence | Interpretation |
|---|---|---|
| Measured program/workflow energy | Per-program whole-machine meter energy with source records, plus separately scoped external classical measurements | Direct measurement of that execution; program setup and attempts are already included |
| Calibrated model estimate | Matching device snapshot, dated/scoped measured power, full cycle, attempt factor, setup contribution and workload-specific external classical energy | Energy estimated using measured coefficients; it is not direct metering of the simulated run |
| Literature/provider scenario | Explicit published power and timing assumptions | Sensitivity case, kept separate from calibrated/measured totals |

[fresnel_beta_calibration.json](../configs/fresnel_beta_calibration.json) contains **null** measurements, because account access alone does not supply electrical metering. Missing setup, cycle, losses or external classical components never become zero. A measured full cycle includes control, loading/reset and readout, so compiled pulse time is not added to it again. If a measured cycle already includes amortized program setup, set `cycle_includes_program_setup=true`; then setup is not charged separately. Otherwise, a separately measured per-program setup contribution is required for completeness.

The simple calibrated model uses:

```text
E_quantum_shots = returned_shots * attempts_per_returned_shot * full_cycle_s * electrical_power_w
E_complete_estimate = E_quantum_shots + measured_setup_contribution + external_classical_energy_for_this_workload
```

If a provider adapter supplies direct whole-program electrical meter results for every request, those results are summed instead, without adding another modeled setup or cycle term. External classical energy must have the same workload scope and exclude components already inside the QPU/facility meter. A calibration run's host energy is not transferred to a different training run.

The [existing energy papers](literature-energy-cost.md) remain scenario sources. An additional [Scaleway provider table](https://www.scaleway.com/en/docs/quantum-computing/additional-content/pasqal-qpus/) lists beta throughput of **0.25 shots/s**, equivalent to nominal 4 s/shot. At the same assumed power, this is about 200 times the earlier transferred 20 ms overhead for these short sequences. It is a published throughput specification, not a measurement of the booked device. It is combined only with explicitly labeled conditional power envelopes. No provider bill or actual FRESNEL wall power is inferred.

## Running and reusing models

```bash
.venv-hardware/bin/python -m rydberg_mnist.hardware.accuracy_energy \
  --config configs/accuracy_energy_fresnel_beta.json \
  --output hardware_results/new_accuracy_energy_run
```

Use a fresh output directory. Each seed saves models, training-only PCA, head means/scales/weights, predictions, split indices, count records, execution ledgers and example hardware-format sequences. Top-level reports show learning curves, target reachability, actual held-out scores and separate calibrated/scenario budgets.

```python
from rydberg_mnist.hardware.accuracy_energy import predict_saved

# uint8 MNIST pixels of shape (N, 784); inference also uses finite-shot noise.
labels = predict_saved(
    "hardware_results/new_accuracy_energy_run/seed7/models_features.npz",
    images, model_key="qrc_step0", seed=70001,
)
```

Only validation-selected or target-selected checkpoints are saved for inference; their keys appear in the per-seed report.

## Calibrating without retraining

With meter data, first integrate a trace with columns `time_s,power_w`:

```bash
.venv-hardware/bin/python -m rydberg_mnist.hardware.calibrated_cost \
  --power-trace path/to/actual_power_trace.csv --output path/to/trace_integral.json
```

The integral alone does not prove device identity or workload allocation. Supply measured power/cycle/attempt/setup values with dated instrument/provider provenance in a new calibration JSON. External classical evidence has the structure in [the example manifest](../configs/classical_energy_measurements.example.json). Its workload keys include `seed7/qrc/train_to_step0`, `seed7/qrc/test_step0`, `seed7/vqc/all_training` and `seed7/vqc/full_study`. Each entry requires `energy_j`, `measured_at_utc`, `source_record` and `scope`.

```bash
.venv-hardware/bin/python -m rydberg_mnist.hardware.reaccount \
  --source hardware_results/new_accuracy_energy_run \
  --calibration path/to/actual_fresnel_calibration.json \
  --classical-measurements path/to/this_run_classical_measurements.json \
  --output estimates/new_calibrated_accounting
```

This reads the saved execution requests and preserves all accuracy results and checkpoint choices. It does not rerun quantum simulation or training. Incompatible device hashes and unsupported measurement provenance are rejected.

## Adapting execution to another QPU

The default compiler is `AnalogProfile`, and the default execution adapter is `PulserCountsBackend`. `run(..., profile_factory=..., backend_factory=...)` allows both to be replaced. The compiler supplies the device constraints, sequence construction, accounting and export. The backend returns joint bitstrings with a checked sample count. A new Pulser-compatible global analog target primarily needs a new pinned target and calibration profile. Other quantum hardware requires its own physically appropriate compiler and execution adapter; changing a device name alone is insufficient.

`CallableCountsBackend` accepts an application's configured provider execution function. Creating it submits nothing. The caller controls the connection, authentication, submission, retries and result retrieval. The callback can return plain counts or `ExecutionCounts` with attempted shots, active execution time, whole-program meter energy and a traceable source record. Do not substitute queue wait or billed duration for measured active cycle time.

This session only uses the local simulator and exports examples. Before a later QPU run, confirm the booked device specification, reuse/calibrate its layout and perform provider-side emulator validation as described in [Pasqal's Fresnel execution guide](https://docs.pasqal.com/cloud/fresnel-job/).

## Literature basis

- [Kornjača et al., *Large-scale quantum reservoir learning with an analog quantum computer*](https://arxiv.org/html/2407.02553v1), Supplementary I.2 and II.2: measured embeddings, sampling uncertainty and comparable physical scales. Our small three-repeat experiment is an adaptation with a common regression head, not a reproduction of its SVM benchmark.
- [Bowles, Ahmed and Schuld, *Better than classical?*](https://arxiv.org/html/2403.07059v2), Section IV: optimization and hyperparameter resources materially affect benchmarking conclusions.
- [Alves et al., *Energetics of Rydberg-atom Quantum Computation*](https://arxiv.org/html/2601.03141v2): radiation-source accounting and its omitted classical components.
- [Carrasco-Codina et al., *Energy efficiency of quantum computers*](https://arxiv.org/html/2605.15090v2), Section 6.2 and Appendix A.4: full electrical inventory and complete shot-cycle/reloading overheads.
