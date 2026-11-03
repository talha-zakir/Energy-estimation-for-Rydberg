# Full-training QRC/VQC comparison on a provisional FRESNEL beta profile

This is the primary performance/resource protocol. Accuracy is an outcome, not a shared stopping threshold. Both methods receive the same unique images and atoms; all additional VQC passes, evolution, measurements, validation checks and optimizer evaluations are charged. Selecting an earlier checkpoint does not erase later consumed work. The older [accuracy-target protocol](accuracy-energy-protocol.md) remains available as an optional, separate question.

## Run and outputs

```bash
python -m rydberg_mnist.hardware.performance_energy \
  --config configs/performance_energy_fresnel_beta.json \
  --output hardware_results/new_performance_energy
```

The default configuration uses MNIST 3/8, 80 train / 20 validation / 40 official-test images per seed, seeds 7/11/19, four atoms, four encoding slots, temporal-prefix Z and ZZ features, and 1,000 total shots per image (250 per prefix). PCA fits training images only. QRC and VQC share initial random pulse offsets and step-zero measurement seeds, which makes QRC exactly the zero-update control of this VQC family. Paired initial results therefore need not be independent. Later checkpoints have new measurements.

Each checkpoint extracts one finite-shot training feature matrix and **three independent finite-shot validation feature matrices** for the same validation images. Both methods use training-only feature standardization and the same multi-output linear regression solver and ridge grid `[0.01, 0.1, 1, 10]`. The grid includes OLS if `0` is explicitly configured. Each alpha is fit on the single training matrix and scored by mean MSE across all validation repetitions. Reusing stored features for the grid needs no additional quantum shots. Test outcomes never select alpha, checkpoint or stopping.

QRC performs that extraction and fit once. VQC uses projected minibatch SPSA (batch size 8), holding the linear head fixed between checkpoint refits every four optimizer steps. Both positive and negative objective measurements are charged. It stops after six checkpoint checks without at least a 1% relative validation-MSE improvement, or at 64 steps. The patience reference is the last sufficiently improved loss; smaller improvements may accumulate to cross its threshold. The selected model is the absolute lowest mean validation MSE, with earliest checkpoint winning ties, even when its improvement is below 1%. If plateau and cap occur together, the plateau reason takes precedence. Neither stopping reason proves global convergence.

The sample uses two Monte Carlo trajectories per local noisy emulation. These approximate the device noise distribution and are **not** physical shot multipliers, hardware repetitions or a verified noise-convergence setting. Joint bitstrings include the public device noise/SPAM model; training does not use exact expectation values.

Every completed repeat saves:

- `models_features.npz`: **every checkpoint's** quantum parameters, regression weights/bias, training-only feature scaler, training/validation feature matrices, plus shared PCA and configuration. Selected and terminal checkpoints also retain test features.
- `splits_predictions.npz`: exact indices, labels and selected/terminal predictions. Training and validation indices refer to the official training set; test indices refer to the separate official test set.
- `execution_ledger.csv` and `counts.jsonl.gz`: every actual count request, stage, source image, checkpoint, seed, prefix, program hash, compiled modulated control duration and any provider/meter evidence.
- `report.json`: all checkpoints, SPSA objective history, selected and terminal tests, stop reason and full consumed budgets; example Pulser sequences with layouts and measurement instructions.

The top-level report, summary and figure retain all seeds. `device_snapshot.json` contains the exact hashed specification; `portable_target.json` points to it for relocation. Original target/calibration/literature inputs are also snapshotted. Output directories must be new. A failed run does not generate a completed report; `progress.json` preserves earlier completed repeats.

Saved models use the existing finite-shot inference API:

```python
from rydberg_mnist.hardware.accuracy_energy import predict_saved
labels = predict_saved("hardware_results/new_performance_energy/seed7/models_features.npz",
                       images, model_key="vqc_step4")
```

`images` are flattened uint8 MNIST images. The requested model key must exist. Inference verifies the device hash; the configured source target must still be accessible. For an archived/relocated study, supply a `profile_factory` loading its `portable_target.json` with the recorded regime. The same compiler and execution interfaces permit other device snapshots/backends; the default implementation is Pulser analog, not a universal digital gate compiler.

## What the resource comparison charges

For each algorithm, `training_budget` includes **all** training features, all repeated validation, all SPSA evaluations and all discarded checkpoints. `selected_inference_budget` and `terminal_inference_budget` measure a held-out pass each. When the same checkpoint serves both roles, its actual measured features and ledger rows are reused, so `complete_study_execution_budget` charges it once. When checkpoints differ, that full-study budget includes both test passes. `training_plus_selected_inference_budget` includes the complete consumed training plus only the selected model's test pass, not just training through the selected step. Do not sum overlapping budget views.

Checkpoint curves show resources accumulated up to that checkpoint for learning dynamics; the primary final training comparison uses the full consumed budget. Equal unique datasets do not imply equal numbers of uses: each repetition/pass/objective is explicitly in the ledger. Optimizer history records minibatch source indices and positive/negative losses. The regression candidate count is recorded at every checkpoint. External classical processing has an energy component of its own and remains unknown until scoped measurements are supplied.

The feature engine splits a logical prefix measurement into jobs no larger than `device.max_runs` (500 in this pinned FRESNEL snapshot). Counts are pooled **before** computing Z/ZZ features, preserving shot weighting. Each job gets an independent seed and ledger row. `logical_request_id`, `job_index` and `job_count` identify the split; each row has one actual program hash. Example exports obey the same shot cap. The default 250-shot prefix requires no splitting.

`CallableCountsBackend` remains an explicit integration point for a blocking authenticated provider execution function; construction submits nothing. It may return `ExecutionCounts` with `attempted_shots`, `active_duration_s`, `whole_program_electrical_j`, `submitted_jobs` and a `source_record`. `submitted_jobs` includes the successful submission and any provider retries represented by that result. Attempts and metered program energy must cover those retries as well. The application must retain failed-job evidence and expose those attempts; hidden retries cannot be inferred from successful bitstrings. The engine does not retry or contact a QPU automatically. The local simulator records count requests, not physical cloud submissions.

## Hardware validity and calibration

The default target is a **public, unauthenticated FRESNEL specification**, provisionally associated with FRESNEL beta. The user's account-specific export is not available in this repository. This is not a verified account hardware specification, live calibration or server acceptance record. Replace the target with the exact account device export and update its SHA-256 and calibration identity before a hardware submission; rerun compilation/simulation if the specification changes. Do not re-label results simulated on a different snapshot.

`AnalogProfile` derives a comparable-scale drive/interaction/input regime, validates the actual register and layout, global channel, pulse ramps, clock and duration locally, and embeds the layout in exports. [Pasqal's FRESNEL job documentation](https://docs.pasqal.com/cloud/fresnel-job/) recommends pre-calibrated layouts. The default custom chain needs provider calibration. A selected pre-calibrated layout uses explicit trap IDs, and the report computes actual per-atom nearest-neighbor interactions from those coordinates. Available pre-calibrated layouts may change the desired regime; the actual values are reported rather than silently assuming the custom chain's spacing. Server-side emulation/validation and account device checks remain required before running the QPU.

The physical regime is adapted from [Kornjača et al., Supplementary II.2](https://arxiv.org/html/2407.02553v1#A2.SS2), which motivates comparable input/drive/interaction/probing scales. The native analog VQC is a trainable member of that same encoding family, inspired by the wider Rydberg variational literature including [Lu et al.](https://arxiv.org/abs/2401.02940). It is not a hardware reproduction of Lu's digital–analog architecture, nor an assertion that these settings are optimal on FRESNEL. The original digital-only and digital–analog research baselines are retained separately; exposed FRESNEL global controls do not certify those architectures as written.

Electrical accounting uses each actual compiled program and requested count, not another paper's circuit workload. In a measured calibration, modeled execution energy is `power × full-cycle-time × attempts`. Per-request measured attempts override the calibration's average attempts factor for that request. Program setup is charged per submitted job including reported retries, unless already included in the measured cycle. A full cycle includes control, loading/reset and readout; compiled control time is not added again. A direct whole-program meter total includes its own setup/attempts and is reported separately without adding modeled overheads on top. Complete workflow totals require allocated external classical energy. Simulator runtime/CPU energy cannot substitute for QPU runtime/energy.

**Calibrated FRESNEL joules are currently unknown.** Other-device power/cycle numbers are not valid calibration evidence for this account. The default calibration profile leaves them null. Literature sensitivity remains an appendix: [Carrasco-Codina et al.](https://arxiv.org/abs/2605.15090) informs whole-machine inventory and timing scenarios; [Alves et al.](https://arxiv.org/abs/2601.03141) concerns a distinct radiation-energy boundary, not whole-machine electrical power. Historical Pasqal power and nominal beta throughput are conditional envelopes. The provider's beta/gamma naming ambiguity is retained in the source configuration. No scenario or shot ratio establishes a FRESNEL electrical-energy advantage or provider bill.

## Reaccount without retraining

```bash
python -m rydberg_mnist.hardware.reaccount \
  --source hardware_results/new_performance_energy \
  --calibration configs/my_measured_fresnel_calibration.json \
  --classical-measurements configs/my_allocated_classical_measurements.json \
  --output hardware_results/new_performance_energy_measured
```

This reads the original execution ledgers and verifies calibration identity/provenance. It does not simulate, retrain, reselect or change predictions. It supports both the new full-training reports and legacy target reports. Classical measurements are optional; missing values keep complete totals unknown. Workload keys for the new format are:

| Key | Attributed external classical work |
|---|---|
| `seed7/vqc/all_training` | All preprocessing allocation, optimization, checkpoint/head fitting and validation |
| `seed7/vqc/test_step4` | One held-out inference pass at checkpoint 4 |
| `seed7/vqc/all_training_and_test_step4` | Full training plus that inference; if absent, sum the two component measurements only when both exist |
| `seed7/vqc/full_study` | All work including every evaluated test checkpoint |
| `seed7/vqc/train_to_step4` | Optional checkpoint-curve allocation through checkpoint 4 |

Use analogous keys for each seed and QRC. Entries follow `configs/classical_energy_measurements.example.json`: device hash, nonnegative energy, measurement date, source record and explicit scope. These are overlapping alternative views, not additive workloads. The model's fixed setup/calibration coefficient must not include the same external classical energy a second time.

## Publication limits

The four-atom symmetric chain has six symmetry-distinct ideal Z/ZZ channels per prefix (24 over four prefixes), versus 40 nominal sampled features. Finite-shot noise can inflate numerical rank. Two noise trajectories need convergence checks; public noise/SPAM coefficients are not account calibration. Validation measurement repetitions are not independent dataset repetitions, and the seeds use overlapping official-test subsets previously inspected during exploratory work. Report descriptive accuracy/shot tradeoffs, stopping reasons and these limits; a fresh held-out study, measured calibration and hardware validation are needed before publication claims. The reset-per-image reservoir does not establish streaming memory capacity. An observed optimization plateau does not imply a fully converged or globally optimal VQC.
