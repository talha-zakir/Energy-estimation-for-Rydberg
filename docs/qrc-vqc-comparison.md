# Quantum reservoir computing versus the Pasqal-native VQC

The [paired results](../hardware_results/pasqal_qrc_mnist_3_8_seed7/SUMMARY.md) implement a fixed Rydberg reservoir on the same pinned Pasqal FRESNEL model as the native VQC. QoolQit compiles global pulses, and **Pulser** simulates them with the SDK's finite-bandwidth response and public noise model. All execution is on a CPU; no hardware jobs have been submitted.

## What the reservoir does

The quantum controls contain fixed biases copied from the VQC's saved random initialization. There is no search over reservoir seeds, geometry or pulse parameters. Eight PCA features drive global amplitude and detuning over four consecutive slots. The interacting Rydberg state evolves through those slots, retaining information about earlier inputs. Only the final classical linear regression weights and bias are trained. Regularization is chosen using validation MSE, as in the VQC pipeline.

Two QRC readouts are included:

| Readout | Quantum features | Physical measurement program |
|---|---:|---|
| Endpoint | 4 site Z values + 6 pair ZZ values = 10 | One full four-slot pulse sequence |
| Temporal | The same 10 features at each of four prefixes = 40 | Separate one-, two-, three- and four-slot sequences |

Each prefix starts from the ground state and undergoes its own terminal modulation tail and destructive measurement. The later prefixes execute all earlier input slots coherently before measurement. There is no reset between those slots, and there is no state carried between different images. Intermediate readouts are not free nondestructive observations of a single physical shot. This is a reset-per-image classifier with temporal input encoding, not a benchmark of continuous-stream recurrent prediction or a demonstrated echo-state property.

The endpoint QRC is mathematically the native VQC's existing **frozen-pulse baseline**. It has been promoted to an explicit reusable fixed-reservoir model, with a saved readout and cost accounting. Its coherent predictions and fitted weights are checked against the archive. The temporal model is the new readout architecture. Both can be loaded for independent inference.

## Shared protocol and differences

The comparison loads and validates the original VQC's exact **80 training, 20 validation and 40 held-out MNIST 3/8 test images**, fitted PCA, scaling/clipping, atomic geometry, input encoding, public device snapshot and readout regularization grid. It preserves the original train/test separation and checks deterministic split indices and labels. Preprocessing is reused without refitting on validation or test data.

The VQC has eight pulse offsets previously trained by twelve SPSA updates and a ten-feature linear head. That is a fixed, small optimization budget, with no convergence claim. QRC trains zero quantum parameters. Temporal QRC's forty-feature head is larger, so this compares complete learning systems with the same atoms, compressed inputs and total inference shots; it is not a feature-count-controlled test of whether variational training itself helps. The endpoint QRC supplies a control with the same ten-feature budget. All heads train on coherent expectations and remain frozen for noisy inference.

Noisy results for all models are reevaluated with **eight SDK noise trajectories and sampling seed 17007**. Device/detector settings are shared. The archived VQC used different noise-sampling settings; its old 62.5% finite-shot score is not used in this new paired comparison. Coherent predictions must reproduce the archived VQC, frozen baseline and classical classifier exactly. Trajectories are numerical realizations of device noise, not physical shots or additional hardware cost.

All Z and ZZ observables at a readout node come from the same measured bitstrings. The main finite-shot comparison allocates **1,000 total shots per image**: VQC and endpoint QRC use all 1,000 for one sequence, while temporal QRC uses 250 per prefix. A separate condition uses 1,000 shots per prefix, or 4,000 per image. The smaller sample is drawn without replacement from the larger recorded count sample, preserving correlations between sites. Consequently, the two QRC shot-budget scores are correlated and their small numerical differences should not be interpreted as a monotonic shot-scaling study.

The test subset was inspected in prior work. This is an exploratory pipeline check, with Wilson intervals and paired discordance/McNemar statistics in the JSON. It does not establish generalization over reservoir seeds or measured quantum hardware. The classical PCA-plus-linear classifier remains part of the comparison.

## Energy and electricity-cost accounting

The new comparison reuses the [literature-based estimator](literature-energy-cost.md), with the same versioned papers, power envelopes, shot overheads and illustrative electricity tariff for every model. Actual SDK samples determine the per-prefix control windows and integrated squared-Rabi exposure. The ledger records each image and prefix; no pulse duration is inferred from an abstract layer count.

For prefix `k` and its allocated shots `S_k`:

```text
E_image = sum_k S_k * P_machine * (t_reset + t_control,k + t_readout + configured_extra_time)
```

Each prefix pays its own reset/readout for every shot. At an equal total-shot budget, the transferred 20 ms overhead dominates the roughly 0.3–0.7 us prefix windows, so inference energy is approximately equal. Keeping 1,000 shots per prefix increases temporal-QRC inference energy to approximately four times the single-readout cost. The source-radiation scenario uses both optical legs and is recorded separately from the inclusive electrical power envelope; they are never added.

Training/validation feature extraction is cached for QRC. At the main shot allocation, its 100 images require 400 prefix-feature evaluations and **100,000 hypothetical physical shots**. The recorded VQC SPSA recipe requires 2,500 full-sequence feature evaluations and **2.5 million hypothetical shots** at 1,000 shots per evaluation. The report includes both a count breakdown and separate training versus inference budgets. It also includes endpoint-only and equal-shots-per-prefix projections.

These are finite-shot **budget projections** for future hardware training. The implemented training used exact coherent CPU expectations. The reported finite-shot inference accuracy uses those coherent-trained heads, not readouts retrained on the projected noisy measurements. Actual finite-shot training would change readout selection and the VQC optimizer trajectory. External classical preprocessing/optimization/readout energy, failures, reloading, idle allocation and provider service fees remain unmeasured. The estimates cannot establish matched-accuracy energy advantage.

## Papers and adaptation

- Araiza Bravo et al., [*Quantum Reservoir Computing Using Arrays of Rydberg Atoms*, PRX Quantum 3, 030325 (2022)](https://doi.org/10.1103/PRXQuantum.3.030325): physical precedent for fixed interacting Rydberg reservoirs and temporal information processing.
- Kornjača et al., [*Large-scale quantum reservoir learning with an analog quantum computer*, arXiv:2407.02553](https://arxiv.org/html/2407.02553v1): hardware demonstration of fixed analog Rydberg features, repeated measurements at different evolution times, and classical learning, including MNIST. Our global encoding, four-atom register and regression readout retain the existing VQC experiment's settings; this does not reproduce that paper's hardware architecture or SVM accuracies.
- [Pulser backend documentation](https://docs.pasqal.com/pulser/tutorials/backends/): local `QutipBackendV2`, execution configuration and returned results. Simulator observables at multiple times alone do not implement multiple nondestructive device measurements; the code instead builds separate pulse prefixes.

## Reproduce and reuse

Use the existing hardware environment and a fresh output directory:

```bash
.venv-hardware/bin/python -m rydberg_mnist.hardware.qrc \
  --config configs/pasqal_qrc_mnist.json \
  --output hardware_results/new_qrc_comparison
.venv-hardware/bin/python -m unittest discover -s tests -v
```

```python
from rydberg_mnist.hardware.qrc import predict_saved

# Original uint8 MNIST pixels, shape (N, 784); coherent local inference.
labels = predict_saved("hardware_results/pasqal_qrc_mnist_3_8_seed7/model.npz", images)
endpoint_labels = predict_saved(
    "hardware_results/pasqal_qrc_mnist_3_8_seed7/model.npz", images,
    variant="qrc_endpoint",
)
```

Saved artifacts include frozen reservoir biases and both heads, preprocessing, all feature matrices, split indices, predictions, raw noisy QRC counts, per-prefix control ledgers, compiled pulse examples, model/split/device hashes, paired statistics, energy projections, and PNG/SVG comparison figures. The eight-atom ideal digital/hybrid experiments and their previous reports remain available separately.
