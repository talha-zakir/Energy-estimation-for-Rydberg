# Pasqal target and native analog classifier

Target: **FRESNEL**, pinned public specification, using QoolQit/Pulser. No authenticated hardware access or live calibration was established. All execution was local.

## Compatibility

The retrieved interface exposes global ground–Rydberg control. The original digital and digital–analog circuits are preserved as research baselines and are rejected by the Pasqal backend because they require unexposed local/storage controls. The new global-pulse classifier is a separate architecture, not a relabeling of either earlier circuit.

[Target configuration](../configs/pasqal_target.json) · [Compatibility audit](pasqal_fresnel_8q/compatibility.json)

## Eight-atom analog-block check

Compiled instruction duration: 556 ns. This isolated check starts all atoms in the ground state; it is not an MNIST classifier.

| Condition | Mean Rydberg occupation | Returned bitstring samples |
|---|---:|---:|
| ideal | 0.08099 | 1000 |
| bandwidth | 0.07404 | 1000 |
| device_noise_and_bandwidth | 0.10990 | 1000 |

The noisy condition uses the supplied public-device noise model. Occupations are quantum-state expectations; detector errors affect the sampled bitstrings. The noise model's recency/applicability to a booked device is not independently verified.

[Block simulation report](pasqal_fresnel_8q/report.json)

## Pasqal-native MNIST 3/8 classifier

4 atoms; 4 global pulse slots; 8 trained pulse offsets; linear regression readout. 80 training, 20 validation, 40 held-out test images; seed 7.

Every training evaluation is compiled with QoolQit and emulated with Pulser's finite-bandwidth response. Geometry and instruction duration are fixed across inputs/parameters. SPSA trains pulse offsets; the readout is fitted analytically. Training uses the coherent bandwidth model. Pulse/readout parameters are frozen before noisy evaluation.

| Model/condition | Held-out accuracy | Correct / test images |
|---|---:|---:|
| classical_pca_linear | 87.5% | 35 / 40 |
| frozen_native_pulses_linear | 57.5% | 23 / 40 |
| trained_native_pulses_linear | 52.5% | 21 / 40 |
| trained_native_pulses_device_noise | 52.5% | 21 / 40 |
| trained_native_pulses_device_noise_shots | 62.5% | 25 / 40 |

Fixed SPSA budget: 12 iterations; selected checkpoint: 12 by validation MSE. No convergence claim. Compiled/emulated sequences: 2620. Runtime: 224.1 s.

This is a small single-seed pipeline check. The datasets, architecture, feature budget and training budget differ from the original eight-qubit baselines. These scores do not establish a fair cross-architecture advantage or quantum advantage.

The device-noise expectation condition includes state dynamics; the shot condition also includes SDK detector/SPAM errors. Their distinction matters when using a readout trained without those errors.

[Classifier report](pasqal_native_mnist_3_8_seed7/report.json) · [Training history](pasqal_native_mnist_3_8_seed7/history.json) · [Compiled input example](pasqal_native_mnist_3_8_seed7/compiled_example.json) · [Figure](pasqal_native_comparison.png)

## Energy status

For the first test input, scheduled modulated control window: 0.682 us. Its laser-control exposure is recorded from the actual SDK waveform. Full shot cycle time and optical/electrical/end-to-end joules remain unknown. No laser power or wall efficiency is inferred from Rabi frequency alone.

The next energy study must establish common task/data/accuracy criteria, compatible hardware targets for the original baselines, measured power/overhead parameters, and explicit training versus inference workload counts.

[Design, assumptions and reproduction commands](../docs/pasqal-target.md)
