# QRC/VQC full-training performance and resources

Accuracy is an observed outcome. Both methods use identical atoms, unique data splits, training-only PCA, pulse encoder, Z/ZZ temporal observables, shots per image, device/noise specification and linear regression procedure. QRC keeps pulse offsets fixed; VQC uses minibatch SPSA and periodically refits the same readout.

MNIST [3, 8]; 80 train / 20 validation / 40 test; seeds [7, 11, 19]; 4 atoms; 4 prefixes; 1,000 total shots/image. Validation uses 3 independent finite-shot feature measurements of the same validation images for both methods. The readout and alpha are selected by mean validation MSE. Test data is evaluated after checkpoint selection.

VQC stops after 6 checkpoint checks without at least 1% relative MSE improvement, or 64 SPSA steps. Checks occur every 4 steps and at the cap. A plateau is an operational stopping rule; a budget-limited run has not met it. Neither establishes global convergence. The absolute best validation loss selects the model even if its improvement is smaller than the plateau threshold.

## Held-out results and all consumed training

Training costs below include **all** training, validation repetitions, optimizer evaluations and discarded checkpoints. Selecting step zero never removes later work from these costs. Terminal results show the last checkpoint separately. Test shots are separate; identical selected/terminal checkpoints reuse one test evaluation.

| Seed | Method | Selected step | Selected test | Terminal step | Terminal test | Stop reason | All training shots |
|---|---|---:|---:|---:|---:|---|---:|
| 7 | QRC | 0 | 80.0% | 0 | 80.0% | fixed_reservoir_fit_complete | 140,000 |
| 7 | VQC | 20 | 82.5% | 44 | 80.0% | validation_loss_plateau | 2,384,000 |
| 11 | QRC | 0 | 80.0% | 0 | 80.0% | fixed_reservoir_fit_complete | 140,000 |
| 11 | VQC | 20 | 82.5% | 44 | 77.5% | validation_loss_plateau | 2,384,000 |
| 19 | QRC | 0 | 92.5% | 0 | 92.5% | fixed_reservoir_fit_complete | 140,000 |
| 19 | VQC | 16 | 75.0% | 40 | 85.0% | validation_loss_plateau | 2,180,000 |

| Method | Selected test mean ± sample SD | Terminal test mean ± sample SD |
|---|---:|---:|
| QRC | 84.2% ± 7.2% | 84.2% ± 7.2% |
| VQC | 80.0% ± 4.3% | 80.8% ± 3.8% |
| CLASSICAL | 90.0% ± 2.5% | 90.0% ± 2.5% |

## Device provenance and energy

Compiled target: **FRESNEL**, snapshot `d70cc7d46a56ab236ac21ef51415d60fb240cadef784d38480f14ca7047d0b52`. The public FRESNEL snapshot is provisional for the user's FRESNEL beta account. Local Pulser geometry, layout, global-channel, timing and shot-cap validation does not establish account-specific server acceptance. Layout pre-calibrated: **False**. Provider layout calibration and a matching account device export remain required. This default run submitted no QPU jobs.

The comparable interaction/drive/input regime follows [Kornjača et al., Supplementary II.2](https://arxiv.org/html/2407.02553v1#A2.SS2), adapted to this snapshot's constraints. This is a native analog VQC versus fixed analog reservoir comparison. Earlier digital/hybrid research architectures are separate and are not certified for these exposed global controls.

Electrical calibration status: **unavailable**. Calibrated joules remain unknown without target-specific measured whole-machine power, full cycle, attempts, setup and allocated external classical energy. The JSON reports those components separately. Each ledger row identifies the exact compiled program hash, actual shot request and any supplied meter/retry evidence. The full cycle includes control, loading/reset and readout; control duration is not added twice. CPU simulation runtime is not QPU runtime or electrical energy.

| Seed | Method | All training shots | Inference shots: selected / terminal | All-study jobs | Calibrated training J |
|---|---|---:|---:|---:|---:|
| 7 | QRC | 140,000 | 40,000 / 40,000 | 720 | unknown |
| 7 | VQC | 2,384,000 | 40,000 / 40,000 | 9,856 | unknown |
| 11 | QRC | 140,000 | 40,000 / 40,000 | 720 | unknown |
| 11 | VQC | 2,384,000 | 40,000 / 40,000 | 9,856 | unknown |
| 19 | QRC | 140,000 | 40,000 / 40,000 | 720 | unknown |
| 19 | VQC | 2,180,000 | 40,000 / 40,000 | 9,040 | unknown |

## Interpretation limits

This is a small exploratory simulator study; earlier work inspected overlapping MNIST test images. Validation repetitions reduce measurement noise, not finite-dataset uncertainty or validation selection bias. The default two Monte Carlo noise trajectories are a computational approximation; a trajectory-convergence study and fresh held-out data are needed for publication claims. Repeated seed SDs are descriptive, not independent-test confidence intervals. The symmetric four-atom chain has only six symmetry-distinct ideal Z/ZZ channels per prefix, despite 40 nominal features across four prefixes; sampling can inflate numerical rank. The reservoir resets per image, so no streaming memory capacity is established. No quantum or electrical-energy advantage is inferred from a shot ratio.

## Appendix: conditional literature sensitivity

JSON `literature_scenarios` retains Carrasco-Codina electrical inventory/timing transfers and historical Pasqal power envelopes, plus nominal provider beta throughput. These are other-device or historical sensitivity assumptions, not measured FRESNEL beta values. They apply to the actual compiled controls and consumed shots. They do not determine the main results, provider bills, or calibrated energy totals. See [the protocol](../../docs/performance-energy-protocol.md) for reaccounting and literature boundaries.
