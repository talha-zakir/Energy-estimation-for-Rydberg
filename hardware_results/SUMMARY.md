# Hardware-informed simulation results

These are research-model simulations with illustrative control/noise parameters, not calibrated QPU predictions. No hardware jobs were submitted. The original ideal eight-qubit accuracy benchmarks are retained separately.

## SDK cross-check: isolated eight-atom analog block

QoolQit compiles to constrained `AnalogDevice` and Pulser/QutipBackendV2 simulates the sequence. Compiled duration: 500 ns. Compiled rectangular-Hamiltonian state fidelity to the original ideal block: 1.000000000000. Sampled-backend fidelity: 0.999956474.

The backend's 1 ns sampled waveform has a zero end sample. Its small discrepancy from a mathematical rectangle is reported rather than hidden. The SDK's compilation rescales C6/geometry, amplitude and time. This block validation does not compile the full digital/hybrid storage controls, and its compiled settings are not used in the separate research-profile energy ledger.

[SDK report](qoolqit_analog_block_8q/sdk_report.json) · [Compiled Pulser sequence](qoolqit_analog_block_8q/compiled_sequence.json)

## Full pulse/noise smoke test: four atoms

Four balanced MNIST 3/8 inputs, two paired static-noise realizations, and 1,000 simulated detector shots per input. Ideal-trained parameters/readouts are frozen. Fidelity below is unconditional state fidelity to each architecture's own ideal output; it is not digit classification accuracy and is not a direct similarity between architectures.

| Architecture | Mean coherent pulse fidelity | Mean noisy fidelity | Mean residual Rydberg probability (noisy) |
|---|---:|---:|---:|
| digital_analog | 0.2602 | 0.2628 | 0.0222 |
| digital | 0.9887 | 0.1597 | 0.2577 |

**This first pulse compiler is not yet a validated implementation of the ideal classifiers on hardware.** Finite transfer and always-on interactions substantially alter the hybrid outputs under this profile. The slow, serialized blockade gates are also sensitive to the assumed noise. Pulse calibration, phase compensation or pulse-aware retraining must precede an accuracy-matched energy comparison. These four examples and two noise draws cannot establish hardware accuracy or an energy winner.

[Per-input outputs and diagnostics](pulse_probe_4q/pulse_report.json) · [Figure](pulse_probe.png)

## Full eight-qubit schedules: accounting only

One paired official MNIST test input (index 7024). Uses the existing trained eight-qubit models. This section compiles schedules; it does not run full noisy eight-atom dynamics.

| Architecture | Linear waveform segments | Scheduled control time per shot | Optical joules | Full pipeline joules |
|---|---:|---:|---|---|
| digital_analog | 285 | 18.431 us | Unknown | Unknown |
| digital | 563 | 333.579 us | Unknown | Unknown |

These times include encoding, storage rotations, transfer, CNOT basis/phase corrections and analog evolution. They exclude uncalibrated loading, cooling, measurement/reset and classical processing. The one-local-beam scheduler serializes CNOTs; optimized native gates or parallel addressing can change this comparison substantially. The values are not vendor performance specifications.

The hybrid's short but strong transfer pulses have a much larger local-Rydberg squared-amplitude integral than the slower digital pulses. Therefore even the shorter control window does not by itself establish lower laser energy. Channel power calibrations and wall efficiencies are still absent; all joule totals remain null. Full pipeline energy additionally needs classical training/inference measurements.

[Eight-qubit accounting](accounting_8q/accounting_report.json) · [Hybrid schedule](accounting_8q/digital_analog_schedule.json) · [Digital schedule](accounting_8q/digital_schedule.json)

## Validation and next step

Tests cover CNOT blockade dependence and phase corrections, the hybrid's fast-transfer limit, QuTiP integration against exact constant-segment exponentials, Lindblad decay, normalization, erasure handling, finite-rise pulse area/slew limits, global-beam power counting, shot multiplication, and missing/nonfinite energy calibration. Earlier ideal-circuit and data checks remain in the suite.

Next select a device/control specification, calibrate powers and noise, improve or retrain the pulse implementations, and expand to a meaningful matched-accuracy classification benchmark. Extending full noisy dynamics beyond four atoms requires a trajectory/tensor-network implementation or additional compute.

[Model, assumptions, equations and reproduction commands](../docs/pulse-model-and-energy.md)
