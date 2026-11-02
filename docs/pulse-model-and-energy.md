# Hardware-informed pulse simulation and energy ledger

This layer adds finite pulses and open-system dynamics to both VQC architectures. It is an **explicit research model**, not a calibrated model of a particular Pasqal or QuEra QPU. All previous eight-qubit ideal results remain separate.

## Libraries and scope

- **QuTiP** simulates the full three-level storage/Rydberg pulse sequences. Both architectures use the same species coefficient, geometry, noise model, control resources and detector policy.
- **QoolQit** compiles an isolated global analog block to its constrained `AnalogDevice`; **Pulser / QutipBackendV2** independently emulate that compiled block. This does not compile the full hybrid or digital circuit to an available commercial device.
- **NumPy/SciPy/scikit-learn** retain the original ideal training/data pipeline. In the initial probe, linear readouts and circuit parameters are trained ideally and then frozen for physical inference.

QoolQit is specialized for the Rydberg Analog Model. It cannot by itself turn the current arbitrary hyperfine rotations and blockade CNOTs into a supported analog-only QPU program. The shared research backend models those additional controls explicitly rather than silently dropping them.

## Three-level atom model

Each atom has two storage states `|0>`, `|1>` and a Rydberg state `|r>`. With time in microseconds, the Hamiltonian uses rad/us:

```text
H / hbar = sum_{i<j} C6 / r_ij^6 n_r,i n_r,j
           + local Raman/Stark controls on |0>,|1>
           + addressed excitation controls on |1>,|r>
           + global analog drive on |0>,|r> and Rydberg detuning.
```

Van der Waals interactions remain on during every pulse, including state transfer, basis changes and residual-leakage evolution. No pairwise blockade constraint or perfect CZ matrix is imposed.

**Digital-only:** CNOT is constructed from physical target Hadamards, a control pi / target 2pi / control pi Rydberg sequence, and physical phase corrections on both atoms. Hadamards and corrections consume pulse time and control exposure. Finite blockade produces coherent error and leakage.

**Digital–analog:** storage rotations are followed by explicit `|1> -> |r>` transfer, global analog evolution in the `|0>,|r>` subspace, and reverse transfer. Transfer phases are accounted for with signed Y pulses. Transfer time and interactions are included. In the limit of instantaneous transfer and rectangular global driving, this recovers the original ideal hybrid model; tests verify that limit.

Local rotations use triangular or trapezoidal pulses with the requested integrated rotation angle, a peak limit and a finite slew limit. The global analog block has finite rise/fall time within the original total quench duration. This changes its dynamics slightly from the rectangular ideal baseline.

## Explicit assumptions

The versioned [profile](../configs/hardware_research.json) uses the paper's approximately 8.901 um spacing and C6 coefficient. Local control rates, 10 ns ramps, 150 us Rydberg lifetime, 300 us pure-dephasing time, branching fractions, position jitter, amplitude fluctuations, detuning offsets and detector errors are **illustrative assumptions**, not measured vendor calibration. Rydberg state selection, laser transition details and attainable transfer rate still need hardware validation.

One local beam is assumed, so locally addressed operations and CNOTs are serialized. This is a conservative, nonoptimized compiler. A device supporting multiple beams, native optimized CZ gates, atom movement or parallel addressing can have very different timing and energy. The slow digital excitation rate is chosen relative to the weak interactions at the paper's spacing; it is not a typical performance claim for all neutral-atom digital gates.

Decay is modeled as Lindblad return from `|r>` into the two storage states with the configured branching fraction. Pure dephasing acts on Rydberg/storage coherences. Static Gaussian position, detuning and amplitude errors are drawn per realization and shared across inputs in that realization. These draws are paired across architectures. The reported detector-shot prediction samples the equal mixture of those states; it does not represent one experiment under a single fixed drift. Doppler motion, detailed intermediate-state scattering, control electronics response, trap loss and all real detection mechanisms are not fully modeled.

Residual Rydberg population is retained. The probe assumes it can be flagged as an erasure, assigned `Z=0`, while storage outcomes have `Z=+1,-1`; nothing is postselected away. Actual hardware might require a different detection model. Independent storage-state detection flips are included. Z and ZZ use the same sampled site outcomes.

## What ran

1. A small four-atom MNIST 3/8 probe compares ideal, coherent finite-pulse, and Lindblad/noisy inference on the same four balanced held-out examples. Ideal training uses 200 train/40 validation images, and 40 official test images. Four probe examples and two static-noise draws are far too few to establish hardware classification accuracy. Readouts are frozen, not refitted to noise.
2. Full pulse schedules for the original paired **eight-qubit trained models** are compiled and accounted for on one shared MNIST input. Full noisy eight-atom dynamics are not run: exact three-level density matrices scale as `9^n`. The current density solver deliberately rejects more than four atoms. A trajectory/tensor-network backend and a larger benchmark remain future work.
3. An independent eight-atom QoolQit/Pulser analog-block check uses real SDK device constraints. Compilation rescales geometry, amplitude and time. The compiled rectangle is checked against the original ideal unitary at a stricter tolerance than the sampled backend. Pulser appends a zero sample at its 1 ns grid endpoint, causing a small documented difference from the mathematical rectangle. Neither this check nor an SDK device definition is current calibration of a connected QPU.

See [the generated result report](../hardware_results/SUMMARY.md).

## Energy: what is known and what remains unknown

For every input and architecture, the ledger records total scheduled control time, time by stage, per-channel beam-on time and `integral Omega_eff(t)^2 dt`. For a linear ramp from a to b over dt, the latter is `dt*(a*a+a*b+b*b)/3`. Global-beam power is counted once, not multiplied by atom count. These are control-exposure quantities, **not joules**.

To obtain optical energy, supply calibrated coefficients for each laser/control channel:

```text
P_channel(t) = bias_W + k_W_per_(rad/us)^2 * Omega_eff(t)^2
E_optical = sum_channels integral P_channel(t) dt
E_control_electrical = sum_channels E_optical / wall_efficiency
```

This parameterization is only valid if supported by the chosen drive calibration. For a two-photon Rydberg transition it must account for both optical legs, their detuning and modulation scheme. An effective two-photon Rabi frequency alone does not specify laser power. Bias power applies during scheduled channel pulses; lasers kept on between pulses need corresponding background/duty-cycle accounting.

The optional full quantum-cycle estimate adds preparation/loading/cooling outside the compiled encoding, measurement/reset duration, background wall power, and separately measured fixed per-shot energy. Background must exclude controlled-laser power already counted. Logical image encoding is already inside the schedule and must not be counted a second time. Missing coefficients remain JSON `null`; they are not replaced by zero or guessed powers.

The end-to-end pipeline also includes classical preprocessing, training/optimizer work and linear-readout processing. Those measurements are absent, so **end-to-end joules remain unknown even if quantum-control coefficients are supplied**. No energy-saving claim follows from shorter pulse duration or from the summed exposure of different channels.

`workload_budget()` multiplies a calibrated per-shot ledger by explicit example, shot and circuit-evaluation counts. Solver trajectories/static-noise realizations are simulation samples, not physical hardware shots. The current ideal adjoint optimizer does not determine the circuit count for future hardware training; its hardware gradient method and batching must be specified separately.

For a scientific comparison, first calibrate a target device and validate the pulse compiler, then evaluate both methods at comparable accuracy/fidelity and training/inference workload. Comparing our two raw schedules alone would not establish hardware energy efficiency.

## Reproduce

Use a separate environment; the original ideal lock pins a SciPy version excluded by the installed QuTiP release.

```bash
python -m venv .venv-hardware
.venv-hardware/bin/python -m pip install -r requirements-hardware-tested.txt
.venv-hardware/bin/python -m pip install --no-deps -e .
.venv-hardware/bin/python -m unittest discover -s tests -v

# Choose fresh output directories; existing artifacts are preserved.
.venv-hardware/bin/python -m rydberg_mnist.hardware.sdk_check --output hardware_results/new_sdk_check
.venv-hardware/bin/python -m rydberg_mnist.hardware.experiment --output hardware_results/new_probe
.venv-hardware/bin/python -m rydberg_mnist.hardware.experiment --account-saved --output hardware_results/new_accounting
```

The probe can resume a failed calculation with `--resume`, reusing ideal training artifacts only if their configurations match. QuTiP's pure-state solver uses high-order integration and checks norm preservation; Lindblad integration checks trace preservation. Power calibrations, profiles and emitted schedules are all inspectable JSON.

## Sources

- [QoolQit overview](https://docs.pasqal.com/qoolqit/) and [device compilation](https://docs.pasqal.com/qoolqit/fundamentals/compilation/device_and_compilation/): supported Rydberg Analog Model, device constraints and rescaling.
- [Pulser noise modeling](https://pulser.readthedocs.io/en/stable/noise_model.html): types of neutral-atom noise and their required parameters.
- [QuTiP solvers](https://qutip.readthedocs.io/en/stable/apidoc/solver.html): state-vector and Lindblad dynamics.
- [Saffman, Walker and Mølmer, Quantum information with Rydberg atoms](https://arxiv.org/abs/0909.4777): blockade gates and physical error mechanisms.
- [Lu et al., Digital-analog quantum learning on Rydberg atom arrays](https://arxiv.org/abs/2401.02940): original physical baseline and effective storage/analog picture.
