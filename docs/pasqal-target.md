# Pasqal target and native analog classifier

The selected provider is **Pasqal**, using **QoolQit/Pulser**. The default target is now a pinned public **FRESNEL** specification retrieved through `PasqalCloudConnection().fetch_available_devices()`. This was an unauthenticated, read-only specification fetch; no QPU or cloud-emulator jobs were submitted.

The public snapshot is [versioned here](../configs/pasqal_fresnel_device.json); its hash and retrieval provenance are in [the target configuration](../configs/pasqal_target.json). A public device specification is not proof of account access or calibration for a future booked run.

## Architecture compatibility

The retrieved FRESNEL interface exposes one **global ground–Rydberg** channel. It supplies its own geometry/control limits, modulation bandwidth and noise model. It does not expose the individually addressed storage rotations or storage-to-Rydberg transfers used by our earlier pair of VQCs.

| Implementation | Status for this selected target |
|---|---|
| Original digital-only VQC | Preserved research baseline; unsupported as written on this interface |
| Original digital–analog VQC | Preserved research baseline; unsupported as written on this interface |
| Isolated global analog block | Compiled and simulated with the FRESNEL specification |
| New Pasqal-native analog classifier | Uses only the exposed global drive; compiled and simulated locally |

The Pasqal backend rejects requests to run the original digital or hybrid circuits. It also rejects a missing device-noise model rather than silently reporting an ideal run as noisy. The target configuration cannot be passed to the earlier three-level research compiler. These checks prevent unsupported local controls from being treated as available hardware operations.

Pasqal's [QPU catalog](https://docs.pasqal.com/qpu-emulators/qpus/) and [SDK device compilation documentation](https://docs.pasqal.com/qoolqit/fundamentals/compilation/device_and_compilation/) describe the analog devices and retrieval of device-specific specifications. This capability assessment is specific to the retrieved interface; it is not a claim about every present or future Pasqal research system.

## Separate Pasqal-native classifier

The user-selected approach retains both earlier research baselines and adds this separate analog architecture:

```text
MNIST 3/8 images
  -> training-only PCA (8 features), training-only scaling, clipping
  -> four global pulse slots (amplitude and detuning encode the features)
  -> train eight bounded pulse-offset parameters
  -> measure Z and ZZ features
  -> linear regression readout -> digit class
```

Atoms start in their ground states. There is no ideal product-state initialization, individually addressed Raman gate, storage manifold transfer, or CNOT. Four atoms occupy a fixed chain at the target's minimum spacing. Four 128 ns slots use 16 ns ramps, for 512 ns of programmed instruction time before bandwidth tail effects. QoolQit compiles every evaluated pulse program; Pulser's `QutipBackendV2` simulates its finite-bandwidth response.

The input and trainable offsets are mapped through bounded functions to pulse amplitudes/detunings below the target's limits. Compilation is interaction-limited, keeping atom geometry and instruction duration fixed across data and parameter changes. Only the global waveforms carry the data. Tests verify this property.

This is a new hardware-adapted analog ansatz, **not a reproduction of Lu et al.'s digital–analog circuit**. Its data budget, architecture and training method differ from the earlier eight-qubit comparison. Its accuracy and energy must not be compared directly to those earlier results as if only the hardware backend changed.

## Training and evaluation

The initial demonstration uses 80 training, 20 validation and 40 held-out official-test MNIST images, one seed, four atoms and eight trainable pulse parameters. It is a small pipeline check, not a full MNIST benchmark.

SPSA trains the global pulses through the finite-bandwidth coherent simulator, with an analytic linear regression readout solve per objective evaluation. The linear regularization strength is chosen on validation data, and the pulse checkpoint minimizes validation MSE. The iteration budget is fixed; convergence is not claimed. All choices are frozen before test scoring.

Three test conditions are retained:

1. Finite-bandwidth coherent pulse inference.
2. Quantum-state expectations with the **supplied FRESNEL snapshot noise model**.
3. Shared-bitstring features using 1,000 simulated shots per input, including SDK detector/SPAM errors.

The noisy conditions use the coherent-trained readout without noise-specific refitting. Occupation/correlation observables describe the state, while sampled bitstrings include detection errors; those conditions are reported separately. The public snapshot's noise model has not been independently validated against a live device. See Pasqal's [device-noise emulation guide](https://docs.pasqal.com/qoolqit/extended_usage/device_noise/) for the SDK mechanism used.

Classical PCA + linear regression and frozen-pulse + linear regression are included as controls. The existing original digital/hybrid benchmarks remain unchanged.

## Energy and deployment limits

An energy ledger is generated from the actual compiled, bandwidth-modulated global waveform. It includes the SDK waveform tail and records squared-drive exposure. Beam powers, wall efficiencies, cooling/loading/readout/reset overheads and classical processing energy remain unknown. They are not supplied by the public device specification, so joule totals stay null.

The fixed program duration is not complete shot-cycle time. A waveform compatible with an SDK device is also not automatically ready for cloud submission: account access, register/layout rules, up-to-date specifications and booking-specific calibration still need validation. No production deployment or energy advantage is claimed.

## Commands

Use the hardware environment documented in [the pulse-model guide](pulse-model-and-energy.md).

```bash
# Local control/noise audit of the selected FRESNEL snapshot
.venv-hardware/bin/python -m rydberg_mnist.hardware.pasqal --output hardware_results/new_pasqal_check

# Separate Pasqal-native classifier; new output directory required
.venv-hardware/bin/python -m rydberg_mnist.hardware.pasqal_native --config configs/pasqal_native_mnist.json --output hardware_results/new_native_classifier

# Regenerate the initial Pasqal result summary
.venv-hardware/bin/python -m rydberg_mnist.hardware.pasqal_summary
```

`rydberg_mnist.hardware.pasqal_native.predict_saved()` reloads the fitted PCA, pulse parameters and regression readout, validates the device-snapshot hash, and simulates inference through the same backend. Changes to the device snapshot require explicit provenance updates and revalidation.

Results: [Pasqal target and classifier summary](../hardware_results/PASQAL_SUMMARY.md).
