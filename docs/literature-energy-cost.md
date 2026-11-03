# Literature-based Rydberg hardware energy and electricity cost

The [generated estimates](../estimates/literature_rydberg/SUMMARY.md) apply two published energy-accounting methods to the saved MNIST pulse schedules. They include digital-only, digital–analog and the separate Pasqal-native analog classifier. All quantum jobs so far ran in CPU simulation. These estimates describe hypothetical hardware execution, not the energy used by that simulation.

The [paired QRC/VQC study](../hardware_results/pasqal_qrc_mnist_3_8_seed7/SUMMARY.md) reuses these same source parameters and adds explicit shot allocations and reset/readout charges for each destructive reservoir-prefix measurement. See its [method](qrc-vqc-comparison.md) for feature caching, finite-shot budgets and training-versus-inference accounting.

## Primary sources and accounting boundaries

| Source | Published reference | Use here |
|---|---|---|
| Óscar Alves, Marco Pezzutto and Yasser Omar, [*Energetics of Rydberg-atom Quantum Computation*, arXiv:2601.03141v2](https://arxiv.org/html/2601.03141v2), 18 May 2026 | Cesium Rydberg gate experiment; source radiation for gates, traps, cooling, optical pumping and measurement | Reproduce its worked energy budget; optionally transfer constant source on-powers to our compiled controls |
| Miquel Carrasco-Codina et al., [*Energy efficiency of quantum computers*, arXiv:2605.15090v2](https://arxiv.org/html/2605.15090v2), 31 August 2026 | Rubidium neutral-atom digital reference; full electrical hardware/facility inventory, shot overheads and atom reloading | Sum the reference machine's power; transfer a clearly specified shot-time scenario to each schedule |
| Pasqal, [*Quantum Computing to Greener Calculations*](https://www.pasqal.com/blog/quantum-computing-rethinking-energy-consumption/), 5 October 2023 | Historical vendor statement of approximately 2.6 kW | Additional power-transfer sensitivity case, not a paper or current FRESNEL calibration |

Both papers are cited as versioned arXiv preprints. Neither supplies our current Pasqal FRESNEL power calibration. Their reference systems, atomic species, controls and operating protocols differ. Radiation-source energy and electrical energy have different accounting boundaries and must **not be added together**.

All numeric inputs, units, assumptions and source locators are versioned in [literature_energy.json](../configs/literature_energy.json). Generated JSON includes its SHA-256, model-file hashes, the research-profile hash, target-device hash and SDK versions. Per-input CSV preserves the MNIST test indices and schedule costs.

## Reproducing the Rydberg energetics paper

Alves et al., Sections IV.1–IV.2, give the following radiation-source powers and on-times for the reference phase-estimation example:

| Source | Power | On-time per execution |
|---|---:|---:|
| Microwave radiation | 0.0574 W | 615.999 us |
| Blue 459 nm source | 0.1 W | 279.581 us |
| Shared infrared 1040 nm source | 11 W | 54.454 us |

`sum(P_source * t_on)` gives **0.662310 mJ** of computation radiation, consistent with the paper's rounded 0.662 mJ. The 57.4 mW value is the radiation power used in its energy table; it is not the electrical rating of a microwave generator. Optical source powers already include the paper's optical-path losses. Wall-plug conversion efficiencies are absent.

The paper also uses 49 optical traps at 10 mW source power each, active for 200.9 ms; cooling at 1 mW for 100 ms; optical pumping at 1 mW for 10 ms; and four 220 uW measurement beams for 90 ms. Adding these costs and multiplying by 700 executions yields **69.504757 J**, matching the rounded **69.5 J**. Tests verify both published results independently of our classifiers.

Our optional source-radiation transfer makes additional assumptions:

- Keep those reference source powers constant throughout the compiled windows. Do not infer laser power from effective Rabi frequency.
- Charge storage Raman rotations at the paper's microwave radiation power, even though their mechanisms differ; charge storage Stark rotations at the blue reference power.
- Charge one blue reference beam per illuminated atom and one shared infrared beam per Rydberg pulse window. This scaling is an assumption about optics, not a Pasqal specification.
- Keep the reference 49 traps, preparation and measurement protocol. For the native SDK waveform, charge source power over the full bandwidth-modulated control window, including ramps and tails.

These transfers are counterfactual bookkeeping scenarios. Matching actual Rabi frequencies, beam waists, detunings, wavelengths, atomic levels, channel sharing and modulation would require a new calibration. Their architecture ranking is not validated. The original [research power profile](../configs/hardware_research.json) and [Pasqal target](../configs/pasqal_target.json) retain their unknown calibration fields; the new estimates do not populate them with transferred values.

## Whole-machine electrical energy

Carrasco-Codina et al., Appendix A.4 Table 8, inventory trap/Rydberg/other lasers, magnetic fields, camera, vacuum, classical computers, electronics, chillers and HVAC. The baseline components sum to **22,205 W**; the paper rounds this to approximately 22 kW. The additional continuous-reloading hardware is excluded. The reference supports a 400-qubit workload; allocating its full power to our small circuits assumes the entire reference machine is dedicated to this workload. We do not scale its power down by active-qubit count.

Equation 18 gives a separate reference runtime model:

```text
t_shot = t_reset + t_measurement + D * t_clock
         + beta_transport * D * t_transport + (D/2400) * t_reload
```

Here `D` is post-compilation depth, not our VQC ansatz layer count. The implementation preserves this equation and tests its numerical values. With the listed timing parameters and `beta_transport=0.25`, depths 10, 100 and 1,000 give 28.333, 103.333 and 853.333 ms. The paper's approximate prose examples (27, 91, 720 ms) do not reproduce the equation with that transport term; we record the discrepancy rather than silently changing the formula.

Our fixed-register schedule scenario uses **actual compiled control time** in place of guessing `D`. It borrows Table 9's **10 ms reset + 10 ms measurement** and explicitly sets additional transport, reconfiguration and reservoir-reloading counts to zero. Real hardware may require those operations. Reset/readout values are transferred from the reference system, not asserted for Pasqal; 2, 20 and 200 ms combined overheads are included as sensitivity cases. The native control window includes SDK finite-bandwidth modulation.

```text
t_cycle = t_compiled_control + t_reset + t_measurement
          + n_transport * t_transport + n_reconfiguration * t_clock
          + reloads_per_shot * t_reload
E_prediction = P_electrical * t_cycle * shots_per_prediction
electricity_cost = E_prediction / 3,600,000 * tariff_per_kWh
```

The default uses 1,000 shots per image; 100 and 10,000 are also examined. One common computational-basis sample estimates all Z and ZZ features, so there is no extra multiplier for the number of features. These are successful requested shots, with no modeled loss/retry charge. Full-machine energy includes its laser electrical consumption already; source radiation is not added again.

The optional 2.6 kW case transfers a vendor statement from **2023** to the same assumed timing. It is not a current FRESNEL power measurement, a statistical lower bound, or an uncertainty interval. Similarly, **0.25 EUR/kWh** is an editable arithmetic assumption, not a market quote or a provider's QPU price. Electricity cost excludes service fees and hardware purchase costs.

## Saved workloads and training projection

The estimator compiles all **400 common held-out seed-7 inputs per original architecture** and all **40 held-out native-model inputs**. It reuses saved model/PCA parameters and trains nothing. Original digital/hybrid schedules use the explicit three-level research compiler. Native schedules use QoolQit/Pulser and the pinned public FRESNEL snapshot. Their different test-set sizes are reported and do not support comparing aggregate dataset costs as if they were the same workload.

The native SPSA recipe has the following hypothetical circuit-feature budget:

| Stage | Circuit-feature evaluations |
|---|---:|
| Initial readout regularization selection: 80 train + 20 validation | 100 |
| Twelve two-sided SPSA updates: 2 x 12 x 80 | 1,920 |
| Initial and three later validation checkpoints: 4 x (80 + 20) | 400 |
| Final linear-readout fit on 80 training features | 80 |
| **Total** | **2,500** |

Adding frozen, clean and noisy scoring on 40 test inputs reproduces the archive's **2,620** compiled/emulated sequences. This check validates the projected circuit count. It does not turn simulator evaluations into physical shots.

At 1,000 shots per feature evaluation, the training projection would request 2.5 million hardware shots. It uses the saved selected model's representative control duration; training-time pulse traces were not saved. Actual optimization used exact simulated expectations. Finite-shot noisy training would follow a different trajectory and could attain different accuracy. The projection excludes external classical optimizer, PCA and linear-regression energy, queueing, idle allocation and failures. Original exact-adjoint training has no inferred hardware budget because its future hardware gradient protocol is unspecified.

## Interpretation and reproduction

At the default assumed overhead, preparation/readout dominate the sub-millisecond control schedules. This explains why control-duration ratios can greatly exceed full-cycle energy ratios. Our original digital/hybrid recipes are unsupported on the selected global-only Pasqal interface and have unequal physical fidelity. The native analog classifier has a different data/architecture budget. No energy-efficiency winner or matched-accuracy advantage follows from these tables.

Measured per-device energy and complete end-to-end pipeline energy remain unknown. Calibration of wall power, shot-cycle overhead, retries, atom reloading and classical processing is the next requirement for a device-specific comparison.

```bash
.venv-hardware/bin/python -m rydberg_mnist.hardware.literature_cost \
  --config configs/literature_energy.json \
  --output estimates/new_literature_run
.venv-hardware/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory. The generated report, per-input CSV, machine-readable estimates and PNG/SVG figures are saved together. Change the literature config for alternate powers, shot counts, overheads or tariffs, preserving the associated assumption labels.
