# Literature-sized MNIST datasets

The default QRC/VQC comparison now uses the binary MNIST sample sizes reported by Kornjača et al., with validation reserved inside the published training pool. Both methods receive the same image indices, training-only PCA, finite-shot allocation and regression procedure in each repeat. The four-atom FRESNEL-informed compiler and optimizer are retained.

## Published sizes and available presets

| Preset | Classes | Fit-training | Validation | Test | Configuration |
|---|---|---:|---:|---:|---|
| Default binary hardware-size study | 3 versus 8 | 800 | 200 | 200 | [Binary](../configs/performance_energy_fresnel_beta.json) |
| Ten-class hardware-size study | 0–9 | 1,200 | 300 | 500 | [Ten-class](../configs/performance_energy_fresnel_beta_tenclass.json) |
| Larger ten-class simulation study | 0–9 | 8,000 | 2,000 | 1,000 | [Larger simulation](../configs/performance_energy_fresnel_beta_tenclass_simulation.json) |
| Historical exploratory sample | 3 versus 8 | 80 | 20 | 40 | [Smoke test](../configs/performance_energy_fresnel_beta_smoke.json) |

The paper reports binary training pools of 1,000 images and ten-class hardware training pools of 1,500 images in [Methods I.1.1](https://arxiv.org/html/2407.02553v1#S1.SS1.SSS1). Its classical-model selection uses an internal 80/20 training/validation split in [Methods I.2.2](https://arxiv.org/html/2407.02553v1#S1.SS2.SSS2). The larger simulations use 10,000 training-pool and 1,000 test images in [Supplement II.1](https://arxiv.org/html/2407.02553v1#S2.SS1). Our presets apply that internal validation convention: the binary pool is **800 + 200 = 1,000**, not 1,000 fit-training images plus another 200 validation images.

These are dataset-size adaptations. They do not reproduce the paper's Aquila hardware, position encoding or SVM classifier, and they do not establish the paper's reported accuracy. Named presets reject conflicting counts or classes; use `dataset_protocol="custom"` for a deliberately changed study.

## Prior test inspection and reproducibility

The [frozen exclusion manifest](../configs/mnist_prior_test_exclusions.json) records 22 Git-tracked historical split archives, their SHA-256 hashes and 2,303 unique official MNIST test indices. Each source's saved labels were checked against the original MNIST test labels. For digits 3 and 8, **1,148 of 1,984** official test images appear in that history, leaving **836 eligible images** for the updated binary test subsets.

The default and optional literature-sized presets exclude those known indices. The manifest is a lower bound on saved inspection history; it cannot rule out unrecorded inspection elsewhere. Different repeat seeds can still select overlapping eligible test subsets, so seed standard deviations are descriptive rather than independent-test confidence intervals. The historical smoke configuration keeps its previous selection behavior and makes no fresh-holdout claim.

History is not dynamically rescanned during a run. Each preparation or study records the loaded manifest hash and snapshots its contents. Future inspection history requires an explicit updated manifest and study version; adding result directories must not silently alter an existing experiment's split selection.

## Prepare data before a full simulation

From the repository root:

```bash
python -m rydberg_mnist.hardware.performance_energy \
  --config configs/performance_energy_fresnel_beta.json \
  --prepare-only \
  --output hardware_results/dataset_fresnel_beta_binary_literature
```

The [prepared binary dataset](../hardware_results/dataset_fresnel_beta_binary_literature/SUMMARY.md) contains verified original-MNIST splits, class counts, split hashes, training-only PCA/features and planned resource bounds for seeds 7, 11 and 19. Its status is `prepared_not_trained`; it contains no new QRC/VQC accuracy results. Output directories must be new; choose another name when repeating preparation.

A full noisy comparison is a separate command:

```bash
python -m rydberg_mnist.hardware.performance_energy \
  --config configs/performance_energy_fresnel_beta.json \
  --output hardware_results/new_binary_literature_run
```

The full run recomputes splits and PCA from the configuration and original data. It does not consume the prepared `dataset.npz` files as separate locked inputs. Compare its recorded split and exclusion hashes with preparation to verify that the intended data was used. Both algorithms within a run share the computed splits and PCA.

The existing [80/20/40 results](../hardware_results/performance_energy_fresnel_beta_sample/SUMMARY.md) remain historical exploratory results. Their accuracies do not describe the larger presets. This dataset update does **not** include a newly trained full benchmark.

## Training coverage and planned resources

The optimizer still has a 64-step safety cap and minibatches of eight images. That permits 512 sampled training-image draws, or **0.64 draws per fit-training image** for the binary preset. Repeated sampling does not guarantee distinct-image coverage. Every linear-head checkpoint nevertheless extracts features for all 800 fit-training images. Completed studies report actual objective-image coverage and all consumed work; the unchanged optimizer cap does not establish adequate training or convergence on the larger dataset.

At 1,000 total shots per image and three validation measurement repetitions, the default binary preparation estimates:

| Workload per seed | Planned requested shots |
|---|---:|
| QRC training and validation | 1,400,000 |
| VQC training, validation and objectives at the safety cap | 24,824,000 |
| One held-out inference pass | 200,000 |

Plateau stopping can reduce VQC consumption. Selected and terminal models need separate test passes only when their checkpoints differ. These values are planned shot counts, not executed QPU jobs or measured electrical energy. Calibrated FRESNEL joules remain unknown. See the [full accounting protocol](performance-energy-protocol.md) for cost boundaries and device-verification limits.
