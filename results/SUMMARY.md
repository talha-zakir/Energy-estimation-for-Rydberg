# Initial MNIST results

CPU statevector simulations, not QPU results. Each run uses 1,000 training examples and 200 validation examples. Binary tests use 400 official-test examples; ten-class tests use 500. Both architectures use two layers and identical trainable parameter counts. See [the paired architecture comparison](ARCHITECTURE_COMPARISON.md).

## MNIST all ten digits — digital

Seeds: [7, 11, 19]. Values are mean test accuracy ± sample standard deviation across runs (percentage points). Seeds vary both the stratified subset and parameter initialization; this is not an isolated initialization study or a statistical significance test.

| Model | Mean ± SD | Individual runs |
|---|---:|---|
| classical pca linear | 75.33% ± 1.70 pp | 77.00%, 75.40%, 73.60% |
| encoding only linear | 75.73% ± 3.67 pp | 78.60%, 77.00%, 71.60% |
| frozen digital linear | 43.73% ± 8.97 pp | 34.20%, 52.00%, 45.00% |
| trained digital linear | 68.47% ± 2.64 pp | 70.80%, 69.00%, 65.60% |
| trained digital linear shots | 68.33% ± 2.39 pp | 70.00%, 69.40%, 65.60% |

Run artifacts:

- Seed 7: [metrics](mnist_10class_digital_seed7/metrics.json), [figure](mnist_10class_digital_seed7/comparison.png), [history](mnist_10class_digital_seed7/history.json), [model](mnist_10class_digital_seed7/model.npz). Runtime 38.3 s; selected iteration 34; optimizer converged: False.
- Seed 11: [metrics](mnist_10class_digital_seed11/metrics.json), [figure](mnist_10class_digital_seed11/comparison.png), [history](mnist_10class_digital_seed11/history.json), [model](mnist_10class_digital_seed11/model.npz). Runtime 35.0 s; selected iteration 19; optimizer converged: False.
- Seed 19: [metrics](mnist_10class_digital_seed19/metrics.json), [figure](mnist_10class_digital_seed19/comparison.png), [history](mnist_10class_digital_seed19/history.json), [model](mnist_10class_digital_seed19/model.npz). Runtime 36.2 s; selected iteration 37; optimizer converged: False.

## MNIST all ten digits — digital_analog

Seeds: [7, 11, 19]. Values are mean test accuracy ± sample standard deviation across runs (percentage points). Seeds vary both the stratified subset and parameter initialization; this is not an isolated initialization study or a statistical significance test.

| Model | Mean ± SD | Individual runs |
|---|---:|---|
| classical pca linear | 75.33% ± 1.70 pp | 77.00%, 75.40%, 73.60% |
| encoding only linear | 75.73% ± 3.67 pp | 78.60%, 77.00%, 71.60% |
| frozen rydberg linear | 56.27% ± 4.77 pp | 50.80%, 59.60%, 58.40% |
| trained rydberg linear | 76.87% ± 3.06 pp | 78.00%, 79.20%, 73.40% |
| trained rydberg linear shots | 76.67% ± 2.84 pp | 78.00%, 78.60%, 73.40% |

Run artifacts:

- Seed 7: [metrics](mnist_10class_seed7/metrics.json), [figure](mnist_10class_seed7/comparison.png), [history](mnist_10class_seed7/history.json), [model](mnist_10class_seed7/model.npz). Runtime 41.8 s; selected iteration 40; optimizer converged: False.
- Seed 11: [metrics](mnist_10class_seed11/metrics.json), [figure](mnist_10class_seed11/comparison.png), [history](mnist_10class_seed11/history.json), [model](mnist_10class_seed11/model.npz). Runtime 35.1 s; selected iteration 33; optimizer converged: False.
- Seed 19: [metrics](mnist_10class_seed19/metrics.json), [figure](mnist_10class_seed19/comparison.png), [history](mnist_10class_seed19/history.json), [model](mnist_10class_seed19/model.npz). Runtime 37.1 s; selected iteration 34; optimizer converged: False.

## MNIST 3 vs 8 — digital

Seeds: [7, 11, 19]. Values are mean test accuracy ± sample standard deviation across runs (percentage points). Seeds vary both the stratified subset and parameter initialization; this is not an isolated initialization study or a statistical significance test.

| Model | Mean ± SD | Individual runs |
|---|---:|---|
| classical pca linear | 95.42% ± 0.29 pp | 95.25%, 95.75%, 95.25% |
| encoding only linear | 95.67% ± 0.76 pp | 96.50%, 95.50%, 95.00% |
| frozen digital linear | 77.50% ± 6.61 pp | 70.00%, 80.00%, 82.50% |
| trained digital linear | 92.00% ± 1.89 pp | 94.00%, 90.25%, 91.75% |
| trained digital linear shots | 92.08% ± 1.63 pp | 93.75%, 90.50%, 92.00% |

Run artifacts:

- Seed 7: [metrics](mnist_3_8_digital_seed7/metrics.json), [figure](mnist_3_8_digital_seed7/comparison.png), [history](mnist_3_8_digital_seed7/history.json), [model](mnist_3_8_digital_seed7/model.npz). Runtime 38.9 s; selected iteration 40; optimizer converged: False.
- Seed 11: [metrics](mnist_3_8_digital_seed11/metrics.json), [figure](mnist_3_8_digital_seed11/comparison.png), [history](mnist_3_8_digital_seed11/history.json), [model](mnist_3_8_digital_seed11/model.npz). Runtime 36.2 s; selected iteration 12; optimizer converged: False.
- Seed 19: [metrics](mnist_3_8_digital_seed19/metrics.json), [figure](mnist_3_8_digital_seed19/comparison.png), [history](mnist_3_8_digital_seed19/history.json), [model](mnist_3_8_digital_seed19/model.npz). Runtime 36.3 s; selected iteration 40; optimizer converged: False.

## MNIST 3 vs 8 — digital_analog

Seeds: [7, 11, 19]. Values are mean test accuracy ± sample standard deviation across runs (percentage points). Seeds vary both the stratified subset and parameter initialization; this is not an isolated initialization study or a statistical significance test.

| Model | Mean ± SD | Individual runs |
|---|---:|---|
| classical pca linear | 95.42% ± 0.29 pp | 95.25%, 95.75%, 95.25% |
| encoding only linear | 95.67% ± 0.76 pp | 96.50%, 95.50%, 95.00% |
| frozen rydberg linear | 80.83% ± 4.23 pp | 77.25%, 85.50%, 79.75% |
| trained rydberg linear | 95.25% ± 0.75 pp | 96.00%, 94.50%, 95.25% |
| trained rydberg linear shots | 95.08% ± 0.88 pp | 96.00%, 94.25%, 95.00% |

Run artifacts:

- Seed 7: [metrics](mnist_3_8_seed7/metrics.json), [figure](mnist_3_8_seed7/comparison.png), [history](mnist_3_8_seed7/history.json), [model](mnist_3_8_seed7/model.npz). Runtime 36.6 s; selected iteration 26; optimizer converged: False.
- Seed 11: [metrics](mnist_3_8_seed11/metrics.json), [figure](mnist_3_8_seed11/comparison.png), [history](mnist_3_8_seed11/history.json), [model](mnist_3_8_seed11/model.npz). Runtime 45.6 s; selected iteration 6; optimizer converged: False.
- Seed 19: [metrics](mnist_3_8_seed19/metrics.json), [figure](mnist_3_8_seed19/comparison.png), [history](mnist_3_8_seed19/history.json), [model](mnist_3_8_seed19/model.npz). Runtime 35.7 s; selected iteration 36; optimizer converged: False.

## Interpretation

Compare each trained model with its corresponding frozen circuit. The encoding-only control remains relevant, and these experiments do not establish a benefit attributable to Rydberg interactions or quantum advantage. The classical comparison uses only 16 PCA features; it is not a state-of-the-art MNIST benchmark. The held-out subsets are small and overlap across seeds.

The shot condition uses 1,000 sampled bitstrings per example with a refitted linear readout. It does not model control noise, atom loss, decoherence or readout errors. Shot counts are simulated; no hardware jobs were submitted.

The maximum optimizer budget is 40 iterations. A budget-limit exit is explicitly recorded, not reported as convergence. The selected model minimizes validation MSE over checkpoints; test labels do not select models. Per-run Wilson intervals are conditional on that fitted model and do not include training variation.

Regenerate with `python -m rydberg_mnist.summarize`. Physics, data checksums, split indices, package versions and all deviations from the paper are recorded in this repository.
