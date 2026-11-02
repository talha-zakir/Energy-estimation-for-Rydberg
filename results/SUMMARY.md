# Initial MNIST results

CPU statevector simulations, not QPU results. Each run uses 1,000 training examples and 200 validation examples. Binary tests use 400 official-test examples; ten-class tests use 500. Physics and two-layer architecture are fixed across tasks.

## MNIST all ten digits

Seeds: [11, 19, 7]. Values are mean test accuracy ± sample standard deviation across runs (percentage points). Seeds vary both the stratified subset and parameter initialization; this is not an isolated initialization study or a statistical significance test.

| Model | Mean ± SD | Individual runs |
|---|---:|---|
| classical pca linear | 75.33% ± 1.70 pp | 75.40%, 73.60%, 77.00% |
| encoding only linear | 75.73% ± 3.67 pp | 77.00%, 71.60%, 78.60% |
| frozen rydberg linear | 56.27% ± 4.77 pp | 59.60%, 58.40%, 50.80% |
| trained rydberg linear | 76.87% ± 3.06 pp | 79.20%, 73.40%, 78.00% |
| trained rydberg linear shots | 76.67% ± 2.84 pp | 78.60%, 73.40%, 78.00% |

Run artifacts:

- Seed 11: [metrics](mnist_10class_seed11/metrics.json), [figure](mnist_10class_seed11/comparison.png), [history](mnist_10class_seed11/history.json), [model](mnist_10class_seed11/model.npz). Runtime 35.1 s; selected iteration 33; optimizer converged: False.
- Seed 19: [metrics](mnist_10class_seed19/metrics.json), [figure](mnist_10class_seed19/comparison.png), [history](mnist_10class_seed19/history.json), [model](mnist_10class_seed19/model.npz). Runtime 37.1 s; selected iteration 34; optimizer converged: False.
- Seed 7: [metrics](mnist_10class_seed7/metrics.json), [figure](mnist_10class_seed7/comparison.png), [history](mnist_10class_seed7/history.json), [model](mnist_10class_seed7/model.npz). Runtime 41.8 s; selected iteration 40; optimizer converged: False.

## MNIST 3 vs 8

Seeds: [11, 19, 7]. Values are mean test accuracy ± sample standard deviation across runs (percentage points). Seeds vary both the stratified subset and parameter initialization; this is not an isolated initialization study or a statistical significance test.

| Model | Mean ± SD | Individual runs |
|---|---:|---|
| classical pca linear | 95.42% ± 0.29 pp | 95.75%, 95.25%, 95.25% |
| encoding only linear | 95.67% ± 0.76 pp | 95.50%, 95.00%, 96.50% |
| frozen rydberg linear | 80.83% ± 4.23 pp | 85.50%, 79.75%, 77.25% |
| trained rydberg linear | 95.25% ± 0.75 pp | 94.50%, 95.25%, 96.00% |
| trained rydberg linear shots | 95.08% ± 0.88 pp | 94.25%, 95.00%, 96.00% |

Run artifacts:

- Seed 11: [metrics](mnist_3_8_seed11/metrics.json), [figure](mnist_3_8_seed11/comparison.png), [history](mnist_3_8_seed11/history.json), [model](mnist_3_8_seed11/model.npz). Runtime 45.6 s; selected iteration 6; optimizer converged: False.
- Seed 19: [metrics](mnist_3_8_seed19/metrics.json), [figure](mnist_3_8_seed19/comparison.png), [history](mnist_3_8_seed19/history.json), [model](mnist_3_8_seed19/model.npz). Runtime 35.7 s; selected iteration 36; optimizer converged: False.
- Seed 7: [metrics](mnist_3_8_seed7/metrics.json), [figure](mnist_3_8_seed7/comparison.png), [history](mnist_3_8_seed7/history.json), [model](mnist_3_8_seed7/model.npz). Runtime 36.6 s; selected iteration 26; optimizer converged: False.

## Interpretation

Training improves on the frozen random Rydberg circuit in these runs. The encoding-only control is competitive, and these experiments do not establish a benefit attributable to Rydberg interactions or quantum advantage. The classical comparison uses only 16 PCA features; it is not a state-of-the-art MNIST benchmark. The held-out subsets are small and overlap across seeds.

The shot condition uses 1,000 sampled bitstrings per example with a refitted linear readout. It does not model control noise, atom loss, decoherence or readout errors. Shot counts are simulated; no hardware jobs were submitted.

The maximum optimizer budget is 40 iterations. A budget-limit exit is explicitly recorded, not reported as convergence. The selected model minimizes validation MSE over checkpoints; test labels do not select models. Per-run Wilson intervals are conditional on that fitted model and do not include training variation.

Regenerate with `python -m rydberg_mnist.summarize`. Physics, data checksums, split indices, package versions and all deviations from the paper are recorded in this repository.
