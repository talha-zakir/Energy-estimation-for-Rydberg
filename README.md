# Rydberg classification baseline

MNIST sanity checks for a reusable **digital–analog Rydberg variational classifier with a linear regression readout**. This repository currently runs on a CPU simulator. It does not submit jobs to a quantum device.

The physical baseline is [Lu et al., Digital-analog quantum learning on Rydberg atom arrays](https://arxiv.org/abs/2401.02940), using its [public Julia implementation](https://github.com/jz-lu/daqml) as a reference. Our independent Python implementation changes the readout and training objective as requested; **these runs are an adaptation, not a reproduction of the paper's published accuracies**.

See [the paper shortlist](docs/paper-shortlist.md), [the baseline settings and deviations](docs/baseline.md), [hardware requirements](docs/hardware.md), and [measured results](results/SUMMARY.md).

## Pipeline

```text
Original MNIST, official train/test split
    -> stratified training / validation / held-out test subsets
    -> PCA fitted only on training images (16 components)
    -> product-state angle encoding (8 qubits)
    -> 2 layers: local RX-RZ-RX rotations + Rydberg Hamiltonian evolution
    -> Z_i and Z_i Z_j expectation values (36 features)
    -> multi-output linear least-squares regression on one-hot targets
    -> argmax of class scores
```

Only the rotation angles are variational quantum parameters. Linear readout weights are solved analytically at each training evaluation. L2 regularization is selected on validation data; `alpha=0` selects ordinary linear regression. Scores are not probabilities. This is not regression on the numeric labels 0–9.

## Run

Python 3.11+ is recommended for `requirements-tested.txt` (these runs used Python 3.12). Detailed versions are also saved per run. The package supports Python 3.10 with compatible dependency versions installed through `pip install -e .`.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v

# Paper-aligned binary task: digits 3 and 8
python -m rydberg_mnist.experiment --config configs/mnist_3_8.json --output results/new_binary

# Adaptation to all ten MNIST digits
python -m rydberg_mnist.experiment --config configs/mnist_10class.json --output results/new_tenclass

# Additional seed, with a new output directory
python -m rydberg_mnist.experiment --config configs/mnist_10class.json --seed 23 --output results/new_seed23
```

The first run downloads original MNIST IDX files to `data/mnist/` and verifies their published MD5 checksums. No substitute digits dataset is used. Data files are excluded from Git. Output directories must be new to prevent overwriting experiments.

Each run saves metrics, confusion matrices, confidence intervals, training history, split indices, predictions, model/PCA parameters, runtime, and a comparison figure. `model.npz` can be reused for inference:

```python
from rydberg_mnist.experiment import predict_saved

# images: uint8 MNIST images flattened to (N, 784), values 0..255
labels = predict_saved("results/mnist_10class_seed7/model.npz", images)
```

## Comparisons and correctness

All methods share identical data subsets and training-only PCA. Comparisons include linear regression on PCA features, encoding-only quantum measurements, a frozen random Rydberg circuit, trained Rydberg rotations, and a finite-shot readout of the trained circuit. The encoding-only comparison retains the nonlinear angle encoding and pairwise measurements, making it a stronger ablation than removing only the entangler from a random circuit.

Tests check the Hamiltonian against an independent Kronecker construction, a solvable noninteracting Rabi limit, state normalization, qubit ordering, correlated sampling, analytic gradients against finite differences, linear regression, split separation, and saved-model inference.

## Limits and next research step

These are small-subset MNIST validation experiments, not full-dataset benchmarks or evidence of quantum advantage. Finite-shot simulation includes sampling noise only. It excludes decoherence, loss, control errors, and readout errors. Optimization has a fixed iteration budget and may stop before convergence; the optimizer status is recorded.

Transferring to a different task requires specifying that task's inputs, labels, and evaluation metric. The circuit/readout can then remain fixed while a training-only feature adapter replaces MNIST preprocessing. Re-run classical and encoding-only comparisons on the new task. A task change by itself does not establish a new scientific contribution.

For energy research, analog evolution duration is recorded separately from CPU runtime. Neither is wall-plug hardware energy. Laser powers/efficiencies, preparation, cooling, trapping, readout, shot counts, and classical processing must be measured or explicitly modeled before reporting joules.
