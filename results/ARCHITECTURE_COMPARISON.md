# Digital-only versus digital–analog VQC

CPU simulations only. Each architecture uses the same 8 qubits, two layers, 48 trainable RX/RZ angles, 16 PCA inputs, 36 measured Z/ZZ features, and linear regression readout. All runs have a 40-iteration L-BFGS budget. Readout alpha is selected separately by the same validation procedure; checkpoint selection uses validation MSE. Test labels do not tune either architecture.

The digital entangler contains nearest-neighbor CNOTs on even bonds followed by odd bonds, matching Lu et al.'s noiseless digital reference. It is an ideal gate circuit, not a digital approximation of the analog Hamiltonian.

The comparison checks identical split indices/labels, initial angles, fitted PCA, classical/encoding-only predictions, shot budget, hyperparameter grid, and training budget before including a pair. Original analog runs are retained.

## Exact expectations

Mean test accuracy ± sample SD across seeds (percentage points). The difference is digital minus digital–analog.

| Task | Seeds | Digital–analog | Digital only | Mean difference |
|---|---|---:|---:|---:|
| All ten digits | 7, 11, 19 | 76.87% ± 3.06 pp | 68.47% ± 2.64 pp | -8.40 pp |
| 3 vs 8 | 7, 11, 19 | 95.25% ± 0.75 pp | 92.00% ± 1.89 pp | -3.25 pp |

## 1,000-shot readout

Mean test accuracy ± sample SD across seeds (percentage points). The difference is digital minus digital–analog.

| Task | Seeds | Digital–analog | Digital only | Mean difference |
|---|---|---:|---:|---:|
| All ten digits | 7, 11, 19 | 76.67% ± 2.84 pp | 68.33% ± 2.39 pp | -8.33 pp |
| 3 vs 8 | 7, 11, 19 | 95.08% ± 0.88 pp | 92.08% ± 1.63 pp | -3.00 pp |

## Resource interpretation

| Per ansatz circuit | Digital–analog | Digital only |
|---|---:|---:|
| Trainable single-qubit rotations | 48 | 48 |
| Logical CNOT gates | 0 | 14 |
| CNOT parallel sublayers | 0 | 4 |
| Global analog evolution blocks | 2 | 0 |
| Global analog evolution duration | 0.5 us | Not applicable |

Counts exclude input preparation/readout. Matching layer and parameter counts does not match physical time, fidelity, laser energy or native gate count. The digital model does not simulate Rydberg pulses or CNOT errors. Native CNOT decomposition and its duration remain device-dependent. These results cannot establish which architecture uses less hardware energy.

## Limits

Only three seeds and small held-out MNIST subsets are used. Seeds change both data selection and initialization; test subsets overlap between seeds. The initial analog test results were already inspected before adding the digital arm, so this is an exploratory paired comparison, not a fresh confirmatory test. Quantum training is noiseless; the shot condition only resamples final features and refits the linear head. Optimization can stop at its budget limit. No quantum advantage or hardware superiority is established.

## Paired artifacts

- Digits [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], seed 7: [digital–analog](mnist_10class_seed7/metrics.json), [digital](mnist_10class_digital_seed7/metrics.json). Pairing checks passed.
- Digits [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], seed 11: [digital–analog](mnist_10class_seed11/metrics.json), [digital](mnist_10class_digital_seed11/metrics.json). Pairing checks passed.
- Digits [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], seed 19: [digital–analog](mnist_10class_seed19/metrics.json), [digital](mnist_10class_digital_seed19/metrics.json). Pairing checks passed.
- Digits [3, 8], seed 7: [digital–analog](mnist_3_8_seed7/metrics.json), [digital](mnist_3_8_digital_seed7/metrics.json). Pairing checks passed.
- Digits [3, 8], seed 11: [digital–analog](mnist_3_8_seed11/metrics.json), [digital](mnist_3_8_digital_seed11/metrics.json). Pairing checks passed.
- Digits [3, 8], seed 19: [digital–analog](mnist_3_8_seed19/metrics.json), [digital](mnist_3_8_digital_seed19/metrics.json). Pairing checks passed.

Regenerate: `python -m rydberg_mnist.compare`.
