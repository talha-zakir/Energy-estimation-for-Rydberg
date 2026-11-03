# Rydberg classification baseline

MNIST sanity checks for reusable **digital–analog and digital-only Rydberg variational classifiers with a linear regression readout**. This repository currently runs on a CPU simulator. It does not submit jobs to a quantum device.

The physical baseline is [Lu et al., Digital-analog quantum learning on Rydberg atom arrays](https://arxiv.org/abs/2401.02940), using its [public Julia implementation](https://github.com/jz-lu/daqml) as a reference. Our independent Python implementation changes the readout and training objective as requested; **these runs are an adaptation, not a reproduction of the paper's published accuracies**.

See [the paired architecture comparison](results/ARCHITECTURE_COMPARISON.md), [the paper shortlist](docs/paper-shortlist.md), [the baseline settings and deviations](docs/baseline.md), [hardware requirements](docs/hardware.md), and [all measured results](results/SUMMARY.md).

## Hardware-informed simulation

**Selected target: Pasqal FRESNEL via QoolQit/Pulser.** A pinned public device/noise specification and compatibility guard are included. Its exposed global analog controls cannot run the original digital/hybrid pair as written, so those research baselines are retained and a separate **Pasqal-native global-pulse classifier** is added with the requested linear readout. See [Pasqal results](hardware_results/PASQAL_SUMMARY.md) and [target/model details](docs/pasqal-target.md).

An optional **QuTiP three-level pulse backend** now models both architectures with finite-rise controls, Rydberg blockade, storage/Rydberg transfers, decay, dephasing and explicit static-noise assumptions. A separate **QoolQit/Pulser** check compiles and simulates the eight-atom global analog block against SDK device constraints. Full eight-qubit trained schedules have an inspectable energy ledger; full noisy dynamics were checked on a small four-atom MNIST probe.

This is a research model, not device-calibrated hardware. The probe exposes substantial pulse/noise errors, so a trustworthy energy-per-accuracy comparison needs calibration and pulse improvements. Device-calibrated joules remain unknown until measured power/efficiency and overhead data are provided. See [hardware results](hardware_results/SUMMARY.md) and [model, energy equations and setup](docs/pulse-model-and-energy.md). The ideal results below remain separate.

**Paper-based energy/cost estimates are available:** [results and figures](estimates/literature_rydberg/SUMMARY.md), [methods and source papers](docs/literature-energy-cost.md), and [editable parameters](configs/literature_energy.json). The estimator reproduces Alves et al.'s Rydberg energy example and separately uses Carrasco-Codina et al.'s electrical component inventory. It applies explicit scenarios to 400 saved inputs per original architecture and 40 Pasqal-native inputs, with shot/overhead sensitivity and an illustrative electricity tariff. These are literature transfers, not measured Pasqal costs or evidence of an energy advantage.

**Quantum reservoir computing is implemented:** [paired QRC/VQC accuracy and cost results](hardware_results/pasqal_qrc_mnist_3_8_seed7/SUMMARY.md), [method and QRC papers](docs/qrc-vqc-comparison.md), and [config](configs/pasqal_qrc_mnist.json). A fixed reservoir uses the native VQC's exact split, PCA, atoms and global input encoding, and trains only a linear readout. Endpoint and temporal-prefix readouts are compared with the saved VQC, including device noise, equal total-shot budgets and separate training/inference energy projections. This is local QoolQit/Pulser simulation on the pinned Pasqal target.

**The primary comparison now trains both methods and reports accuracy as an outcome:** [full-training results](hardware_results/performance_energy_fresnel_beta_sample/SUMMARY.md), [protocol](docs/performance-energy-protocol.md), and [configuration](configs/performance_energy_fresnel_beta.json). Both use the same four atoms, unique MNIST splits, finite-shot temporal features, and linear regression procedure. VQC stops at a validation-loss plateau or safety cap; every optimizer evaluation, validation repetition and discarded checkpoint remains in its consumed cost, even if step zero is selected. Both selected and terminal accuracies are reported. The pinned public FRESNEL profile remains provisional for the account's beta QPU, and calibrated electrical joules remain unknown. The [earlier accuracy-target protocol](docs/accuracy-energy-protocol.md) and its [historical results](hardware_results/accuracy_energy_fresnel_beta_sample/SUMMARY.md) remain available separately. Neither study establishes a hardware-energy advantage.

## Pipeline

```text
Original MNIST, official train/test split
    -> stratified training / validation / held-out test subsets
    -> PCA fitted only on training images (16 components)
    -> product-state angle encoding (8 qubits)
    -> 2 layers: local RX-RZ-RX rotations + chosen entangler
    -> Z_i and Z_i Z_j expectation values (36 features)
    -> multi-output linear least-squares regression on one-hot targets
    -> argmax of class scores
```

The digital–analog entangler is global Rydberg Hamiltonian evolution. The digital-only entangler is an explicit nearest-neighbor CNOT circuit: gates on bonds (0,1), (2,3), (4,5), (6,7), followed by (1,2), (3,4), (5,6). It follows the reference's noiseless digital baseline. It does not approximate the analog Hamiltonian. At two layers, both versions have 48 trainable angles; the digital circuit has 14 logical CNOTs. No Rydberg pulse errors or native CNOT decomposition are simulated.

Only the rotation angles are variational quantum parameters. Linear readout weights are solved analytically at each training evaluation. L2 regularization is selected on validation data; `alpha=0` selects ordinary linear regression. Scores are not probabilities. This is not regression on the numeric labels 0–9.

## Run

Python 3.11+ is recommended for `requirements-tested.txt` (these runs used Python 3.12). Detailed versions are also saved per run. The package supports Python 3.10 with compatible dependency versions installed through `pip install -e .`.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v

# Primary native analog QRC/VQC full-training study
python -m rydberg_mnist.hardware.performance_energy --output hardware_results/new_performance_energy

# Paper-aligned binary task: digits 3 and 8
python -m rydberg_mnist.experiment --config configs/mnist_3_8.json --output results/new_binary

# Adaptation to all ten MNIST digits
python -m rydberg_mnist.experiment --config configs/mnist_10class.json --output results/new_tenclass

# Additional seed, with a new output directory
python -m rydberg_mnist.experiment --config configs/mnist_10class.json --seed 23 --output results/new_seed23

# Matched digital-only experiments
python -m rydberg_mnist.experiment --config configs/mnist_3_8_digital.json --output results/new_binary_digital
python -m rydberg_mnist.experiment --config configs/mnist_10class_digital.json --output results/new_tenclass_digital

# The architecture can also be selected without a separate config
python -m rydberg_mnist.experiment --config configs/mnist_3_8.json --architecture digital --output results/another_digital

# Summarize all runs, then validate matched architecture pairs
python -m rydberg_mnist.summarize
python -m rydberg_mnist.compare
```

The first run downloads original MNIST IDX files to `data/mnist/` and verifies their published MD5 checksums. No substitute digits dataset is used. Data files are excluded from Git. Output directories must be new to prevent overwriting experiments. For new paired studies, place runs under a separate root and pass `--root that_directory` to both summary commands. The comparison deliberately rejects duplicate task/seed/architecture entries rather than choosing a result silently.

Each run saves metrics, confusion matrices, confidence intervals, training history, split indices, predictions, model/PCA parameters, runtime, and a comparison figure. `model.npz` can be reused for inference:

```python
from rydberg_mnist.experiment import predict_saved

# images: uint8 MNIST images flattened to (N, 784), values 0..255
labels = predict_saved("results/mnist_10class_seed7/model.npz", images)
# The saved model chooses the correct architecture automatically.
digital_labels = predict_saved("results/mnist_10class_digital_seed7/model.npz", images)
```

## Comparisons and correctness

All methods share identical data subsets and training-only PCA. Comparisons include linear regression on PCA features, encoding-only quantum measurements, a frozen random Rydberg circuit, trained Rydberg rotations, and a finite-shot readout of the trained circuit. The encoding-only comparison retains the nonlinear angle encoding and pairwise measurements, making it a stronger ablation than removing only the entangler from a random circuit.

Tests check the Hamiltonian against an independent Kronecker construction, a solvable noninteracting Rabi limit, CNOT truth tables/direction/order, state normalization, qubit ordering, correlated sampling, both architectures' gradients against finite differences, linear regression, split separation, saved-model inference (including earlier archives), and rejection of unmatched architecture comparisons.

## Limits and next research step

These are small-subset MNIST validation experiments, not full-dataset benchmarks or evidence of quantum advantage. Finite-shot simulation includes sampling noise only. It excludes decoherence, loss, control errors, and readout errors. Optimization has a fixed iteration budget and may stop before convergence; the optimizer status is recorded.

Transferring to a different task requires specifying that task's inputs, labels, and evaluation metric. The circuit/readout can then remain fixed while a training-only feature adapter replaces MNIST preprocessing. Re-run classical and encoding-only comparisons on the new task. A task change by itself does not establish a new scientific contribution.

For energy research, analog evolution duration is recorded separately from CPU runtime. Neither is wall-plug hardware energy. Laser powers/efficiencies, preparation, cooling, trapping, readout, shot counts, and classical processing must be measured or explicitly modeled before reporting joules.
