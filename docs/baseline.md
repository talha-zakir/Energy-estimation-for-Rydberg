# Baseline settings and deviations

Reference: [Lu et al., arXiv:2401.02940v2](https://arxiv.org/html/2401.02940v2), Sec. II, II.2, III.1 and Appendix C.1.2; [public implementation](https://github.com/jz-lu/daqml).

## Physics retained

With `n_i = |1><1|`, we simulate

```text
H / hbar = (Omega/2) sum_i X_i - Delta sum_i n_i
           + sum_{i<j} C6 / r_ij^6 n_i n_j
U = exp(-i (H/hbar) t)
```

| Setting | Value | Basis |
|---|---|---|
| Geometry | Equally spaced 1D open chain | Paper Sec. II |
| Qubits | 8 | Main digit example / author classifier |
| Omega | 2*pi*4 rad/us | Paper Sec. II.2 |
| Delta/Omega | 0.8 | Paper Sec. II.2 |
| C6 | 2*pi*862690 rad/us um^6 | Paper Eq. 3 |
| Rb/a | 0.87 | Paper Sec. II.2 |
| Spacing a | (C6/Omega)^(1/6)/0.87, about 8.901 um | Derived |
| Time t | 0.25 us = 2*pi/Omega | Paper's main physical choice |
| Phase of global drive | 0 | Paper Sec. II |
| Rotation layout | RX layer, RZ layer, RX layer, then U | Author `gates.jl` / classifier |
| Layers | 2 | Our small test budget; paper studies multiple depths |

All pairs interact; interactions are not truncated at nearest neighbors. Qubit zero is the most significant bit in this implementation. Hyperfine-to-Rydberg manifold transfers are idealized, as in the paper's effective two-level simulation. No pulse compilation or transfer dynamics are modeled.

**Reference discrepancy:** the inspected author CLI sets `--evo_time` to `0.87` by default, whereas the manuscript motivates `0.25 us`. We explicitly choose the manuscript's `0.25 us`; the choice is stored in each configuration. Do not silently equate those values.

## Preprocessing retained or clarified

PCA reduces images to `2*n_qubits` values. The first half gives theta and the second half phi in a product state `cos(theta/2)|0> + exp(i*phi)sin(theta/2)|1>`. Following the public notebook, each image's theta half and phi half are separately shifted/scaled into `[0, pi]`. Constant halves map to zero to avoid division by zero. PCA is fitted on the training subset only.

We use original 28x28 MNIST images and its official training/test partitions. Validation examples come only from the official training partition. Train, validation and test subsets are stratified, with exact indices saved.

## Intentional modifications

| Component | Original baseline | This repository |
|---|---|---|
| Targets | Binary MNIST digit pairs | 3/8 sanity check plus separate 10-class adaptation |
| Readout | First-qubit probability | All Z_i and distinct Z_i Z_j (36 features for 8 qubits) |
| Classical head | Probability threshold | Multi-output linear regression on one-hot targets; argmax scores |
| Loss | Cross entropy | Mean squared error plus optional L2 weight penalty |
| Optimizer | AdaGrad, learning rate 0.1, 70 iterations in manuscript | L-BFGS, at most 40 iterations, exact simulator adjoints and analytic readout solves |
| Parameter budget | Depth-dependent | 48 quantum angles, plus 36*K regression weights and K biases |
| Model selection | Paper's own protocol | Minimum validation MSE, no use of test labels |
| Noise | Paper's control/position-noise models | Ideal evolution; separate multinomial finite-shot readout |
| Repeats | Paper reports many randomized trials | Three seeds in the initial repository experiments |

The regression head solves `(||F W + b - Y||^2 + alpha ||W||^2)/(N*K)`; the intercept is unpenalized. We include `alpha=0` in a predeclared validation grid, so ordinary least squares is available. When alpha is positive, the selected model is ridge regression, still a linear regression model.

Variable projection solves W,b for each quantum-angle evaluation. The envelope theorem gives the feature derivative while holding the optimal W,b fixed; we backpropagate through the unitary sequence using exact adjoints. This is a simulation technique. A device needs a separately budgeted parameter-shift or pulse-gradient method.

## Comparisons

1. Linear regression on 16 PCA features, scaled by training standard deviations.
2. Linear regression on the encoded state's Z/ZZ features with no variational circuit.
3. Linear regression on a frozen random Rydberg circuit.
4. Linear regression with trained Rydberg rotations; physical Hamiltonian stays fixed.
5. The selected trained circuit with 1,000 shared computational-basis shots per input. Refit/select only the readout using sampled training/validation features; do not retrain the quantum circuit under sampling noise.

All exact readouts use the same validation grid, except the trained circuit retains the frozen circuit's chosen alpha throughout quantum optimization. All model selection is locked before scoring the held-out labels. Exact features and shot features are different evaluation conditions; the latter is not a complete noise model.

## Reuse for a different task

Keep physical constants and circuit fixed for the first transfer experiment. Replace `data.py` and the image feature adapter with the new dataset's preprocessing; fit it only on training data. Change class targets/readout width and evaluate on a held-out split. Tune depth or input encoding only in a separately labeled follow-up experiment. Compare to the same classical and encoding-only controls before attributing gains to the Rydberg circuit.
