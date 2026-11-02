"""Multi-output linear least squares; optional L2 regularization (ridge)."""
import numpy as np


def fit_readout(features, targets, alpha=0.1):
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    mean_x, mean_y = features.mean(axis=0), targets.mean(axis=0)
    x, y = features - mean_x, targets - mean_y
    if alpha == 0:
        weights = np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    bias = mean_y - mean_x @ weights
    return weights, bias


def projected_loss(features, targets, alpha):
    """Eliminate linear weights analytically, differentiate via envelope theorem.

    Objective: (||F W + b - Y||^2 + alpha ||W||^2)/(N*K).
    Targets are one-hot classes; the scores are not calibrated probabilities.
    """
    weights, bias = fit_readout(features, targets, alpha)
    residual = features @ weights + bias - targets
    scale = targets.size
    loss = (np.sum(residual**2) + alpha * np.sum(weights**2)) / scale
    gradient = 2 * residual @ weights.T / scale
    return float(loss), gradient, weights, bias
