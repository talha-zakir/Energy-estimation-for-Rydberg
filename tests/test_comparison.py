"""Reject misleading architecture comparisons even when scores look comparable."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rydberg_mnist.compare import validate_pair


class ComparisonTests(unittest.TestCase):
    def make_pair(self, root):
        directories = [root / "analog", root / "digital"]
        config = {"digits": [3, 8], "seed": 7, "n_train": 2, "n_val": 2,
                  "n_test": 2, "maxiter": 40, "shots": 1000, "alpha_grid": [0, .1],
                  "physics": {"n_qubits": 8, "layers": 2}}
        split = {"train_idx": [0, 1], "validation_idx": [2, 3], "test_idx": [0, 1],
                 "train_labels": [3, 8], "validation_labels": [3, 8], "test_labels": [3, 8]}
        for model in ("classical_pca_linear", "encoding_only_linear"):
            for name in ("train", "validation", "test"):
                split[f"{model}_{name}"] = [3, 8]
        model = {"initial_parameters": [0., 1.], "pca_components": [[1., 0.]],
                 "pca_mean": [0., 0.], "pca_std": [1.], "digits": [3, 8]}
        for directory, architecture in zip(directories, ("digital_analog", "digital")):
            directory.mkdir()
            (directory / "metrics.json").write_text(json.dumps({"architecture": architecture,
                            "config": config, "optimizer": {"method": "same optimizer"}}))
            np.savez(directory / "splits_predictions.npz", **split)
            np.savez(directory / "model.npz", **model)
        return directories

    def test_identical_scores_do_not_hide_different_test_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            analog, digital = self.make_pair(Path(directory))
            validate_pair(analog, digital)
            with np.load(digital / "splits_predictions.npz") as saved:
                changed = {key: saved[key] for key in saved.files}
            changed["test_idx"] = [4, 5]
            np.savez(digital / "splits_predictions.npz", **changed)
            with self.assertRaises(AssertionError):
                validate_pair(analog, digital)

    def test_different_preprocessing_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            analog, digital = self.make_pair(Path(directory))
            with np.load(digital / "model.npz") as saved:
                changed = {key: saved[key] for key in saved.files}
            changed["pca_components"] = [[0., 1.]]
            np.savez(digital / "model.npz", **changed)
            with self.assertRaises(AssertionError):
                validate_pair(analog, digital)


if __name__ == "__main__":
    unittest.main()
