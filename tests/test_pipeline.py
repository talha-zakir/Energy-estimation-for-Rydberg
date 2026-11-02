"""Small synthetic-data integration test; not an MNIST accuracy benchmark."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from rydberg_mnist.experiment import run, predict_saved


class PipelineTests(unittest.TestCase):
    def test_training_and_saved_inference_roundtrip(self):
        for architecture in ("digital_analog", "digital"):
            with self.subTest(architecture=architecture):
                self.check_pipeline(architecture)

    def check_pipeline(self, architecture):
        rng = np.random.default_rng(17)
        train_images = rng.integers(0, 256, (100, 784), dtype=np.uint8)
        test_images = rng.integers(0, 256, (40, 784), dtype=np.uint8)
        train_labels = np.tile([3, 8], 50)
        test_labels = np.tile([3, 8], 20)
        prefix = "digital" if architecture == "digital" else "rydberg"
        config = {"architecture": architecture,
                  "digits": [3, 8], "seed": 7, "n_train": 60, "n_val": 20,
                  "n_test": 20, "maxiter": 2, "shots": 100,
                  "alpha_grid": [0, .1], "physics": {"n_qubits": 2, "layers": 1}}
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            output = Path(directory) / "run"
            with patch("rydberg_mnist.experiment.load_mnist", return_value=(
                      train_images, train_labels, test_images, test_labels)), redirect_stdout(io.StringIO()):
                report = run(config, output, cache=Path(directory) / "cache")
            self.assertEqual(report["architecture"], architecture)
            self.assertEqual(report["models"][f"trained_{prefix}_linear"]["test"]["n"], 20)
            self.assertLess(report["max_state_norm_error"], 1e-12)
            with np.load(output / "splits_predictions.npz") as predictions:
                images = test_images[predictions["test_idx"]]
                for model in ["classical_pca_linear", "encoding_only_linear",
                              f"frozen_{prefix}_linear", f"trained_{prefix}_linear"]:
                    np.testing.assert_array_equal(predict_saved(output / "model.npz", images, model),
                                                   predictions[model + "_test"])
                np.testing.assert_array_equal(predict_saved(output / "model.npz", images),
                                               predictions[f"trained_{prefix}_linear_test"])
                if architecture == "digital_analog":
                    # Initial committed archives have no architecture field.
                    with np.load(output / "model.npz") as saved:
                        legacy = {k: saved[k] for k in saved.files if k != "architecture"}
                    np.savez(output / "legacy.npz", **legacy)
                    np.testing.assert_array_equal(predict_saved(output / "legacy.npz", images),
                                                   predictions[f"trained_{prefix}_linear_test"])
            # Validation may correctly retain the initial checkpoint on random labels.
            # Verify optimization ran and lowered the training objective instead.
            history = json.loads((output / "history.json").read_text())
            self.assertGreater(report["optimizer"]["iterations"], 0)
            self.assertLess(history[-1]["train_objective"], history[0]["train_objective"])
            self.assertTrue((output / "comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
