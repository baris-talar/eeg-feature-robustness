import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def import_models_module():
    try:
        from eeg_feature_robustness import models
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise unittest.SkipTest(f"modeling dependencies unavailable: {exc}") from exc
    return models


class ModelingTests(unittest.TestCase):
    def setUp(self):
        module = import_models_module()
        self.module = module
        self.np = module.np
        zeros = self.np.zeros((12, 4))
        ones = self.np.ones((12, 4))
        self.X = self.np.vstack([zeros, ones])
        self.y = self.np.array([0] * 12 + [1] * 12)
        self.grouped_X = self.np.vstack([
            self.np.zeros((2, 4)),
            self.np.ones((2, 4)),
            self.np.zeros((2, 4)),
            self.np.ones((2, 4)),
            self.np.zeros((2, 4)),
            self.np.ones((2, 4)),
            self.np.zeros((2, 4)),
            self.np.ones((2, 4)),
        ])
        self.grouped_y = self.np.array([0, 0, 1, 1] * 4)
        self.groups = self.np.repeat(self.np.arange(4), 4)

    def test_compute_gap(self):
        gap = self.module.compute_gap(0.8, 0.6)
        self.assertEqual(gap["gap"], 0.2)
        self.assertEqual(gap["normalised_gap"], 0.25)

    def test_evaluate_within_schema(self):
        results = self.module.evaluate_within(self.X, self.y, n_splits=2)
        self.assertEqual(set(results), {"LogReg", "SVM", "RandomForest"})
        for model_result in results.values():
            self.assertIn("mean", model_result)
            self.assertIn("std", model_result)
            self.assertIn("ci95", model_result)
            self.assertEqual(len(model_result["scores"]), 2)

    def test_evaluate_within_grouped_schema(self):
        results = self.module.evaluate_within(
            self.grouped_X,
            self.grouped_y,
            n_splits=2,
            groups=self.groups,
            pca_components=2,
        )
        for model_result in results.values():
            self.assertEqual(model_result["protocol"], "subject_grouped_stratified")
            self.assertEqual(model_result["n_splits"], 2)

    def test_evaluate_cross_schema(self):
        target_groups = self.np.tile(self.np.arange(4), 6)
        results = self.module.evaluate_cross(self.X, self.y, self.X, self.y, target_groups=target_groups)
        self.assertEqual(set(results), {"LogReg", "SVM", "RandomForest"})
        for model_result in results.values():
            self.assertIn("balanced_accuracy", model_result)
            self.assertIn("ci95", model_result)
            self.assertIn("subject_scores", model_result)

    def test_run_per_subject_analysis_unknown_dataset(self):
        with self.assertRaises(ValueError):
            self.module.run_per_subject_analysis("unknown")

    def test_run_per_subject_analysis_bci2a_smoke(self):
        np = self.np
        rng = np.random.default_rng(0)
        n_subjects = 3
        trials_per_subject = 20
        n_features = 8
        n_total = n_subjects * trials_per_subject

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            y = np.tile([0, 1], n_total // 2)
            np.save(tmpdir_path / "bci2a_y.npy", y)
            for name in ["bci_A", "bci_B", "bci_C"]:
                np.save(tmpdir_path / f"{name}.npy", rng.standard_normal((n_total, n_features)))

            metadata_path = tmpdir_path / "bci2a_trial_metadata.csv"
            subjects = np.repeat(np.arange(1, n_subjects + 1), trials_per_subject)
            metadata = "trial_index,subject\n" + "\n".join(
                f"{i},{subj}" for i, subj in enumerate(subjects)
            )
            metadata_path.write_text(metadata)

            from eeg_feature_robustness import config
            original_metadata = config.TRIAL_METADATA_FILES["bci2a"]
            config.TRIAL_METADATA_FILES["bci2a"] = metadata_path
            try:
                summary = self.module.run_per_subject_analysis(
                    "bci2a", results_dir=tmpdir_path,
                )
            finally:
                config.TRIAL_METADATA_FILES["bci2a"] = original_metadata

            for feat in ["TimeDomain", "BandPower", "FFT"]:
                for model in ["LogReg", "SVM", "RandomForest"]:
                    entry = summary[feat][model]
                    self.assertIn("mean", entry)
                    self.assertEqual(len(entry["per_subject"]), n_subjects)

            saved = tmpdir_path / "bci2a_per_subject_results.json"
            self.assertTrue(saved.exists())
            with saved.open() as handle:
                self.assertEqual(set(json.load(handle).keys()), {"TimeDomain", "BandPower", "FFT"})


if __name__ == "__main__":
    unittest.main()
