import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def import_feature_module():
    try:
        from eeg_feature_robustness import features
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise unittest.SkipTest(f"feature extraction dependencies unavailable: {exc}") from exc
    return features


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        module = import_feature_module()
        self.module = module
        self.np = module.np
        self.X = self.np.linspace(0.0, 1.0, 2 * 3 * 512).reshape(2, 3, 512)

    def test_time_features_shape(self):
        features = self.module.extract_time_features(self.X)
        self.assertEqual(features.shape, (2, 21))

    def test_band_features_shape(self):
        features = self.module.extract_band_features(self.X, sfreq=250)
        self.assertEqual(features.shape, (2, 12))

    def test_fft_features_shape(self):
        features = self.module.extract_fft_features(self.X, sfreq=250, n_bins=10)
        self.assertEqual(features.shape, (2, 30))


if __name__ == "__main__":
    unittest.main()
