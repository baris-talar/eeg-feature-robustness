"""Feature-extraction utilities and pipeline entrypoints."""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch

from .config import FEATURE_ARRAYS, PREPROCESSED_ARRAYS, RESULTS_DIR

BANDS = {
    "theta": (4, 8),
    "mu_alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 40),
}


def extract_time_features(X):
    """Return richer per-channel time-domain descriptors.

    The original mean/variance baseline was too weak after 4-40 Hz filtering:
    channel means are close to zero, leaving mostly amplitude variance. This
    descriptor set keeps the comparison classical while giving the time-domain
    representation access to waveform shape information.
    """
    mean_features = X.mean(axis=2)
    std_features = X.std(axis=2)
    rms_features = np.sqrt(np.mean(np.square(X), axis=2))
    line_length = np.mean(np.abs(np.diff(X, axis=2)), axis=2)
    zero_crossing_rate = np.mean(np.diff(np.signbit(X), axis=2), axis=2)

    first_derivative = np.diff(X, axis=2)
    second_derivative = np.diff(first_derivative, axis=2)
    activity = np.var(X, axis=2)
    derivative_var = np.var(first_derivative, axis=2)
    second_derivative_var = np.var(second_derivative, axis=2)
    eps = np.finfo(float).eps
    mobility = np.sqrt(derivative_var / np.maximum(activity, eps))
    derivative_mobility = np.sqrt(second_derivative_var / np.maximum(derivative_var, eps))
    complexity = derivative_mobility / np.maximum(mobility, eps)

    features = np.hstack([
        mean_features,
        std_features,
        rms_features,
        line_length,
        zero_crossing_rate,
        mobility,
        complexity,
    ])
    _assert_finite(features, "time-domain features")
    print("Time features shape:", features.shape)
    return features


def band_power(psd, freqs, fmin, fmax):
    """Compute mean power within a frequency band."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(f"No PSD bins found for band {fmin}-{fmax} Hz.")
    return psd[mask].mean()


def extract_band_features(X, sfreq=250):
    """Return log-transformed band-power features for each trial."""
    features = []
    for trial in X:
        trial_features = []
        for channel_index in range(trial.shape[0]):
            freqs, psd = welch(trial[channel_index], fs=sfreq, nperseg=256)
            for fmin, fmax in BANDS.values():
                bp = band_power(psd, freqs, fmin, fmax)
                trial_features.append(np.log(bp + 1e-10))
        features.append(np.array(trial_features))

    features = np.array(features)
    _assert_finite(features, "band-power features")
    print("Band-power features shape:", features.shape)
    return features


def extract_fft_features(X, sfreq=250, n_bins=36, log_transform=True):
    """Return interpolated FFT-magnitude features for each trial."""
    features = []
    for trial in X:
        trial_features = []
        for channel_index in range(trial.shape[0]):
            fft_vals = np.abs(np.fft.rfft(trial[channel_index]))
            freqs = np.fft.rfftfreq(trial.shape[1], d=1 / sfreq)
            mask = (freqs >= 4) & (freqs <= 40)
            fft_band = fft_vals[mask]
            if log_transform:
                fft_band = np.log(fft_band + 1e-10)
            x_old = np.linspace(0, 1, len(fft_band))
            x_new = np.linspace(0, 1, n_bins)
            interpolator = interp1d(x_old, fft_band)
            trial_features.append(interpolator(x_new))
        features.append(np.concatenate(trial_features))

    features = np.array(features)
    _assert_finite(features, "FFT features")
    print("FFT features shape:", features.shape)
    return features


def _assert_finite(features, name):
    """Fail fast if a feature extractor produced invalid values."""
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contain NaN or infinite values.")


def build_feature_matrices(X_phys, X_bci):
    """Build all feature matrices for both datasets."""
    return {
        "phys_A": extract_time_features(X_phys),
        "phys_B": extract_band_features(X_phys),
        "phys_C": extract_fft_features(X_phys),
        "bci_A": extract_time_features(X_bci),
        "bci_B": extract_band_features(X_bci),
        "bci_C": extract_fft_features(X_bci),
    }


def save_feature_matrices(results_dir: Path | None = None):
    """Load preprocessed arrays, build features, and save them to disk."""
    output_dir = results_dir or RESULTS_DIR
    X_phys = np.load(output_dir / PREPROCESSED_ARRAYS["physionet_X"].name)
    X_bci = np.load(output_dir / PREPROCESSED_ARRAYS["bci2a_X"].name)
    y_phys = np.load(output_dir / PREPROCESSED_ARRAYS["physionet_y"].name)
    y_bci = np.load(output_dir / PREPROCESSED_ARRAYS["bci2a_y"].name)

    features = build_feature_matrices(X_phys, X_bci)

    for name, values in features.items():
        np.save(output_dir / FEATURE_ARRAYS[name].name, values)

    print("\nSaved feature files:")
    for name in FEATURE_ARRAYS.values():
        print(output_dir / name.name)

    print("\n=== FEATURE SHAPES ===")
    for name in ["phys_A", "phys_B", "phys_C", "bci_A", "bci_B", "bci_C"]:
        print(f"{name} shape:", features[name].shape)

    print("\n=== MIN / MAX ===")
    for name in ["phys_A", "phys_B", "phys_C", "bci_A", "bci_B", "bci_C"]:
        print(f"{name} min/max:", features[name].min(), features[name].max())

    print("\n=== NaN CHECK ===")
    for name in ["phys_A", "phys_B", "phys_C"]:
        print(f"{name} has NaN:", np.isnan(features[name]).any())

    print("\n=== LABEL DISTRIBUTION ===")
    print("PhysioNet unique labels and counts:", np.unique(y_phys, return_counts=True))
    print("BCI2a unique labels and counts:", np.unique(y_bci, return_counts=True))

    print("\n=== RAW DATA SHAPES ===")
    print("X_phys shape:", X_phys.shape)
    print("X_bci shape:", X_bci.shape)

    print("\n=== DESIGN SUMMARY ===")
    print("band_power returns scalar: YES")
    print("FFT log transform: YES")


def main():
    """CLI entrypoint for feature extraction."""
    save_feature_matrices()


if __name__ == "__main__":
    main()
