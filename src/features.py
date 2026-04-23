import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d


def extract_time_features(X):
    """
    X shape: (n_trials, n_channels, n_timepoints)

    Returns:
        features shape: (n_trials, 44)
    """
    mean_features = X.mean(axis=2)
    var_features = X.var(axis=2)

    features = np.hstack([mean_features, var_features])

    print("Time features shape:", features.shape)
    return features


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


def band_power(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[mask].mean()


def extract_band_features(X, sfreq=250):
    """
    X shape: (n_trials, n_channels, n_timepoints)

    Returns:
        features shape: (n_trials, 88)
    """
    features = []

    for trial in X:
        trial_features = []

        for ch in range(trial.shape[0]):
            freqs, psd = welch(trial[ch], fs=sfreq, nperseg=256)

            for _, (fmin, fmax) in BANDS.items():
                bp = band_power(psd, freqs, fmin, fmax)
                trial_features.append(np.log(bp + 1e-10))

        features.append(np.array(trial_features))

    features = np.array(features)

    print("Band-power features shape:", features.shape)
    return features


def extract_fft_features(X, sfreq=250, n_bins=36):
    """
    FFT design choice:
    Keep FFT as raw magnitude (no log transform).
    Scaling will be handled later by StandardScaler in Phase 4.
    """
    features = []

    for trial in X:
        trial_features = []

        for ch in range(trial.shape[0]):
            fft_vals = np.abs(np.fft.rfft(trial[ch]))
            freqs = np.fft.rfftfreq(trial.shape[1], d=1 / sfreq)

            mask = (freqs >= 4) & (freqs <= 40)
            fft_band = fft_vals[mask]

            x_old = np.linspace(0, 1, len(fft_band))
            x_new = np.linspace(0, 1, n_bins)

            f = interp1d(x_old, fft_band)
            trial_features.append(f(x_new))

        features.append(np.concatenate(trial_features))

    features = np.array(features)

    print("FFT features shape:", features.shape)
    return features


if __name__ == "__main__":
    X_phys = np.load("results/physionet_X.npy")
    X_bci = np.load("results/bci2a_X.npy")

    y_phys = np.load("results/physionet_y.npy")
    y_bci = np.load("results/bci2a_y.npy")

    # Feature Set A
    phys_A = extract_time_features(X_phys)
    bci_A = extract_time_features(X_bci)

    print("\nPhysioNet time features:", phys_A.shape)
    print("BCI2a time features:", bci_A.shape)

    print("\nPhysioNet time min/max:", phys_A.min(), phys_A.max())
    print("BCI2a time min/max:", bci_A.min(), bci_A.max())
    print("PhysioNet time has NaN:", np.isnan(phys_A).any())
    print("BCI2a time has NaN:", np.isnan(bci_A).any())
    print("PhysioNet time has Inf:", np.isinf(phys_A).any())
    print("BCI2a time has Inf:", np.isinf(bci_A).any())

    # Feature Set B
    phys_B = extract_band_features(X_phys)
    bci_B = extract_band_features(X_bci)

    print("\nPhysioNet band-power features:", phys_B.shape)
    print("BCI2a band-power features:", bci_B.shape)

    print("\nPhysioNet band-power min/max:", phys_B.min(), phys_B.max())
    print("BCI2a band-power min/max:", bci_B.min(), bci_B.max())
    print("PhysioNet band-power has NaN:", np.isnan(phys_B).any())
    print("BCI2a band-power has NaN:", np.isnan(bci_B).any())
    print("PhysioNet band-power has Inf:", np.isinf(phys_B).any())
    print("BCI2a band-power has Inf:", np.isinf(bci_B).any())

    # Feature Set C
    phys_C = extract_fft_features(X_phys)
    bci_C = extract_fft_features(X_bci)

    print("\nPhysioNet FFT features:", phys_C.shape)
    print("BCI2a FFT features:", bci_C.shape)

    print("\nPhysioNet FFT min/max:", phys_C.min(), phys_C.max())
    print("BCI2a FFT min/max:", bci_C.min(), bci_C.max())
    print("PhysioNet FFT has NaN:", np.isnan(phys_C).any())
    print("BCI2a FFT has NaN:", np.isnan(bci_C).any())
    print("PhysioNet FFT has Inf:", np.isinf(phys_C).any())
    print("BCI2a FFT has Inf:", np.isinf(bci_C).any())

    np.save("results/phys_A.npy", phys_A)
    np.save("results/phys_B.npy", phys_B)
    np.save("results/phys_C.npy", phys_C)

    np.save("results/bci_A.npy", bci_A)
    np.save("results/bci_B.npy", bci_B)
    np.save("results/bci_C.npy", bci_C)

    print("\nSaved feature files:")
    print("results/phys_A.npy")
    print("results/phys_B.npy")
    print("results/phys_C.npy")
    print("results/bci_A.npy")
    print("results/bci_B.npy")
    print("results/bci_C.npy")

    print("\n=== FEATURE SHAPES ===")
    print("phys_A shape:", phys_A.shape)
    print("phys_B shape:", phys_B.shape)
    print("phys_C shape:", phys_C.shape)
    print("bci_A shape:", bci_A.shape)
    print("bci_B shape:", bci_B.shape)
    print("bci_C shape:", bci_C.shape)

    print("\n=== MIN / MAX ===")
    print("phys_A min/max:", phys_A.min(), phys_A.max())
    print("phys_B min/max:", phys_B.min(), phys_B.max())
    print("phys_C min/max:", phys_C.min(), phys_C.max())
    print("bci_A min/max:", bci_A.min(), bci_A.max())
    print("bci_B min/max:", bci_B.min(), bci_B.max())
    print("bci_C min/max:", bci_C.min(), bci_C.max())

    print("\n=== NaN CHECK ===")
    print("phys_A has NaN:", np.isnan(phys_A).any())
    print("phys_B has NaN:", np.isnan(phys_B).any())
    print("phys_C has NaN:", np.isnan(phys_C).any())

    print("\n=== LABEL DISTRIBUTION ===")
    print("PhysioNet unique labels and counts:", np.unique(y_phys, return_counts=True))
    print("BCI2a unique labels and counts:", np.unique(y_bci, return_counts=True))

    print("\n=== RAW DATA SHAPES ===")
    print("X_phys shape:", X_phys.shape)
    print("X_bci shape:", X_bci.shape)

    print("\n=== DESIGN CHOICES ===")
    print("band_power returns scalar: YES")
    print("FFT log transform: NO (intentional)")