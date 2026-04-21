import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d


def extract_time_features(X):
    """
    X shape: (n_trials, n_channels, n_timepoints)

    Returns:
        features shape: (n_trials, 44)
    """
    mean_features = X.mean(axis=2)   # (n_trials, 22)
    var_features = X.var(axis=2)     # (n_trials, 22)

    features = np.hstack([mean_features, var_features])

    print("Time features shape:", features.shape)
    return features


BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}


def band_power(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, mask].mean(axis=1)


def extract_band_features(X, sfreq=250):
    """
    X shape: (n_trials, n_channels, n_timepoints)

    Returns:
        features shape: (n_trials, 88)
    """
    n_trials, n_channels, n_times = X.shape
    features = []

    for trial in X:
        trial_features = []

        for ch in range(n_channels):
            freqs, psd = welch(trial[ch], fs=sfreq, nperseg=256)

            for band_name, (fmin, fmax) in BANDS.items():
                bp = band_power(psd.reshape(1, -1), freqs, fmin, fmax)
                trial_features.append(np.log(bp + 1e-10))

        features.append(np.concatenate(trial_features))

    features = np.array(features)

    print("Band-power features shape:", features.shape)
    return features


def extract_fft_features(X, sfreq=250, n_bins=36):
    """
    X shape: (n_trials, n_channels, n_timepoints)

    Returns:
        features shape: (n_trials, n_channels * n_bins)
    """
    n_trials, n_channels, n_times = X.shape
    features = []

    for trial in X:
        trial_features = []

        for ch in range(n_channels):
            fft_vals = np.abs(np.fft.rfft(trial[ch]))
            freqs = np.fft.rfftfreq(n_times, d=1 / sfreq)

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

    # Feature Set A
    phys_A = extract_time_features(X_phys)
    bci_A = extract_time_features(X_bci)

    print("\nPhysioNet time features:", phys_A.shape)
    print("BCI2a time features:", bci_A.shape)

    print("\nPhysioNet time min/max:", phys_A.min(), phys_A.max())
    print("BCI2a time min/max:", bci_A.min(), bci_A.max())
    print("PhysioNet time has NaN:", np.isnan(phys_A).any())
    print("BCI2a time has NaN:", np.isnan(bci_A).any())

    # Feature Set B
    phys_B = extract_band_features(X_phys)
    bci_B = extract_band_features(X_bci)

    print("\nPhysioNet band-power features:", phys_B.shape)
    print("BCI2a band-power features:", bci_B.shape)

    print("\nPhysioNet band-power min/max:", phys_B.min(), phys_B.max())
    print("BCI2a band-power min/max:", bci_B.min(), bci_B.max())
    print("PhysioNet band-power has NaN:", np.isnan(phys_B).any())
    print("BCI2a band-power has NaN:", np.isnan(bci_B).any())

    # Feature Set C
    phys_C = extract_fft_features(X_phys)
    bci_C = extract_fft_features(X_bci)

    print("\nPhysioNet FFT features:", phys_C.shape)
    print("BCI2a FFT features:", bci_C.shape)

    print("\nPhysioNet FFT min/max:", phys_C.min(), phys_C.max())
    print("BCI2a FFT min/max:", bci_C.min(), bci_C.max())
    print("PhysioNet FFT has NaN:", np.isnan(phys_C).any())
    print("BCI2a FFT has NaN:", np.isnan(bci_C).any())

    # Save all feature matrices
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