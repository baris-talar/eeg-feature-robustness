import mne
import numpy as np
from moabb.datasets import PhysionetMI, BNCI2014_001

BCI2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

SEED = 42
np.random.seed(SEED)


def load_physionet_raw(subject=1):
    dataset = PhysionetMI()
    sessions = dataset.get_data(subjects=[subject])
    raw = list(list(sessions[subject].values())[0].values())[0]
    return raw


def load_bci2a_raw(subject=1):
    dataset = BNCI2014_001()
    sessions = dataset.get_data(subjects=[subject])
    raw = list(list(sessions[subject].values())[0].values())[0]
    return raw


def select_shared_channels(raw):
    raw = raw.copy()
    raw.pick(BCI2A_CHANNELS)
    return raw


def resample_to_250(raw):
    raw = raw.copy()
    raw.resample(250, npad='auto')
    return raw


def bandpass_filter(raw):
    raw = raw.copy()
    raw.filter(4.0, 40.0)
    return raw


def create_epochs(raw, dataset_type="physionet"):
    """
    Cut continuous EEG into labeled trials (epochs).

    Output:
        MNE Epochs object
    """
    events, event_id = mne.events_from_annotations(raw)

    print(f"\nEvent mapping from annotations ({dataset_type}):")
    print(event_id)

    if dataset_type == "physionet":
        # Keep only left/right, remove rest
        event_map = {
            "left_hand": event_id["left_hand"],
            "right_hand": event_id["right_hand"]
        }

    elif dataset_type == "bci2a":
        # Keep only left/right, remove feet/tongue
        event_map = {
            "left_hand": event_id["left_hand"],
            "right_hand": event_id["right_hand"]
        }

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_map,
        tmin=0.5,
        tmax=2.5,
        baseline=None,
        preload=True
    )

    return epochs


def epochs_to_numpy(epochs):
    """
    Convert epochs into NumPy arrays.

    Returns:
        X: shape (n_trials, n_channels, n_timepoints)
        y: shape (n_trials,) with labels mapped to 0/1
           left_hand -> 0
           right_hand -> 1
    """
    X = epochs.get_data()
    y_raw = epochs.events[:, 2]

    # Sort labels so mapping is stable:
    # smaller label -> 0, larger label -> 1
    unique_labels = np.sort(np.unique(y_raw))
    label_map = {
        unique_labels[0]: 0,
        unique_labels[1]: 1
    }
    y = np.array([label_map[label] for label in y_raw])

    print("\nEpoch array shape:", X.shape)
    print("Raw epoch labels:", unique_labels)
    print("Mapped labels/counts:", np.unique(y, return_counts=True))

    return X, y


if __name__ == "__main__":
    # -------------------------
    # 1. Load raw data
    # -------------------------
    phys_raw = load_physionet_raw(1)
    bci_raw = load_bci2a_raw(1)

    # -------------------------
    # 2. Shared channel set
    # -------------------------
    phys_22 = select_shared_channels(phys_raw)
    bci_22 = select_shared_channels(bci_raw)

    # -------------------------
    # 3. Resample to 250 Hz
    # -------------------------
    phys_250 = resample_to_250(phys_22)
    bci_250 = resample_to_250(bci_22)

    print("Before filtering:")
    print("PhysioNet shape:", phys_250.get_data().shape, "sfreq:", phys_250.info["sfreq"])
    print("BCI2a shape:", bci_250.get_data().shape, "sfreq:", bci_250.info["sfreq"])

    # -------------------------
    # 4. Bandpass filter 4–40 Hz
    # -------------------------
    phys_filt = bandpass_filter(phys_250)
    bci_filt = bandpass_filter(bci_250)

    print("\nAfter filtering:")
    print("PhysioNet shape:", phys_filt.get_data().shape, "sfreq:", phys_filt.info["sfreq"])
    print("BCI2a shape:", bci_filt.get_data().shape, "sfreq:", bci_filt.info["sfreq"])

    # -------------------------
    # 5. Extract epochs
    # -------------------------
    phys_epochs = create_epochs(phys_filt, dataset_type="physionet")
    bci_epochs = create_epochs(bci_filt, dataset_type="bci2a")

    print("\nPhysioNet epochs:")
    print(phys_epochs)

    print("\nBCI2a epochs:")
    print(bci_epochs)

    # -------------------------
    # 6. Convert to NumPy
    # -------------------------
    X_phys, y_phys = epochs_to_numpy(phys_epochs)
    X_bci, y_bci = epochs_to_numpy(bci_epochs)

    print("\nFinal outputs:")
    print("PhysioNet X:", X_phys.shape, "y:", y_phys.shape)
    print("BCI2a X:", X_bci.shape, "y:", y_bci.shape)

    # -------------------------
    # 7. Save preprocessed data
    # -------------------------
    np.save("results/physionet_X.npy", X_phys)
    np.save("results/physionet_y.npy", y_phys)
    np.save("results/bci2a_X.npy", X_bci)
    np.save("results/bci2a_y.npy", y_bci)

    print("\nSaved files:")
    print("results/physionet_X.npy")
    print("results/physionet_y.npy")
    print("results/bci2a_X.npy")
    print("results/bci2a_y.npy")