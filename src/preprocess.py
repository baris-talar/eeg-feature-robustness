import os
import mne
import numpy as np
from moabb.datasets import PhysionetMI, BNCI2014_001

mne.set_log_level("WARNING")

BCI2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

PHYSIONET_LEFT_RIGHT_RUNS = {"0", "1", "2"}


def preprocess_raw(raw):
    raw = raw.copy()
    raw.pick_channels(BCI2A_CHANNELS)
    raw.resample(250, npad="auto")
    raw.filter(4.0, 40.0, verbose=False)
    return raw


def extract_epochs(raw):
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    keep = {k: v for k, v in event_id.items() if k in {"left_hand", "right_hand"}}

    if len(keep) < 2:
        return None, None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=keep,
        tmin=0.5,
        tmax=2.5,
        baseline=None,
        preload=True,
        verbose=False,
    )

    if len(epochs) == 0:
        return None, None

    X = epochs.get_data()
    y_raw = epochs.events[:, 2]
    y = np.where(y_raw == keep["left_hand"], 0, 1)

    return X, y


def process_physionet(subjects=None):
    dataset = PhysionetMI()
    subject_list = dataset.subject_list if subjects is None else subjects

    all_X = []
    all_y = []

    for subject in subject_list:
        try:
            sessions = dataset.get_data(subjects=[subject])

            for _, session in sessions[subject].items():
                for run_name, raw in session.items():
                    if run_name not in PHYSIONET_LEFT_RIGHT_RUNS:
                        continue

                    raw = preprocess_raw(raw)
                    X, y = extract_epochs(raw)

                    if X is None:
                        continue

                    all_X.append(X)
                    all_y.append(y)

        except Exception as e:
            print(f"PhysioNet subject {subject} failed: {e}")

    if not all_X:
        raise RuntimeError("No PhysioNet trials were extracted.")

    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)

    print("PhysioNet full shape:", X_full.shape)
    print("PhysioNet labels:", np.unique(y_full, return_counts=True))

    return X_full, y_full


def process_bci2a(subjects=None):
    dataset = BNCI2014_001()
    subject_list = dataset.subject_list if subjects is None else subjects

    all_X = []
    all_y = []

    for subject in subject_list:
        try:
            sessions = dataset.get_data(subjects=[subject])

            for _, session in sessions[subject].items():
                for _, raw in session.items():
                    raw = preprocess_raw(raw)
                    X, y = extract_epochs(raw)

                    if X is None:
                        continue

                    all_X.append(X)
                    all_y.append(y)

        except Exception as e:
            print(f"BCI2a subject {subject} failed: {e}")

    if not all_X:
        raise RuntimeError("No BCI2a trials were extracted.")

    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)

    print("BCI2a full shape:", X_full.shape)
    print("BCI2a labels:", np.unique(y_full, return_counts=True))

    return X_full, y_full


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    X_phys, y_phys = process_physionet()
    X_bci, y_bci = process_bci2a()

    np.save("results/physionet_X.npy", X_phys)
    np.save("results/physionet_y.npy", y_phys)
    np.save("results/bci2a_X.npy", X_bci)
    np.save("results/bci2a_y.npy", y_bci)

    print("\nSaved:")
    print("results/physionet_X.npy")
    print("results/physionet_y.npy")
    print("results/bci2a_X.npy")
    print("results/bci2a_y.npy")