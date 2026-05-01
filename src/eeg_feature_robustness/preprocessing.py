"""Preprocessing pipeline for the two motor-imagery datasets."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_001, PhysionetMI

from .config import PREPROCESSED_ARRAYS, RESULTS_DIR, TRIAL_METADATA_FILES, configure_data_cache

mne.set_log_level("WARNING")
configure_data_cache()

BCI2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

LEFT_RIGHT_LABELS = {"left_hand": 0, "right_hand": 1}

# MOABB's PhysionetMI class exposes the three left/right imagery runs as
# "0", "1", and "2". Keeping this as an explicit constant makes the trial
# provenance auditable in the saved metadata files.
PHYSIONET_LEFT_RIGHT_RUNS = {"0", "1", "2"}


def preprocess_raw(raw):
    """Apply channel selection, referencing, resampling, and filtering."""
    processed = raw.copy()
    processed.pick_channels(BCI2A_CHANNELS)
    processed.resample(250, npad="auto")
    processed.filter(4.0, 40.0, verbose=False)
    processed.set_eeg_reference("average", projection=False, verbose=False)
    return processed


def extract_epochs(raw):
    """Extract left/right motor-imagery epochs from a preprocessed recording."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {name: code for name, code in event_id.items() if name in LEFT_RIGHT_LABELS}

    if len(keep) < 2:
        return None, None, None

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
        return None, None, None

    X = epochs.get_data()
    y_raw = epochs.events[:, 2]
    code_to_name = {code: name for name, code in keep.items()}
    event_names = np.array([code_to_name[int(code)] for code in y_raw])
    y = np.array([LEFT_RIGHT_LABELS[name] for name in event_names], dtype=int)
    return X, y, event_names


def build_trial_metadata(dataset_name, subject, session_name, run_name, y, event_names):
    """Return one metadata row per extracted trial."""
    return pd.DataFrame({
        "dataset": dataset_name,
        "subject": int(subject),
        "session": str(session_name),
        "run": str(run_name),
        "trial_in_run": np.arange(len(y), dtype=int),
        "event_name": event_names,
        "label": y.astype(int),
    })


def process_physionet(subjects=None):
    """Preprocess the PhysioNet motor-imagery dataset."""
    dataset = PhysionetMI()
    subject_list = dataset.subject_list if subjects is None else subjects
    all_X = []
    all_y = []
    all_metadata = []

    for subject in subject_list:
        try:
            sessions = dataset.get_data(subjects=[subject])
            for session_name, session in sessions[subject].items():
                for run_name, raw in session.items():
                    if run_name not in PHYSIONET_LEFT_RIGHT_RUNS:
                        continue
                    processed = preprocess_raw(raw)
                    X, y, event_names = extract_epochs(processed)
                    if X is None:
                        continue
                    all_X.append(X)
                    all_y.append(y)
                    all_metadata.append(
                        build_trial_metadata("PhysioNetMI", subject, session_name, run_name, y, event_names)
                    )
        except Exception as exc:
            print(f"PhysioNet subject {subject} failed: {exc}")

    if not all_X:
        raise RuntimeError("No PhysioNet trials were extracted.")

    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    metadata = pd.concat(all_metadata, ignore_index=True)
    metadata.insert(0, "trial_index", np.arange(len(metadata), dtype=int))

    print("PhysioNet full shape:", X_full.shape)
    print("PhysioNet labels:", np.unique(y_full, return_counts=True))
    print("PhysioNet subjects:", metadata["subject"].nunique())
    return X_full, y_full, metadata


def process_bci2a(subjects=None):
    """Preprocess the BCI Competition IV Dataset 2a recordings."""
    dataset = BNCI2014_001()
    subject_list = dataset.subject_list if subjects is None else subjects
    all_X = []
    all_y = []
    all_metadata = []

    for subject in subject_list:
        try:
            sessions = dataset.get_data(subjects=[subject])
            for session_name, session in sessions[subject].items():
                for run_name, raw in session.items():
                    processed = preprocess_raw(raw)
                    X, y, event_names = extract_epochs(processed)
                    if X is None:
                        continue
                    all_X.append(X)
                    all_y.append(y)
                    all_metadata.append(
                        build_trial_metadata("BNCI2014_001", subject, session_name, run_name, y, event_names)
                    )
        except Exception as exc:
            print(f"BCI2a subject {subject} failed: {exc}")

    if not all_X:
        raise RuntimeError("No BCI2a trials were extracted.")

    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    metadata = pd.concat(all_metadata, ignore_index=True)
    metadata.insert(0, "trial_index", np.arange(len(metadata), dtype=int))

    print("BCI2a full shape:", X_full.shape)
    print("BCI2a labels:", np.unique(y_full, return_counts=True))
    print("BCI2a subjects:", metadata["subject"].nunique())
    return X_full, y_full, metadata


def _resolve_metadata_path(results_dir: Path, dataset: str) -> Path:
    """Return the trial-metadata CSV path under ``results_dir``.

    The published layout stores trial CSVs in ``results/metadata/``. This helper
    preserves that subfolder when callers redirect ``results_dir`` (e.g. tests
    using a temp directory).
    """
    relative = TRIAL_METADATA_FILES[dataset].relative_to(RESULTS_DIR)
    target = results_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_preprocessed_arrays(output_dir: Path | None = None):
    """Run preprocessing and save the resulting trial arrays."""
    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    X_phys, y_phys, phys_metadata = process_physionet()
    X_bci, y_bci, bci_metadata = process_bci2a()

    np.save(results_dir / PREPROCESSED_ARRAYS["physionet_X"].name, X_phys)
    np.save(results_dir / PREPROCESSED_ARRAYS["physionet_y"].name, y_phys)
    np.save(results_dir / PREPROCESSED_ARRAYS["bci2a_X"].name, X_bci)
    np.save(results_dir / PREPROCESSED_ARRAYS["bci2a_y"].name, y_bci)

    phys_metadata_path = _resolve_metadata_path(results_dir, "physionet")
    bci_metadata_path = _resolve_metadata_path(results_dir, "bci2a")
    phys_metadata.to_csv(phys_metadata_path, index=False)
    bci_metadata.to_csv(bci_metadata_path, index=False)

    print("\nSaved:")
    print(results_dir / PREPROCESSED_ARRAYS["physionet_X"].name)
    print(results_dir / PREPROCESSED_ARRAYS["physionet_y"].name)
    print(results_dir / PREPROCESSED_ARRAYS["bci2a_X"].name)
    print(results_dir / PREPROCESSED_ARRAYS["bci2a_y"].name)
    print(phys_metadata_path)
    print(bci_metadata_path)


def main():
    """CLI entrypoint for preprocessing."""
    save_preprocessed_arrays()


if __name__ == "__main__":
    main()
