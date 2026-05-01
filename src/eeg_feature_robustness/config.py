"""Project-wide configuration: random seed, label maps, and filesystem paths.

This module is the single source of truth for shared constants. Every other
module in the package imports from here. Keep it dependency-free apart from
the standard library so it can be imported during environment bootstrap
(e.g. before MNE or sklearn are available).
"""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

FEATURE_NAMES = ["TimeDomain", "BandPower", "FFT"]
FEATURE_LABELS = {
    "TimeDomain": "Time Domain",
    "BandPower": "Band Power",
    "FFT": "FFT",
}
FEATURE_COLORS = {
    "TimeDomain": "#e74c3c",
    "BandPower": "#2ecc71",
    "FFT": "#3498db",
}

MODEL_NAMES = ["LogReg", "SVM", "RandomForest"]
MODEL_LABELS = {
    "LogReg": "Log. Reg.",
    "SVM": "SVM",
    "RandomForest": "Rand. Forest",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"

# Subfolders for the publication-facing layout. Modules that write derived
# outputs target these directories so the flat `results/` root only contains
# JSON sources of truth.
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_METADATA_DIR = RESULTS_DIR / "metadata"

# Default MNE/MOABB cache; overridable via EEGFR_MNE_DATA_ROOT.
LOCAL_MNE_DATA_DIR = Path.home() / "mne_data"

PREPROCESSED_ARRAYS = {
    "physionet_X": RESULTS_DIR / "physionet_X.npy",
    "physionet_y": RESULTS_DIR / "physionet_y.npy",
    "bci2a_X": RESULTS_DIR / "bci2a_X.npy",
    "bci2a_y": RESULTS_DIR / "bci2a_y.npy",
}

TRIAL_METADATA_FILES = {
    "physionet": RESULTS_METADATA_DIR / "physionet_trial_metadata.csv",
    "bci2a": RESULTS_METADATA_DIR / "bci2a_trial_metadata.csv",
}

FEATURE_ARRAYS = {
    "phys_A": RESULTS_DIR / "phys_A.npy",
    "phys_B": RESULTS_DIR / "phys_B.npy",
    "phys_C": RESULTS_DIR / "phys_C.npy",
    "bci_A": RESULTS_DIR / "bci_A.npy",
    "bci_B": RESULTS_DIR / "bci_B.npy",
    "bci_C": RESULTS_DIR / "bci_C.npy",
}

# Source-of-truth JSONs live at the results/ root; tables, figures, and trial
# metadata live in their dedicated subfolders so the publication-facing layout
# is easy to scan.
TRACKED_RESULT_PATHS = {
    "physionet_trial_metadata": RESULTS_METADATA_DIR / "physionet_trial_metadata.csv",
    "bci2a_trial_metadata": RESULTS_METADATA_DIR / "bci2a_trial_metadata.csv",
    "main_experiment_results": RESULTS_DIR / "main_experiment_results.json",
    "statistical_summary": RESULTS_DIR / "statistical_summary.json",
    "within_dataset_results": RESULTS_TABLES_DIR / "within_dataset_results.csv",
    "cross_dataset_results": RESULTS_TABLES_DIR / "cross_dataset_results.csv",
    "within_dataset_accuracy_heatmap": RESULTS_FIGURES_DIR / "within_dataset_accuracy_heatmap.png",
    "generalisation_gap_by_direction": RESULTS_FIGURES_DIR / "generalisation_gap_by_direction.png",
    "within_vs_cross_dataset_accuracy": RESULTS_FIGURES_DIR / "within_vs_cross_dataset_accuracy.png",
    "bci2a_per_subject_results": RESULTS_DIR / "bci2a_per_subject_results.json",
    "bci2a_per_subject_accuracy": RESULTS_FIGURES_DIR / "bci2a_per_subject_accuracy.png",
    "physionet_subject_results": RESULTS_DIR / "physionet_subject_results.json",
    "combined_per_subject_accuracy": RESULTS_FIGURES_DIR / "combined_per_subject_accuracy.png",
}


def configure_data_cache():
    """Route MOABB/MNE dataset access to a local cache.

    Defaults to ``~/mne_data`` so reproduction does not attempt network
    downloads when a user has the public datasets cached there. Set
    ``EEGFR_MNE_DATA_ROOT`` to point at a different cache root.
    """
    configured_root = os.environ.get("EEGFR_MNE_DATA_ROOT")
    if configured_root:
        cache_root = Path(configured_root).expanduser()
    elif LOCAL_MNE_DATA_DIR.exists():
        cache_root = LOCAL_MNE_DATA_DIR
    else:
        cache_root = EXTERNAL_DATA_DIR

    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(cache_root))
    os.environ.setdefault("MNE_DATASETS_EEGBCI_PATH", str(cache_root))
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", str(cache_root))
    return cache_root
