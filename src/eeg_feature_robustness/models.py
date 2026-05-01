"""Model evaluation routines for within- and cross-dataset experiments."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import (
    FEATURE_ARRAYS,
    FEATURE_LABELS,
    FEATURE_NAMES,
    MODEL_NAMES,
    PREPROCESSED_ARRAYS,
    RESULTS_DIR,
    SEED,
    TRACKED_RESULT_PATHS,
    TRIAL_METADATA_FILES,
)

N_BOOTSTRAPS = 1000
N_PERMUTATIONS = 5000
CONFIDENCE_LEVEL = 0.95


def make_models(pca_components: int | None = None):
    """Construct the model family used throughout the benchmark."""
    def reduction_steps():
        if pca_components is None:
            return []
        return [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=SEED)),
        ]

    def linear_steps(clf):
        steps = reduction_steps()
        if steps:
            return steps + [("clf", clf)]
        return [("scaler", StandardScaler()), ("clf", clf)]

    return {
        "LogReg": Pipeline(linear_steps(
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)
        )),
        "SVM": Pipeline(linear_steps(
            LinearSVC(class_weight="balanced", dual="auto", max_iter=10000, random_state=SEED)
        )),
        "RandomForest": Pipeline(
            reduction_steps()
            + [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def mean_confidence_interval(values, n_bootstraps=N_BOOTSTRAPS, confidence=CONFIDENCE_LEVEL):
    """Return a non-parametric bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"low": None, "high": None}
    if len(arr) == 1:
        value = round(float(arr[0]), 4)
        return {"low": value, "high": value}

    rng = np.random.default_rng(SEED)
    boot_means = rng.choice(arr, size=(n_bootstraps, len(arr)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": round(float(np.quantile(boot_means, alpha)), 4),
        "high": round(float(np.quantile(boot_means, 1.0 - alpha)), 4),
    }


def summarize_scores(scores):
    """Summarize cross-validation scores with uncertainty."""
    scores = np.asarray(scores, dtype=float)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "ci95": mean_confidence_interval(scores),
        "scores": scores.tolist(),
    }


def compute_gap(within_score, cross_score):
    """Compute the absolute and normalised generalisation gap."""
    gap = within_score - cross_score
    normalised_gap = gap / within_score if within_score > 0 else 0.0
    return {"gap": round(gap, 4), "normalised_gap": round(normalised_gap, 4)}


def _effective_pca_components(X, requested_components):
    if requested_components is None or X.shape[1] <= requested_components:
        return None
    return requested_components


def evaluate_within(X, y, n_splits=5, groups=None, pca_components=None):
    """Evaluate a feature matrix with either trial-wise or grouped CV."""
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cv_groups = None
        protocol = "trialwise_stratified"
    else:
        unique_groups = np.unique(groups)
        resolved_splits = min(n_splits, len(unique_groups))
        cv = StratifiedGroupKFold(n_splits=resolved_splits, shuffle=True, random_state=SEED)
        cv_groups = groups
        protocol = "subject_grouped_stratified"

    models = make_models(_effective_pca_components(X, pca_components))
    results = {}

    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, groups=cv_groups, cv=cv, scoring="balanced_accuracy")
        results[model_name] = {
            **summarize_scores(scores),
            "n_splits": int(len(scores)),
            "protocol": protocol,
        }
        print(f"  {model_name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def subject_level_scores(y_true, y_pred, groups):
    """Compute balanced accuracy separately for each target subject."""
    rows = []
    groups = np.asarray(groups)
    for subject in np.unique(groups):
        mask = groups == subject
        if len(np.unique(y_true[mask])) < 2:
            score = np.nan
        else:
            score = balanced_accuracy_score(y_true[mask], y_pred[mask])
        rows.append({
            "subject": int(subject),
            "n_trials": int(mask.sum()),
            "balanced_accuracy": None if np.isnan(score) else round(float(score), 4),
        })
    return rows


def bootstrap_balanced_accuracy_by_group(y_true, y_pred, groups, n_bootstraps=N_BOOTSTRAPS):
    """Bootstrap cross-dataset balanced accuracy by resampling subjects."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    scores = []

    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        if len(np.unique(y_true[sampled_indices])) < 2:
            continue
        scores.append(balanced_accuracy_score(y_true[sampled_indices], y_pred[sampled_indices]))

    if not scores:
        return {"low": None, "high": None}

    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "low": round(float(np.quantile(scores, alpha)), 4),
        "high": round(float(np.quantile(scores, 1.0 - alpha)), 4),
    }


def evaluate_cross(X_train, y_train, X_test, y_test, target_groups=None, pca_components=None):
    """Train on one dataset and evaluate on another without target adaptation."""
    models = make_models(_effective_pca_components(X_train, pca_components))
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = balanced_accuracy_score(y_test, y_pred)
        model_result = {
            "balanced_accuracy": float(score),
            "ci95": {"low": None, "high": None},
        }
        if target_groups is not None:
            model_result["ci95"] = bootstrap_balanced_accuracy_by_group(y_test, y_pred, target_groups)
            model_result["subject_scores"] = subject_level_scores(y_test, y_pred, target_groups)
        results[model_name] = model_result
        print(f"  {model_name}: {score:.4f}")

    return results


def _resolve_metadata_path(output_dir: Path, dataset: str) -> Path:
    """Return the trial-metadata CSV path under ``output_dir``.

    Mirrors :func:`preprocessing._resolve_metadata_path` so callers that point
    ``output_dir`` at a temp directory still find the metadata under the same
    relative subfolder layout used in published results.
    """
    relative = TRIAL_METADATA_FILES[dataset].relative_to(RESULTS_DIR)
    return output_dir / relative


def load_subject_groups(output_dir: Path):
    """Load subject identifiers aligned to the saved preprocessed arrays."""
    phys_metadata_path = _resolve_metadata_path(output_dir, "physionet")
    bci_metadata_path = _resolve_metadata_path(output_dir, "bci2a")
    if not phys_metadata_path.exists() or not bci_metadata_path.exists():
        raise FileNotFoundError(
            "Trial metadata files are required for subject-aware evaluation. "
            "Rerun preprocessing to create physionet_trial_metadata.csv and bci2a_trial_metadata.csv."
        )

    phys_metadata = pd.read_csv(phys_metadata_path).sort_values("trial_index")
    bci_metadata = pd.read_csv(bci_metadata_path).sort_values("trial_index")
    return phys_metadata["subject"].to_numpy(), bci_metadata["subject"].to_numpy()


def paired_permutation_pvalue(a, b, n_permutations=N_PERMUTATIONS):
    """Two-sided paired sign-flip permutation test for mean differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    if len(diff) == 0:
        return None
    observed = abs(diff.mean())
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1, 1], size=(n_permutations, len(diff)))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_permutations + 1))


def collect_cross_subject_scores(protocol_results, feature_name):
    """Collect target-subject scores across both transfer directions and models."""
    scores = []
    for direction in ["cross_phys_to_bci", "cross_bci_to_phys"]:
        for model_name in MODEL_NAMES:
            for row in protocol_results[direction][feature_name][model_name].get("subject_scores", []):
                value = row["balanced_accuracy"]
                if value is not None:
                    scores.append(value)
    return scores


def build_statistical_summary(protocol_results):
    """Compare feature sets using paired subject-level cross-dataset scores."""
    feature_scores = {
        feature: collect_cross_subject_scores(protocol_results, feature)
        for feature in FEATURE_NAMES
    }
    pairwise = []
    for left, right in combinations(FEATURE_NAMES, 2):
        left_scores = feature_scores[left]
        right_scores = feature_scores[right]
        n = min(len(left_scores), len(right_scores))
        pairwise.append({
            "feature_a": left,
            "feature_b": right,
            "mean_a": round(float(np.mean(left_scores[:n])), 4) if n else None,
            "mean_b": round(float(np.mean(right_scores[:n])), 4) if n else None,
            "mean_difference_a_minus_b": round(float(np.mean(np.asarray(left_scores[:n]) - np.asarray(right_scores[:n]))), 4) if n else None,
            "paired_permutation_p": paired_permutation_pvalue(left_scores[:n], right_scores[:n]) if n else None,
            "n_paired_subject_model_direction_scores": int(n),
        })
    return {
        "test": "paired sign-flip permutation on target-subject balanced accuracies",
        "n_permutations": N_PERMUTATIONS,
        "feature_pairwise_tests": pairwise,
        "interpretation_guardrail": (
            "Feature rankings should be treated as descriptive unless pairwise intervals "
            "and permutation tests support a stable difference."
        ),
    }


def run_protocol_grid(feature_sets, y_phys, y_bci, groups_phys, groups_bci, pca_components=None, include_trialwise=True):
    """Run pooled, grouped, and cross-dataset evaluations for a feature grid."""
    protocol_results = {
        "within_physionet_trialwise": {},
        "within_bci2a_trialwise": {},
        "within_physionet_subject_grouped": {},
        "within_bci2a_subject_grouped": {},
        "cross_phys_to_bci": {},
        "cross_bci_to_phys": {},
        "gaps": {},
    }

    for feat_name, (X_phys, X_bci) in feature_sets.items():
        print(f"\n{'=' * 50}")
        print(f"FEATURE SET: {feat_name}")
        print(f"{'=' * 50}")

        if include_trialwise:
            print(f"\n[Diagnostic] Within PhysioNet trial-wise CV ({feat_name})")
            protocol_results["within_physionet_trialwise"][feat_name] = evaluate_within(
                X_phys, y_phys, groups=None, pca_components=pca_components
            )

            print(f"\n[Diagnostic] Within BCI2a trial-wise CV ({feat_name})")
            protocol_results["within_bci2a_trialwise"][feat_name] = evaluate_within(
                X_bci, y_bci, groups=None, pca_components=pca_components
            )

        print(f"\n[Primary] Within PhysioNet subject-grouped CV ({feat_name})")
        protocol_results["within_physionet_subject_grouped"][feat_name] = evaluate_within(
            X_phys, y_phys, groups=groups_phys, pca_components=pca_components
        )

        print(f"\n[Primary] Within BCI2a subject-grouped CV ({feat_name})")
        protocol_results["within_bci2a_subject_grouped"][feat_name] = evaluate_within(
            X_bci, y_bci, groups=groups_bci, pca_components=pca_components
        )

        print(f"\n[Primary] Train PhysioNet -> Test BCI2a ({feat_name})")
        protocol_results["cross_phys_to_bci"][feat_name] = evaluate_cross(
            X_phys, y_phys, X_bci, y_bci, target_groups=groups_bci, pca_components=pca_components
        )

        print(f"\n[Primary] Train BCI2a -> Test PhysioNet ({feat_name})")
        protocol_results["cross_bci_to_phys"][feat_name] = evaluate_cross(
            X_bci, y_bci, X_phys, y_phys, target_groups=groups_phys, pca_components=pca_components
        )

        protocol_results["gaps"][feat_name] = {}
        for model_name in MODEL_NAMES:
            within_phys = protocol_results["within_physionet_subject_grouped"][feat_name][model_name]["mean"]
            within_bci = protocol_results["within_bci2a_subject_grouped"][feat_name][model_name]["mean"]
            cross_p2b = protocol_results["cross_phys_to_bci"][feat_name][model_name]["balanced_accuracy"]
            cross_b2p = protocol_results["cross_bci_to_phys"][feat_name][model_name]["balanced_accuracy"]
            protocol_results["gaps"][feat_name][model_name] = {
                "phys_to_bci": compute_gap(within_phys, cross_p2b),
                "bci_to_phys": compute_gap(within_bci, cross_b2p),
            }

    protocol_results["statistical_summary"] = build_statistical_summary(protocol_results)
    return protocol_results


def run_main_experiments(results_dir: Path | None = None):
    """Run the main within- and cross-dataset experiment grid."""
    output_dir = results_dir or RESULTS_DIR

    phys_A = np.load(output_dir / FEATURE_ARRAYS["phys_A"].name)
    phys_B = np.load(output_dir / FEATURE_ARRAYS["phys_B"].name)
    phys_C = np.load(output_dir / FEATURE_ARRAYS["phys_C"].name)
    bci_A = np.load(output_dir / FEATURE_ARRAYS["bci_A"].name)
    bci_B = np.load(output_dir / FEATURE_ARRAYS["bci_B"].name)
    bci_C = np.load(output_dir / FEATURE_ARRAYS["bci_C"].name)
    y_phys = np.load(output_dir / PREPROCESSED_ARRAYS["physionet_y"].name)
    y_bci = np.load(output_dir / PREPROCESSED_ARRAYS["bci2a_y"].name)
    groups_phys, groups_bci = load_subject_groups(output_dir)

    if len(groups_phys) != len(y_phys) or len(groups_bci) != len(y_bci):
        raise ValueError("Trial metadata length does not match the saved label arrays.")

    feature_sets = {
        "TimeDomain": (phys_A, bci_A),
        "BandPower": (phys_B, bci_B),
        "FFT": (phys_C, bci_C),
    }
    feature_dimensions = {
        name: {"physionet": int(values[0].shape[1]), "bci2a": int(values[1].shape[1])}
        for name, values in feature_sets.items()
    }
    matched_components = min(
        dim
        for dims in feature_dimensions.values()
        for dim in dims.values()
    )

    all_results = {
        "schema_version": 3,
        "random_seed": SEED,
        "primary_protocol": "pca_matched.subject_grouped_within_and_zero_shot_cross_dataset",
        "feature_dimensions": feature_dimensions,
        "matched_pca_components": int(matched_components),
        "protocol_notes": [
            "Trial-wise CV is retained only as a diagnostic legacy baseline.",
            "Primary within-dataset estimates use subject-grouped stratified folds.",
            "Primary feature comparison should use pca_matched results to reduce dimensionality confounding.",
            "Cross-dataset uncertainty is bootstrapped over target subjects.",
        ],
        "raw_dimension": run_protocol_grid(
            feature_sets, y_phys, y_bci, groups_phys, groups_bci, pca_components=None
        ),
        "pca_matched": run_protocol_grid(
            feature_sets,
            y_phys,
            y_bci,
            groups_phys,
            groups_bci,
            pca_components=matched_components,
            include_trialwise=False,
        ),
    }

    results_path = output_dir / TRACKED_RESULT_PATHS["main_experiment_results"].name
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    stats_path = output_dir / TRACKED_RESULT_PATHS["statistical_summary"].name
    with stats_path.open("w") as f:
        json.dump(all_results["pca_matched"]["statistical_summary"], f, indent=2)

    print(f"\nAll results saved to {results_path}")
    print(f"Statistical summary saved to {stats_path}")
    return all_results


# ---------------------------------------------------------------------------
# Per-subject analyses
# ---------------------------------------------------------------------------

PER_SUBJECT_N_SPLITS = 5

# BCI2a per-subject pipelines preserve the original supplementary protocol:
# RBF-SVM rather than LinearSVC and unscaled RandomForest. Keeping this here
# so the merged entrypoint produces the exact published numbers.
def _make_bci2a_per_subject_models():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=SEED)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ]),
    }


def _make_physionet_per_subject_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
    ])


def _run_bci2a_per_subject(results_dir: Path):
    """Per-subject BCI2a evaluation across all three classifiers and feature sets."""
    y_full = np.load(results_dir / PREPROCESSED_ARRAYS["bci2a_y"].name)
    metadata = pd.read_csv(TRIAL_METADATA_FILES["bci2a"]).sort_values("trial_index")
    subject_ids = metadata["subject"].to_numpy()

    features_dict = {
        "TimeDomain": np.load(results_dir / FEATURE_ARRAYS["bci_A"].name),
        "BandPower":  np.load(results_dir / FEATURE_ARRAYS["bci_B"].name),
        "FFT":        np.load(results_dir / FEATURE_ARRAYS["bci_C"].name),
    }

    print(f"\nBCI2a per-subject analysis: {len(np.unique(subject_ids))} subjects, "
          f"{len(y_full)} trials")

    cv = StratifiedKFold(n_splits=PER_SUBJECT_N_SPLITS, shuffle=True, random_state=SEED)
    raw = {feat: {model: [] for model in MODEL_NAMES} for feat in FEATURE_NAMES}

    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        y_subj = y_full[mask]
        print(f"  Subject {subj} (n={mask.sum()})")
        for feat_name in FEATURE_NAMES:
            X_subj = features_dict[feat_name][mask]
            for model_name, model in _make_bci2a_per_subject_models().items():
                scores = cross_val_score(model, X_subj, y_subj, cv=cv, scoring="balanced_accuracy")
                raw[feat_name][model_name].append(float(scores.mean()))

    summary = {}
    print("\n" + "=" * 60)
    print("PER-SUBJECT BCI2a SUMMARY (mean \u00b1 std across subjects)")
    print("=" * 60)
    for feat in FEATURE_NAMES:
        summary[feat] = {}
        for model in MODEL_NAMES:
            scores = np.array(raw[feat][model])
            summary[feat][model] = {
                "mean": round(float(scores.mean()), 4),
                "std":  round(float(scores.std()),  4),
                "min":  round(float(scores.min()),  4),
                "max":  round(float(scores.max()),  4),
                "per_subject": [round(float(s), 4) for s in scores],
            }
            print(f"{FEATURE_LABELS[feat]:<14} {model:<14} "
                  f"{scores.mean():.4f} \u00b1 {scores.std():.4f}")

    output_path = results_dir / TRACKED_RESULT_PATHS["bci2a_per_subject_results"].name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_path}")
    return summary


def _run_physionet_per_subject(results_dir: Path):
    """Per-subject PhysioNet evaluation (LogReg) using one MOABB subject at a time."""
    # Imports are deferred so models.py stays importable in environments where
    # MNE/MOABB are unavailable (e.g. CPU CI without raw EEG access).
    import mne  # noqa: WPS433
    from moabb.datasets import PhysionetMI  # noqa: WPS433

    from .config import configure_data_cache  # noqa: WPS433
    from .features import (  # noqa: WPS433
        extract_band_features,
        extract_fft_features,
        extract_time_features,
    )
    from .preprocessing import (  # noqa: WPS433
        PHYSIONET_LEFT_RIGHT_RUNS,
        extract_epochs,
        preprocess_raw,
    )

    mne.set_log_level("WARNING")
    configure_data_cache()

    dataset = PhysionetMI()
    subject_list = dataset.subject_list

    cv = StratifiedKFold(n_splits=PER_SUBJECT_N_SPLITS, shuffle=True, random_state=SEED)
    results = {feat: {} for feat in FEATURE_NAMES}
    skipped = []

    print(f"\nPhysioNet per-subject analysis: {len(subject_list)} subjects")

    for i, subject in enumerate(subject_list, start=1):
        print(f"  [{i:3d}/{len(subject_list)}] subject {subject}", end="  ", flush=True)
        try:
            sessions = dataset.get_data(subjects=[subject])
            all_X, all_y = [], []
            for _, session in sessions[subject].items():
                for run_name, raw in session.items():
                    if run_name not in PHYSIONET_LEFT_RIGHT_RUNS:
                        continue
                    raw = preprocess_raw(raw)
                    X, y, _ = extract_epochs(raw)
                    if X is not None and len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
        except Exception as exc:
            print(f"failed ({exc})")
            skipped.append(subject)
            continue

        if not all_X:
            print("excluded (no left/right trials)")
            skipped.append(subject)
            continue

        X_raw = np.concatenate(all_X)
        y = np.concatenate(all_y)

        if len(X_raw) < 10 or len(np.unique(y)) < 2:
            print(f"excluded (n={len(X_raw)}, classes={len(np.unique(y))})")
            skipped.append(subject)
            continue

        feature_inputs = {
            "TimeDomain": extract_time_features(X_raw),
            "BandPower":  extract_band_features(X_raw),
            "FFT":        extract_fft_features(X_raw),
        }

        scores_str = []
        for feat_name in FEATURE_NAMES:
            min_class = min(np.bincount(y.astype(int)))
            if min_class < PER_SUBJECT_N_SPLITS:
                results[feat_name][subject] = None
                scores_str.append(f"{feat_name[:3]}=n/a")
                continue
            model = _make_physionet_per_subject_logreg()
            cv_scores = cross_val_score(model, feature_inputs[feat_name], y, cv=cv,
                                        scoring="balanced_accuracy")
            results[feat_name][subject] = {
                "mean":     round(float(cv_scores.mean()), 4),
                "std":      round(float(cv_scores.std()),  4),
                "n_trials": int(len(X_raw)),
            }
            scores_str.append(f"{feat_name[:3]}={cv_scores.mean():.3f}")

        print(f"n={len(X_raw):3d}  " + "  ".join(scores_str))

    print(f"\nCompleted. Excluded {len(skipped)} subjects: {skipped}")

    output_path = results_dir / TRACKED_RESULT_PATHS["physionet_subject_results"].name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")
    return results


def run_per_subject_analysis(dataset: str, *, results_dir: Path | None = None):
    """Run within-subject 5-fold CV for the requested dataset.

    Parameters
    ----------
    dataset : {"bci2a", "physionet"}
        ``"bci2a"`` evaluates the saved BCI2a feature arrays across all three
        classifiers (the original supplementary protocol uses RBF-SVM, not
        LinearSVC). ``"physionet"`` re-loads MOABB subjects one at a time so
        the full PhysioNet cohort fits in memory and runs LogReg only.
    """
    output_dir = results_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "bci2a":
        return _run_bci2a_per_subject(output_dir)
    if dataset == "physionet":
        return _run_physionet_per_subject(output_dir)
    raise ValueError(f"Unknown dataset: {dataset!r}. Expected 'bci2a' or 'physionet'.")


def main():
    """CLI entrypoint for the main experiment grid."""
    run_main_experiments()


if __name__ == "__main__":
    main()
