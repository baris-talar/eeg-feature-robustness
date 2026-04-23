import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

SEED = 42
np.random.seed(SEED)


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────

def make_models():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=SEED))
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED))
        ])
    }


# ──────────────────────────────────────────────
# WITHIN-DATASET EVALUATION (Phase 4)
# 5-fold cross-validation on a single dataset
# ──────────────────────────────────────────────

def evaluate_within(X, y, n_splits=5):
    """
    Train and test on the same dataset using 5-fold cross-validation.
    Returns mean, std, and raw fold scores for each model.
    Raw scores are needed later for statistical significance testing.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    models = make_models()
    results = {}

    for model_name, model in models.items():
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring="balanced_accuracy"
        )
        results[model_name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist()   # raw fold scores for significance testing
        }
        print(f"  {model_name}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


# ──────────────────────────────────────────────
# CROSS-DATASET EVALUATION (Phase 5)
# Train on one dataset, test on another — zero-shot transfer
# ──────────────────────────────────────────────

def evaluate_cross(X_train, y_train, X_test, y_test):
    """
    Train on full X_train, test on full X_test.
    No cross-validation — this is zero-shot transfer.
    Returns balanced accuracy for each model.
    """
    models = make_models()
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = balanced_accuracy_score(y_test, y_pred)
        results[model_name] = {
            "balanced_accuracy": float(score)
        }
        print(f"  {model_name}: {score:.4f}")

    return results


# ──────────────────────────────────────────────
# GENERALISATION GAP
# Within-dataset accuracy minus cross-dataset accuracy
# ──────────────────────────────────────────────

def compute_gap(within_score, cross_score):
    """
    Generalisation gap = within - cross.
    A small gap means the feature set is robust.
    A large gap means it collapses under distribution shift.
    """
    gap = within_score - cross_score
    normalised_gap = gap / within_score if within_score > 0 else 0.0
    return {
        "gap": round(gap, 4),
        "normalised_gap": round(normalised_gap, 4)
    }


# ──────────────────────────────────────────────
# MAIN — runs all experiments
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # Load feature matrices
    phys_A = np.load("results/phys_A.npy")
    phys_B = np.load("results/phys_B.npy")
    phys_C = np.load("results/phys_C.npy")

    bci_A = np.load("results/bci_A.npy")
    bci_B = np.load("results/bci_B.npy")
    bci_C = np.load("results/bci_C.npy")

    y_phys = np.load("results/physionet_y.npy")
    y_bci  = np.load("results/bci2a_y.npy")

    feature_sets = {
        "TimeDomain": (phys_A, bci_A),
        "BandPower":  (phys_B, bci_B),
        "FFT":        (phys_C, bci_C),
    }

    all_results = {
        "within_physionet": {},
        "within_bci2a": {},
        "cross_phys_to_bci": {},
        "cross_bci_to_phys": {},
        "gaps": {}
    }

    for feat_name, (X_phys, X_bci) in feature_sets.items():
        print(f"\n{'='*50}")
        print(f"FEATURE SET: {feat_name}")
        print(f"{'='*50}")

        # Experiment 1 — within PhysioNet
        print(f"\n[Exp 1] Within PhysioNet ({feat_name})")
        all_results["within_physionet"][feat_name] = evaluate_within(X_phys, y_phys)

        # Experiment 2 — within BCI 2a
        print(f"\n[Exp 2] Within BCI 2a ({feat_name})")
        all_results["within_bci2a"][feat_name] = evaluate_within(X_bci, y_bci)

        # Experiment 3 — train PhysioNet, test BCI 2a
        print(f"\n[Exp 3] Train PhysioNet → Test BCI 2a ({feat_name})")
        all_results["cross_phys_to_bci"][feat_name] = evaluate_cross(
            X_phys, y_phys, X_bci, y_bci
        )

        # Experiment 4 — train BCI 2a, test PhysioNet
        print(f"\n[Exp 4] Train BCI 2a → Test PhysioNet ({feat_name})")
        all_results["cross_bci_to_phys"][feat_name] = evaluate_cross(
            X_bci, y_bci, X_phys, y_phys
        )

        # Compute generalisation gaps
        all_results["gaps"][feat_name] = {}
        for model_name in ["LogReg", "SVM", "RandomForest"]:
            within_phys = all_results["within_physionet"][feat_name][model_name]["mean"]
            within_bci  = all_results["within_bci2a"][feat_name][model_name]["mean"]
            cross_p2b   = all_results["cross_phys_to_bci"][feat_name][model_name]["balanced_accuracy"]
            cross_b2p   = all_results["cross_bci_to_phys"][feat_name][model_name]["balanced_accuracy"]

            all_results["gaps"][feat_name][model_name] = {
                "phys_to_bci": compute_gap(within_phys, cross_p2b),
                "bci_to_phys": compute_gap(within_bci,  cross_b2p)
            }

    # Save all results to JSON
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nAll results saved to results/all_results.json")

    # Print summary table
    print("\n\n" + "="*60)
    print("GENERALISATION GAP SUMMARY")
    print("="*60)
    print(f"{'Feature':<12} {'Model':<14} {'Phys→BCI gap':>12} {'BCI→Phys gap':>13}")
    print("-"*60)
    for feat_name in feature_sets:
        for model_name in ["LogReg", "SVM", "RandomForest"]:
            g1 = all_results["gaps"][feat_name][model_name]["phys_to_bci"]["normalised_gap"]
            g2 = all_results["gaps"][feat_name][model_name]["bci_to_phys"]["normalised_gap"]
            print(f"{feat_name:<12} {model_name:<14} {g1:>12.4f} {g2:>13.4f}")
