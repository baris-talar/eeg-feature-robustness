"""Orchestration for the reproducible end-to-end pipeline.

This module is the single entry point for regenerating every tracked artifact:
preprocessing, feature extraction, the main experiment grid, both per-subject
analyses, all figures and tables, and a printed evaluation-protocol contrast
that explains the gap between subject-wise best-model reporting and the
pooled trial-wise estimate the paper uses.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import FEATURE_NAMES, MODEL_NAMES, TRACKED_RESULT_PATHS
from .figures import generate_figures_and_tables


def _print_evaluation_protocol_comparison(main_results, bci2a_per_subject):
    """Print a side-by-side comparison of evaluation protocols.

    Compares:
      - subject-wise best-model reporting (per subject, then averaged)
      - pooled trial-wise within-dataset estimates
      - subject-grouped within-dataset estimates (the paper baseline)

    Most published "single subject in the 80s" claims sit in protocol 1.
    Our headline numbers sit in protocol 3. Surfacing the gap explicitly
    keeps the reader from comparing apples to oranges.
    """
    n_subjects = len(bci2a_per_subject[FEATURE_NAMES[0]][MODEL_NAMES[0]]["per_subject"])
    best_per_subject = []
    print("\n" + "=" * 65)
    print("BCI2a PER-SUBJECT ACCURACY \u2014 ALL MODELS")
    print("=" * 65)
    print(f"\n{'Subj':>5}  {'TimeDomain':>12}  {'BandPower':>12}  {'FFT':>12}  {'Best':>8}")
    print("-" * 60)

    for i in range(n_subjects):
        per_feature_best = {
            feat: max(bci2a_per_subject[feat][model]["per_subject"][i] for model in MODEL_NAMES)
            for feat in FEATURE_NAMES
        }
        row_best = max(per_feature_best.values())
        best_per_subject.append(row_best)
        print(
            f"  {i + 1:3d}    "
            f"{per_feature_best['TimeDomain']:>12.4f}  "
            f"{per_feature_best['BandPower']:>12.4f}  "
            f"{per_feature_best['FFT']:>12.4f}  "
            f"{row_best:>8.4f}"
        )

    best_arr = np.array(best_per_subject)

    if main_results.get("schema_version", 1) >= 3:
        trialwise = main_results["raw_dimension"]["within_bci2a_trialwise"]
        grouped = main_results["raw_dimension"]["within_bci2a_subject_grouped"]
    else:
        trialwise = main_results["within_bci2a"]
        grouped = main_results["within_bci2a"]

    pooled_trialwise_mean = np.mean([
        trialwise[feat][model]["mean"]
        for feat in FEATURE_NAMES
        for model in MODEL_NAMES
    ])
    pooled_trialwise_best = max(
        trialwise[feat][model]["mean"]
        for feat in FEATURE_NAMES
        for model in MODEL_NAMES
    )
    grouped_best = max(
        grouped[feat][model]["mean"]
        for feat in FEATURE_NAMES
        for model in MODEL_NAMES
    )

    print("\n" + "=" * 65)
    print("EVALUATION PROTOCOL CONTRAST")
    print("=" * 65)
    print(
        f"  Subject-wise best-model    : {best_arr.mean():.4f} \u00b1 {best_arr.std():.4f} "
        f"(\u2265 80%: {(best_arr >= 0.80).sum()}/{n_subjects}, "
        f"\u2265 70%: {(best_arr >= 0.70).sum()}/{n_subjects})\n"
        f"  Pooled trial-wise mean      : {pooled_trialwise_mean:.4f}\n"
        f"  Pooled trial-wise best cell : {pooled_trialwise_best:.4f}\n"
        f"  Subject-grouped best cell   : {grouped_best:.4f}\n"
        f"  Protocol gap (1 \u2212 2)        : "
        f"{best_arr.mean() - pooled_trialwise_mean:+.4f} "
        f"({(best_arr.mean() - pooled_trialwise_mean) / pooled_trialwise_mean * 100:+.1f}%)"
    )
    print(
        "\n  Most published 'single subject ~80%' claims report protocol 1;\n"
        "  this paper reports protocol 3. Treat that gap as evaluation\n"
        "  methodology, not as missing signal in the data.\n"
    )


def run_core_pipeline():
    """Run the full pipeline from raw data to every tracked artifact."""
    # Imports are local so report-only callers (and unit tests) do not need
    # MNE/MOABB available just to render figures from existing JSONs.
    from .features import save_feature_matrices
    from .models import run_main_experiments, run_per_subject_analysis
    from .preprocessing import save_preprocessed_arrays

    save_preprocessed_arrays()
    save_feature_matrices()
    main_results = run_main_experiments()
    bci_per_subject = run_per_subject_analysis("bci2a")
    physionet_per_subject = run_per_subject_analysis("physionet")
    generate_figures_and_tables(
        main_results=main_results,
        bci_per_subject=bci_per_subject,
        physionet_per_subject=physionet_per_subject,
    )
    _print_evaluation_protocol_comparison(main_results, bci_per_subject)


def report_only():
    """Regenerate figures, tables, and the protocol contrast from existing JSONs.

    Useful when the heavy preprocessing + modelling steps have already run and
    only the publication-facing artifacts need refreshing.
    """
    with TRACKED_RESULT_PATHS["main_experiment_results"].open() as f:
        main_results = json.load(f)
    with TRACKED_RESULT_PATHS["bci2a_per_subject_results"].open() as f:
        bci_per_subject = json.load(f)
    with TRACKED_RESULT_PATHS["physionet_subject_results"].open() as f:
        physionet_per_subject = json.load(f)

    generate_figures_and_tables(
        main_results=main_results,
        bci_per_subject=bci_per_subject,
        physionet_per_subject=physionet_per_subject,
    )
    _print_evaluation_protocol_comparison(main_results, bci_per_subject)


def main(argv: list[str] | None = None):
    """CLI entrypoint for running the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the reproducible EEG feature-robustness pipeline.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip preprocessing/modelling and only regenerate figures and tables.",
    )
    args = parser.parse_args(argv)
    if args.report_only:
        report_only()
    else:
        run_core_pipeline()


if __name__ == "__main__":
    main()
