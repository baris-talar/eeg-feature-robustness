# Results

This directory holds the publication-facing artifacts for the repository.
Large intermediate arrays (`results/*.npy`) are intentionally omitted from the
tidy working tree; regenerate them with `make reproduce-core`.

## Layout

```text
results/
├── main_experiment_results.json     ← single source of truth (within + cross + statistical_summary)
├── statistical_summary.json         ← target-subject paired feature-comparison tests
├── bci2a_per_subject_results.json   ← per-subject CV (BCI2a, all 3 classifiers)
├── physionet_subject_results.json   ← per-subject CV (PhysioNet, LogReg)
├── figures/                         ← PNGs referenced by the paper
│   ├── within_dataset_accuracy_heatmap.png
│   ├── generalisation_gap_by_direction.png
│   ├── within_vs_cross_dataset_accuracy.png
│   ├── bci2a_per_subject_accuracy.png
│   └── combined_per_subject_accuracy.png
├── tables/                          ← CSV summary tables (with CI95 columns)
│   ├── within_dataset_results.csv
│   └── cross_dataset_results.csv
└── metadata/                        ← per-trial subject/session/run identifiers
    ├── bci2a_trial_metadata.csv
    └── physionet_trial_metadata.csv
```

## Regenerating outputs

End-to-end (preprocessing → features → modelling → figures → tables):

```bash
make reproduce-core
```

Just refresh figures and tables from the existing JSONs:

```bash
make figures
```

To remove local build clutter and regenerated arrays:

```bash
make clean
make clean-intermediates
```
