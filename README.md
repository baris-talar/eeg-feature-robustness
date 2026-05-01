# EEG Feature Robustness

Small reproducible benchmark for classical motor-imagery EEG features under
subject-aware cross-dataset transfer.

The current result is deliberately modest: after subject-grouped evaluation,
PCA dimensionality matching, and target-subject uncertainty estimates, no tested
classical feature representation provides reliable zero-shot transfer between
PhysioNet EEG Motor Imagery and BCI Competition IV Dataset 2a. The old
`FFT > Band Power > Time Domain` claim is not supported.

## What To Read First

- Paper draft: [paper/eeg_feature_robustness.pdf](paper/eeg_feature_robustness.pdf)
- Paper source: [paper/eeg_feature_robustness.tex](paper/eeg_feature_robustness.tex)
- Main results: [results/main_experiment_results.json](results/main_experiment_results.json)
- Summary tables: [results/tables/within_dataset_results.csv](results/tables/within_dataset_results.csv), [results/tables/cross_dataset_results.csv](results/tables/cross_dataset_results.csv)
- Statistical tests: [results/statistical_summary.json](results/statistical_summary.json)

## Main Result

Primary PCA-matched cross-dataset balanced accuracy averaged across models:

| Feature | PhysioNet -> BCI2a | BCI2a -> PhysioNet |
|---|---:|---:|
| Time Domain | 0.681 | 0.559 |
| Band Power | 0.645 | 0.545 |
| FFT | 0.604 | 0.557 |

Target-subject paired permutation tests:

| Comparison | Mean difference | p-value |
|---|---:|---:|
| Time Domain - Band Power | +0.0140 | 0.0006 |
| Time Domain - FFT | +0.0066 | 0.1790 |
| Band Power - FFT | -0.0075 | 0.1456 |

These are small effects. Treat feature rankings as descriptive, not as a strong
claim of one feature family "winning."

## Repository Layout

```text
.
├── src/eeg_feature_robustness/   # Package: config, preprocessing, features, models, figures, pipeline
├── tests/                        # Unit tests
├── results/                      # Tracked result summaries, figures, tables, metadata
├── paper/                        # Paper TeX source and PDF
├── data/                         # Dataset notes only; raw data is not tracked
├── Makefile                      # Reproduction commands
├── pyproject.toml                # Package entrypoints
└── requirements.txt              # Pinned pip environment
```

Large intermediate arrays (`results/*.npy`), Python caches, LaTeX build files,
and local MOABB downloads are intentionally not kept in the working tree. They
can be regenerated.

## Setup

Python `3.11.x` is the reproducible target.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The code automatically uses `~/mne_data` when available, or set
`EEGFR_MNE_DATA_ROOT` to override. With either in place, the local dataset
cache avoids network downloads:

- `~/mne_data/MNE-bnci-data`
- `~/mne_data/MNE-eegbci-data`

## Reproduce

End-to-end (preprocessing → features → modelling → figures → tables):

```bash
make reproduce-core
```

Regenerate publication-facing artifacts only (no preprocessing/modelling):

```bash
make figures
```

Verification:

```bash
make check
```

Cleanup:

```bash
make clean
make clean-intermediates
```

`make clean-intermediates` removes regenerated `.npy` arrays from `results/`.
Run `make reproduce-core` again when you need them.

## Data And Outputs

Raw EEG files are not distributed in this repository. MOABB/MNE loads them from
the local cache or downloads them into the configured MNE data directory.

Tracked outputs in `results/` are the publication-facing artifacts: JSON
results, CSV tables, figures, and trial metadata. Details are in
[results/README.md](results/README.md).

## Status

This is preprint-level as a transparent negative benchmark. It is not yet an
IEEE main-track submission. The next research step is to add CSP/FBCSP,
Riemannian geometry, and domain-adaptation baselines.

## License

See [LICENSE](LICENSE).
