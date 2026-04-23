# Robustness of EEG Feature Representations Under Cross-Dataset Distribution Shift

Independent research project comparing how classical EEG feature representations behave under strict zero-shot transfer between public motor-imagery datasets.

## Research Question

When a model is trained on one EEG dataset and evaluated on a different dataset collected under different conditions, which feature representation degrades the least?

This project tests the hypothesis that frequency-domain features are more robust than simple time-domain statistics under cross-dataset distribution shift.

## Why This Matters

Many EEG classification papers report strong within-dataset accuracy, but those results often do not reflect real generalisation. A model trained and tested on the same dataset may perform well while failing on recordings from a different subject pool, amplifier, or lab setting.

This project studies that gap directly using a controlled setup:

- same preprocessing for both datasets
- same label space
- same model family grid
- same evaluation protocol
- only the feature representation changes

## Datasets

Two public motor-imagery datasets are used through [MOABB](https://github.com/NeuroTechX/moabb):

| Dataset | Subjects | Channels | Sampling Rate | Task Used |
|---|---:|---:|---:|---|
| PhysioNet EEG Motor Imagery | 109 | 64 -> 22 shared | 160 Hz -> 250 Hz | Left vs right imagery |
| BCI Competition IV Dataset 2a | 9 | 22 | 250 Hz | Left vs right imagery |

Only the shared binary label space is kept:

- `0` = left hand
- `1` = right hand

The shared channel space is the 22-channel BCI2a montage. PhysioNet is restricted to those channels so both datasets use the same spatial layout.

## Feature Representations

Three feature sets are compared:

| Label | Feature Set | Dimensions | Description |
|---|---|---:|---|
| A | Time Domain | 44 | Per-channel mean and variance |
| B | Band Power | 88 | Welch PSD in delta, theta, alpha, beta bands with log transform |
| C | FFT Magnitude | 792 | FFT magnitude from 4-40 Hz interpolated to fixed bins |

## Models

The same three classical classifiers are used for every feature set:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

Deep learning is deliberately excluded so the comparison isolates the effect of feature representation rather than learned feature extraction.

## Experimental Design

Four evaluations are run:

1. Within PhysioNet: 5-fold stratified cross-validation
2. Within BCI2a: 5-fold stratified cross-validation
3. Train on PhysioNet, test on BCI2a
4. Train on BCI2a, test on PhysioNet

Cross-dataset evaluation is strict zero-shot transfer:

- no overlap between train and test datasets
- no fine-tuning on the target dataset
- no domain adaptation

## Primary Metric

The main metric is balanced accuracy.

To measure robustness under distribution shift, the project computes:

`generalisation gap = within-dataset accuracy - cross-dataset accuracy`

and

`normalised generalisation gap = generalisation gap / within-dataset accuracy`

Lower normalised gap means the feature representation is more robust under transfer.

## Preprocessing Pipeline

The same preprocessing is applied to both datasets:

1. Load raw EEG with MNE through MOABB
2. Select the shared 22 EEG channels
3. Resample to 250 Hz
4. Apply a 4-40 Hz bandpass filter
5. Extract epochs from 0.5 to 2.5 seconds after cue onset
6. Keep only left-hand and right-hand imagery trials
7. Save arrays of shape `(n_trials, 22, 501)`

The `501` samples arise from MNE's inclusive epoch endpoint.

## Main Finding

From the saved experiment results:

| Feature Set | Mean Normalised Gap |
|---|---:|
| FFT | 0.065 |
| Band Power | 0.070 |
| Time Domain | 0.071 |

FFT produced the smallest average normalised generalisation gap, though the differences between feature sets were modest.

The analysis also found a large amplitude-scale mismatch between the two datasets, which appears to be a major mechanistic source of cross-dataset degradation.

## Repository Structure

```text
eeg-feature-robustness/
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   ├── analysis.py
│   ├── loader.py
│   └── experiment.py
├── data/
│   └── README.txt
├── notebooks/
│   └── 01_data_exploration.ipynb
├── paper/
│   ├── eeg_feature_robustness.md
│   ├── eeg_feature_robustness.tex
│   └── eeg_feature_robustness.pdf
├── results/
│   ├── all_results.json
│   ├── table1_within.csv
│   ├── table2_cross.csv
│   ├── fig1_within_dataset.png
│   ├── fig2_gen_gap.png
│   └── fig3_scatter.png
├── research_log.txt
├── requirements.txt
└── README.md
```

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

Run these scripts from the repository root in this order:

```bash
python src/preprocess.py
python src/features.py
python src/models.py
python src/analysis.py
```

What each step produces:

- `src/preprocess.py`: downloads data through MOABB, extracts epochs, saves raw trial arrays to `results/*.npy`
- `src/features.py`: builds feature matrices for A, B, and C, saves them to `results/*.npy`
- `src/models.py`: runs within-dataset and cross-dataset experiments, writes `results/all_results.json`
- `src/analysis.py`: generates figures, CSV tables, and printed written analysis

All generated outputs are written to `results/`.

## Paper

The manuscript is available in three forms:

- Markdown: [paper/eeg_feature_robustness.md](paper/eeg_feature_robustness.md)
- LaTeX: [paper/eeg_feature_robustness.tex](paper/eeg_feature_robustness.tex)
- PDF: [paper/eeg_feature_robustness.pdf](paper/eeg_feature_robustness.pdf)

## Notes

- PhysioNet run naming can differ between source documentation and MOABB's internal representation. This repository uses the run identifiers returned by MOABB in the preprocessing script.
- The project is designed as a controlled classical baseline for cross-dataset robustness, not as a domain-adaptation or deep-learning benchmark.

## License

See [LICENSE](LICENSE).
