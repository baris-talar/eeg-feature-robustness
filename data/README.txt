Raw EEG data are not distributed with this repository owing to file size
constraints and dataset-specific licensing terms. Both datasets are publicly
available and will be downloaded automatically by MOABB upon first execution
of the preprocessing pipeline.

PhysioNet EEG Motor Movement/Imagery Dataset (~1.9 GB)
  Schalk et al. (2004); Goldberger et al. (2000)
  https://physionet.org/content/eegmmidb/1.0.0/
  Accessed programmatically via: moabb.datasets.PhysionetMI()

BCI Competition IV Dataset 2a / BNCI2014-001 (~100 MB)
  Brunner et al. (2008)
  https://www.bbci.de/competition/iv/
  Accessed programmatically via: moabb.datasets.BNCI2014_001()

To reproduce the full experimental pipeline from raw data, install all
dependencies listed in requirements.txt and execute the following scripts
in order:

  python src/preprocess.py   # epoch extraction and raw trial-array generation
  python src/features.py     # feature set construction (A, B, C)
  python src/models.py       # within- and cross-dataset evaluation
  python src/analysis.py     # figures, tables, and written analysis

All results will be written to the results/ directory.
