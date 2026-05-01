# Data Access

Raw EEG data are not distributed with this repository because of file-size
constraints and dataset-specific licensing terms. Both datasets are publicly
available through MOABB. The pipeline automatically uses the local MNE cache
under `~/mne_data` when present, or set `EEGFR_MNE_DATA_ROOT` to point at a
different cache root:

- `~/mne_data/MNE-bnci-data`
- `~/mne_data/MNE-eegbci-data`

If neither is present, MOABB/MNE will use the configured MNE data directory
and may download the public files there.

## Sources

### PhysioNet EEG Motor Movement/Imagery Dataset

- Approximate size: 1.9 GB
- Citation: Schalk et al. (2004); Goldberger et al. (2000)
- URL: https://physionet.org/content/eegmmidb/1.0.0/
- MOABB loader: `moabb.datasets.PhysionetMI()`

### BCI Competition IV Dataset 2a / BNCI2014-001

- Approximate size: 100 MB
- Citation: Brunner et al. (2008)
- URL: https://www.bbci.de/competition/iv/
- MOABB loader: `moabb.datasets.BNCI2014_001()`

## Reproducing the Pipeline

Install the runtime dependencies from `requirements.txt`, then run the
following from the repository root:

```bash
make reproduce-core
```

All generated outputs are written to `results/`.
