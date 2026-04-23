# Robustness of EEG Feature Representations Under Cross-Dataset Distribution Shift Using Classical Models

**Barış Talar**  
baristalar21@gmail.com  

---

## Abstract

EEG-based brain-computer interfaces (BCIs) frequently report high within-dataset classification accuracy, yet performance collapses when models are applied to new datasets recorded under different conditions. This paper investigates whether frequency-domain feature representations are more robust than time-domain statistical features under cross-dataset distribution shift. Using two publicly available motor imagery datasets — PhysioNet EEG Motor Imagery (109 subjects) and BCI Competition IV Dataset 2a (9 subjects) — we evaluate 9 feature-model combinations under strict zero-shot transfer conditions in both transfer directions. We find that FFT magnitude features showed the smallest mean normalised generalisation gap (0.065), followed by band power (0.070) and time-domain features (0.071), though differences were modest. Critically, within-dataset accuracy across all feature sets was near chance for PhysioNet (53–56%), indicating that commonly reported high within-dataset accuracies do not necessarily reflect genuine cross-dataset generalisation. These results suggest that amplitude-scale differences between recording systems — empirically confirmed as a 9× difference in this study — are a primary barrier to cross-dataset transfer, and that log-transformed frequency features partially mitigate this shift.

---

## 1. Introduction

Brain-computer interfaces based on electroencephalography (EEG) have shown promise in enabling motor-impaired individuals to communicate and control external devices through imagined movement. Motor imagery — the mental rehearsal of a motor action without physical execution — produces characteristic suppression of the mu rhythm (8–12 Hz) and beta rhythm (13–30 Hz) over the contralateral motor cortex, providing a reliable neural signal for classification.

Despite decades of research, EEG-based BCIs face a persistent practical barrier: models trained on one dataset rarely transfer to another. Studies routinely report within-dataset accuracies exceeding 80–90%, yet these numbers are produced by training and testing on the same subjects, often with the same recording hardware. When the same model is applied to a different dataset — different subjects, different amplifiers, different electrode configurations — performance collapses to near chance. This gap between reported and real-world accuracy represents a fundamental challenge for clinical deployment of BCI technology.

The causes of this collapse are multiple: differences in electrode impedance, amplifier gain, and recording hardware introduce amplitude-scale differences; different subject populations introduce inter-subject variability in brain anatomy and task engagement; different experimental protocols introduce differences in trial timing and task design. What remains underexplored is which feature representation degrades least under these conditions, and why.

This paper addresses that question directly. We compare three classical feature representations — time-domain statistical features, frequency-band power, and FFT magnitude — under strict zero-shot cross-dataset transfer conditions. By using classical machine learning models rather than deep learning, we ensure that the feature representation is the only variable changing between experiments. Deep learning conflates feature extraction with classification, making it impossible to isolate which aspect of the representation drives generalisation differences.

Our central hypothesis is that frequency-domain features will be more robust under distribution shift than time-domain features, because EEG is fundamentally a frequency-domain signal whose oscillatory band structure is more stable across recording conditions than its absolute amplitude distribution.

The contributions of this work are: (1) a controlled empirical comparison of three feature representations under cross-dataset transfer, (2) empirical quantification of the amplitude-scale distribution shift between two widely used public datasets, and (3) a transparent characterisation of within-dataset accuracy for classical features on an unscreened subject population, providing a realistic pooled trial-wise baseline for interpreting cross-dataset robustness.

---

## 2. Related Work

**Motor imagery classification.** The classification of motor imagery EEG has been studied extensively since Pfurtscheller and Neuper (2001) established the neurophysiological basis of event-related desynchronisation (ERD) in the mu and beta bands. Standard approaches extract band power features using Welch's power spectral density estimation and classify with linear discriminant analysis (LDA) or support vector machines. Common spatial patterns (CSP) and its variants have also been widely applied to enhance spatial discrimination. Within-dataset accuracies of 70–90% are regularly reported, particularly on competition datasets such as BCI Competition IV Dataset 2a.

**Cross-dataset generalisation.** Cross-dataset generalisation in EEG is significantly less studied than within-dataset evaluation. Jayaram et al. (2016) demonstrated through the MOABB benchmark that many published BCI algorithms show dramatic performance drops under cross-session and cross-subject evaluation. Lotte et al. (2018) reviewed EEG feature extraction methods and noted that cross-subject generalisation remains an open problem. Recent work has explored domain adaptation techniques — including transfer learning, covariate shift correction, and alignment methods — to improve cross-dataset performance. However, these approaches modify the training procedure rather than isolating which underlying feature representation is inherently more robust. Wu et al. (2022) and Schirrmeister et al. (2017) have applied deep learning to cross-dataset EEG transfer, but deep architectures couple feature extraction and classification, obscuring the source of any observed robustness.

**Gap this study fills.** To our knowledge, prior work has rarely isolated feature representation as the sole variable under a strict train-on-one-dataset, test-on-another design with identical preprocessing and both transfer directions. Existing cross-dataset work often uses domain adaptation (which changes the transfer setup), focuses on deep learning (which conflates features and models), or evaluates cross-subject variation within a single dataset rather than truly cross-dataset transfer. Our study provides a controlled classical baseline in which the feature representation is the main variable of interest.

---

## 3. Methods

### 3.1 Datasets

**PhysioNet EEG Motor Imagery** (Schalk et al., 2004; Goldberger et al., 2000) comprises recordings from 109 subjects performing imagined and executed hand and foot movements. EEG was recorded using 64 channels at 160 Hz. We used only the motor imagery runs (runs 0, 1, and 2 in MOABB notation) containing left-hand and right-hand imagery trials, verified by annotation inspection. After preprocessing, 4,918 trials were retained (2,480 left, 2,438 right).

**BCI Competition IV Dataset 2a** (Brunner et al., 2008), accessed as BNCI2014\_001 in MOABB, comprises recordings from 9 subjects performing four-class motor imagery (left hand, right hand, feet, tongue). EEG was recorded using 22 channels at 250 Hz. Only left-hand and right-hand trials were retained. After preprocessing, 2,592 trials were retained (1,296 left, 1,296 right).

Both datasets were accessed programmatically via MOABB (Jayaram & Barachant, 2018), ensuring reproducible data loading without manual file manipulation.

### 3.2 Preprocessing

An identical preprocessing pipeline was applied to both datasets to avoid introducing confounds:

1. **Channel selection**: All recordings were reduced to the 22 channels constituting the BCI2a electrode montage, which is a subset of PhysioNet's 64-channel set. No interpolation was required.
2. **Resampling**: PhysioNet recordings were upsampled from 160 Hz to 250 Hz using MNE's polyphase filter resampling with automatic padding. Upsampling was preferred over downsampling to preserve all frequency content.
3. **Bandpass filtering**: A 4–40 Hz bandpass filter was applied using MNE's default FIR filter. This range captures the motor imagery-relevant mu (8–12 Hz) and beta (13–30 Hz) rhythms while removing slow DC drifts and high-frequency EMG artifacts.
4. **Epoch extraction**: Epochs were extracted from 0.5 to 2.5 seconds post-cue onset. The 0.5-second offset avoids the visual evoked potential elicited by the cue. No baseline correction was applied, as the bandpass filter renders epoch means approximately zero.
5. **Label encoding**: Left-hand imagery was encoded as 0, right-hand imagery as 1. All other classes were discarded.

Output arrays had shape (n\_trials, 22, 501), where 501 timepoints corresponds to 2 seconds at 250 Hz with MNE's inclusive endpoint.

### 3.3 Feature Extraction

Three feature representations were extracted from preprocessed epochs:

**Feature Set A — Time Domain (44 dimensions)**: Mean and variance were computed for each of the 22 channels across the time axis, yielding 44 features per trial. Because the signal is bandpass filtered, means are approximately zero for all trials; variance is the primary discriminative feature. This set serves as the weakest baseline.

**Feature Set B — Band Power (88 dimensions)**: Power spectral density was estimated using Welch's method (nperseg=256, 50% overlap, Hann window) for each channel. Average power was computed in four canonical EEG frequency bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), and beta (13–30 Hz). A natural log transform was applied to each band power value to normalise the skewed distribution. This yielded 22 × 4 = 88 features per trial.

**Feature Set C — FFT Magnitude (792 dimensions)**: The magnitude spectrum of the FFT was computed for each channel, restricted to the 4–40 Hz range, and interpolated to 36 fixed frequency bins using linear interpolation. No log transform was applied; StandardScaler within the classification pipeline handles normalisation. This yielded 22 × 36 = 792 features per trial.

### 3.4 Models

Three classical classifiers were evaluated:

- **Logistic Regression**: Linear classifier with L2 regularisation (max\_iter=5000, random\_state=42), preceded by StandardScaler.
- **Support Vector Machine**: RBF kernel SVM (C=1.0, gamma='scale', random\_state=42), preceded by StandardScaler.
- **Random Forest**: Ensemble of 100 decision trees (random\_state=42). No feature scaling applied, as tree-based models are scale-invariant.

StandardScaler was placed inside a scikit-learn Pipeline to ensure scaling was fitted only on training data, preventing data leakage.

### 3.5 Evaluation Protocol

**Within-dataset evaluation** used 5-fold stratified cross-validation (shuffle=True, random\_state=42). Stratification preserved the 50/50 class balance within each fold. Because trials were pooled across subjects before splitting, these within-dataset scores should be interpreted as pooled trial-wise baselines rather than fully subject-independent grouped-subject estimates.

**Cross-dataset evaluation** used zero-shot transfer: models were trained on the entire source dataset and tested on the entire target dataset with no overlap and no fine-tuning. Both transfer directions were evaluated: PhysioNet → BCI2a and BCI2a → PhysioNet.

The primary metric was balanced accuracy, defined as the mean of per-class recall, which is robust to class imbalance. All experiments used random seed 42.

**Generalisation gap** was computed as:

```
Gap = Within accuracy − Cross accuracy
Normalised Gap = Gap / Within accuracy
```

The normalised gap expresses the fraction of within-dataset performance lost under distribution shift, enabling comparison across feature sets with different absolute accuracy levels.

---

## 4. Results

### 4.1 Within-Dataset Accuracy

Table 1 reports balanced accuracy for within-dataset evaluation. BCI2a within-dataset accuracy (59–64%) consistently exceeded PhysioNet accuracy (51–56%) across all feature sets and models. Logistic Regression was the strongest model in both datasets for most feature sets. Band Power achieved the highest single result: 64.2% on BCI2a with Logistic Regression.

PhysioNet within-dataset accuracy was near chance (51–56%) across all feature sets and models. This reflects the unscreened nature of the 109-subject pool and the high inter-subject variability of motor imagery signal quality.

**Table 1. Within-Dataset Balanced Accuracy (mean ± std, 5-fold CV)**

| Feature Set | Model | PhysioNet | BCI2a |
|---|---|---|---|
| Time Domain | Log. Reg. | 0.535 ± 0.018 | 0.639 ± 0.014 |
| Time Domain | SVM | 0.523 ± 0.010 | 0.591 ± 0.029 |
| Time Domain | Rand. Forest | 0.509 ± 0.007 | 0.588 ± 0.038 |
| Band Power | Log. Reg. | 0.559 ± 0.026 | 0.642 ± 0.027 |
| Band Power | SVM | 0.529 ± 0.015 | 0.596 ± 0.021 |
| Band Power | Rand. Forest | 0.512 ± 0.017 | 0.608 ± 0.019 |
| FFT | Log. Reg. | 0.556 ± 0.014 | 0.613 ± 0.015 |
| FFT | SVM | 0.539 ± 0.010 | 0.595 ± 0.012 |
| FFT | Rand. Forest | 0.520 ± 0.024 | 0.567 ± 0.010 |

### 4.2 Cross-Dataset Accuracy and Generalisation Gap

Table 2 reports cross-dataset balanced accuracy and normalised generalisation gap for both transfer directions. The two directions showed strikingly different patterns.

In the PhysioNet → BCI2a direction, gaps were near zero or slightly negative across all feature sets and models. Logistic Regression consistently achieved slightly higher cross-dataset accuracy than within-dataset accuracy on PhysioNet, producing negative gaps. This is a consequence of the near-chance PhysioNet baseline rather than evidence of successful transfer.

In the BCI2a → PhysioNet direction, gaps were substantially larger (0.12–0.17), reflecting genuine degradation when a model trained on 9 subjects is applied to 109 subjects from a different recording system.

**Table 2. Cross-Dataset Accuracy and Normalised Generalisation Gap**

| Feature Set | Model | Phys→BCI acc | Phys→BCI gap | BCI→Phys acc | BCI→Phys gap |
|---|---|---|---|---|---|
| Time Domain | Log. Reg. | 0.563 | −0.054 | 0.531 | 0.168 |
| Time Domain | SVM | 0.500 | 0.044 | 0.513 | 0.133 |
| Time Domain | Rand. Forest | 0.505 | 0.009 | 0.514 | 0.125 |
| Band Power | Log. Reg. | 0.581 | −0.038 | 0.537 | 0.164 |
| Band Power | SVM | 0.526 | 0.006 | 0.512 | 0.142 |
| Band Power | Rand. Forest | 0.522 | −0.019 | 0.507 | 0.167 |
| FFT | Log. Reg. | 0.572 | −0.029 | 0.528 | 0.140 |
| FFT | SVM | 0.536 | 0.006 | 0.510 | 0.142 |
| FFT | Rand. Forest | 0.514 | 0.012 | 0.499 | 0.120 |

Averaged across all models and both directions, FFT showed the smallest mean normalised gap (0.065), followed by Band Power (0.070) and Time Domain (0.071). Logistic Regression was the most robust model (mean normalised gap: 0.059).

---

## 5. Discussion

### 5.1 The Amplitude Distribution Shift

A diagnostic analysis of the preprocessed arrays revealed a 9× amplitude difference between datasets: PhysioNet signals ranged ±856 μV while BCI2a signals ranged ±96 μV. This difference arises from different recording hardware, electrode impedance, and amplifier gain settings between the two labs. This is not a signal-of-interest difference — it reflects recording system characteristics unrelated to the underlying brain activity.

This amplitude gap is the primary mechanistic explanation for cross-dataset degradation. Time-domain variance features are directly proportional to amplitude squared, making them maximally sensitive to this shift. FFT magnitude features are linearly proportional to amplitude. Log-transformed band power compresses the scale difference — verified empirically: despite the 9× raw amplitude difference, log-transformed band power means were nearly identical across datasets (PhysioNet: −22.8, BCI2a: −23.0).

### 5.2 Why FFT Outperformed Band Power

The hypothesis predicted band power as the most robust feature set, but FFT showed a slightly smaller average gap. This is counterintuitive given that log-transformed band power demonstrably compresses the amplitude distribution shift. A likely explanation is that band power's coarser frequency resolution (four broad bands) loses discriminative information that FFT's finer 36-bin representation retains. The StandardScaler inside the pipeline normalises FFT's scale at training time, partially compensating for amplitude differences without the coarsening effect of band aggregation.

### 5.3 The Near-Chance PhysioNet Baseline

Within-dataset accuracy on PhysioNet was near chance (51–56%) across all configurations. This is not a pipeline error — run verification confirmed correct trial selection, and signal diagnostics confirmed valid EEG-range amplitudes. The near-chance result reflects the genuine difficulty of motor imagery classification on an unscreened, heterogeneous 109-subject population using classical features without subject-specific calibration.

This finding has a direct implication for the literature. Reported accuracies of 80–90% on similar paradigms may reflect differences in evaluation protocol, subject selection, or subject-specific calibration rather than population-level cross-dataset generalisation. In this study, the within-dataset scores should be interpreted as pooled trial-wise cross-validation baselines rather than fully subject-independent population-level estimates, because trials from the same subject can appear in different folds.

### 5.4 The Asymmetric Transfer Directions

The two transfer directions told fundamentally different stories. PhysioNet → BCI2a showed near-zero or negative gaps, not because transfer worked well but because the PhysioNet baseline was already at chance. BCI → PhysioNet revealed genuine degradation of 12–17%, as training on 9 subjects from one recording system cannot capture the full distribution of 109 subjects from a different system.

This asymmetry illustrates a methodological point: running only one transfer direction can be deeply misleading. A study that only reports PhysioNet → BCI2a would conclude that all feature sets transfer equally well. The reverse direction reveals the real pattern.

### 5.5 Logistic Regression as the Most Robust Model

Logistic Regression showed the smallest average generalisation gap (0.059) across all feature sets. This is consistent with the principle that simpler models generalise better under distribution shift. Complex models like Random Forest can overfit to dataset-specific patterns during training, amplifying degradation when those patterns do not exist in the target dataset.

---

## 6. Limitations

**Two datasets only.** All findings are based on two datasets with similar motor imagery paradigms (left vs. right hand) recorded with different but structurally compatible hardware. Generalisation of these findings to other EEG tasks (P300, SSVEP, cognitive load) or more diverse recording hardware cannot be assumed.

**Near-chance PhysioNet baseline.** The near-chance within-dataset accuracy on PhysioNet severely limits the interpretability of the PhysioNet → BCI2a transfer direction. A meaningful generalisation gap requires a baseline sufficiently above chance that degradation is observable. Future work should use datasets with stronger within-dataset baselines, potentially through per-subject calibration or subject selection criteria.

**Classical models only.** Deep learning may show qualitatively different generalisation patterns. Convolutional neural networks applied to raw EEG can learn representations that are not directly comparable to the hand-crafted features studied here. This study deliberately excludes deep learning to isolate feature representation effects, but the findings should not be extrapolated to deep architectures.

**Binary classification.** Only left vs. right hand imagery was studied. Multi-class settings involving feet or tongue imagery may show different feature robustness patterns due to different neural generator locations.

**No domain adaptation.** Results represent strict zero-shot transfer with no fine-tuning, alignment, or covariate shift correction. Domain adaptation methods are known to improve cross-dataset performance; this study establishes the unadapted baseline against which such methods should be compared.

**Single random seed.** Cross-dataset experiments were run with a single random seed. While within-dataset results include variance estimates from 5-fold cross-validation, the cross-dataset point estimates lack confidence intervals. Future work should report results across multiple seeds.

**Subject variability not analysed.** Individual subject differences were not studied. Some subjects may show strong motor imagery signals while others show none. Subject-level analysis might reveal that the population-level near-chance accuracy masks a bimodal distribution of strong and weak responders.

---

## 7. Conclusion

This study provides a controlled empirical comparison of three classical EEG feature representations — time-domain statistics, log-transformed band power, and FFT magnitude — under strict zero-shot cross-dataset transfer between two publicly available motor imagery datasets. We find that FFT features showed the smallest mean normalised generalisation gap (0.065), followed by band power (0.070) and time-domain features (0.071), with Logistic Regression as the most robust classifier. A 9× amplitude difference between recording systems was identified as the primary mechanistic source of distribution shift. Log-transformed band power partially mitigates this shift through scale compression, but the differences between feature sets were modest, suggesting that no classical feature representation fully solves the cross-dataset transfer problem. Within-dataset accuracy on an unscreened 109-subject population was near chance (51–56%), and in this study should be interpreted as a pooled trial-wise baseline rather than a fully subject-independent estimate. These findings highlight the importance of cross-dataset evaluation as a standard practice in BCI research and motivate future work on amplitude-invariant feature representations for robust neural decoding.

---

## References

Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008 – Graz data set A. Institute for Knowledge Discovery, Graz University of Technology.

Goldberger, A. L., Amaral, L. A. N., Glass, L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220.

Jayaram, V., & Barachant, A. (2018). MOABB: Trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*, 15(6).

Jayaram, V., Alamgir, M., Altun, Y., Schölkopf, B., & Grosse-Wentrup, M. (2016). Transfer learning in brain-computer interfaces. *IEEE Computational Intelligence Magazine*, 11(1), 20–31.

Lotte, F., Bougrain, L., Cichocki, A., et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: A 10-year update. *Journal of Neural Engineering*, 15(3).

Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123–1134.

Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.
