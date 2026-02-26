# Not All Entropy Is Equal

Analysis code and results for: **"Not All Entropy Is Equal: Parameter Sensitivity, Amplitude Blindness, and the Case for Sample Entropy in Dementia EEG"**

Preprint: *TODO — medRxiv link*

## What This Does

Four scripts that analyze EEG entropy measures as biomarkers for dementia on 1,177 clinical recordings from the CAUEEG dataset. The main findings:

- Permutation entropy (PE) gives effect sizes from d = −0.700 to d = +0.709 on the *same data* depending on parameter choice — including a direction reversal
- PE with proper alpha-timescale parameters is a complete null (d = −0.025)
- Sample entropy (SE_α) is spectrally independent (r = −0.043) and the strongest entropy measure (d = 0.519, AUC = 0.720)
- Combining SE_α with the spectral power ratio yields cross-validated AUC = 0.786

## Requirements

Python 3.8+

```bash
pip install -r requirements.txt
```

Dependencies: mne, antropy, numpy, pandas, scipy, scikit-learn, matplotlib

## Data Access

Raw EEG data is not included. You need to obtain these datasets separately:

**CAUEEG** (required for all scripts): Request access via [the official form](https://forms.gle/gqas9pXfDZLPWqX97). See [Kim et al., 2023](https://doi.org/10.1016/j.neuroimage.2023.120054). Place the downloaded `caueeg-dataset/` folder in the project root or a parent directory — the scripts will find it automatically.

**ds004504** (discovery dataset, referenced in paper only): Available on [OpenNeuro](https://openneuro.org/datasets/ds004504). Not required to run any scripts.

## Scripts

Run in this order:

### 1. `analysis_code/caueeg_entropy_v3.py` — Main pipeline (~60 min)

Processes all 1,388 CAUEEG EDF files. For each recording: extracts eyes-closed segments from event markers, excludes artifact-marked intervals (±1s buffer), bandpass filters to alpha (8–12 Hz) and theta (4–8 Hz), and computes entropy per-segment with weighted averaging. Measures computed per channel, averaged across 19 channels:

- **Permutation entropy:** 4 parameterizations (order/delay: 5/5, 3/10, 7/3, 3/1) spanning different timescales
- **Sample entropy:** embedding dimension 2, Chebyshev metric, tolerance r = 0.2 × SD (per-segment)
- **Lempel-Ziv complexity:** median-threshold binarization, normalized
- **Spectral power:** Welch PSD, absolute/relative alpha and theta power, alpha/theta power ratio

Outputs `data_results/caueeg_entropy_v3.csv` (1,177 subjects × 31 columns) and `data_results/caueeg_entropy_v3_summary.txt` with all group comparisons (Welch's t, Cohen's d, AUC), age correction (residualized + age-matched 70–80), and entropy–spectral correlations.

### 2. `analysis_code/generate_figures.py` — Publication figures (~30 sec)

Reads the v3 CSV and produces 4 figures in `figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_pe_sensitivity.png` | PE parameter sensitivity — Cohen's d with bootstrap 95% CIs across 4 parameterizations |
| `fig2_embedding_windows.png` | Embedding window schematic — sub-cycle (10 ms) vs proper alpha (100 ms) on synthetic waveform |
| `fig3_headtohead.png` | Head-to-head comparison of all measures (entropy + spectral), horizontal bar chart |
| `fig4_age_correction.png` | Age correction impact — raw vs age-residualized effect sizes |

### 3. `analysis_code/cross_validate_combined_model.py` — Cross-validation (~2 min)

10-fold stratified CV, repeated 10 times. Tests logistic regression models with different feature combinations (Power Ratio alone, SE_α alone, LZC Ratio alone, Power + SE_α, Power + LZC, Power + SE + LZC, Power + SE + Age) for both Dementia vs Normal and MCI vs Normal. Reports mean AUC with 95% CI and apparent AUCs for overfitting comparison.

### 4. `analysis_code/test_interaction_term.py` — Interaction term test (~2 min)

Tests whether Power_Ratio × SE_α interaction improves the logistic model. Reports cross-validated AUCs with and without interaction, plus Wald test coefficients. Spoiler: the interaction is non-significant (p = 0.93), confirming pure linear additivity.

## Results

The `data_results/` folder contains precomputed output from all scripts (except the large CSV, which you regenerate with script 1). The `figures/` folder contains the 4 publication PNGs. These are included so you can inspect results without running the full pipeline.

## Pre-Registration

`pre_registration/` contains the dataset access request document written before any CAUEEG data were examined. It specifies a priori predictions for the alpha/theta LZC ratio. The PE parameterization analysis is entirely exploratory.

## Citation

If you use this code, please cite the preprint:

```
TODO — citation once preprint is posted
```

And the datasets:

```bibtex
@article{kim2023deep,
  title={Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia: algorithms and dataset},
  author={Kim, Min-jae and Youn, Young Chul and Paik, Joonki},
  journal={NeuroImage},
  volume={272},
  pages={120054},
  year={2023},
  publisher={Elsevier}
}

@article{miltiadous2023dataset,
  title={A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG},
  author={Miltiadous, Andreas and others},
  journal={Data},
  volume={8},
  number={6},
  pages={95},
  year={2023}
}
```

## License

MIT

## Author

Victor Edmonds
