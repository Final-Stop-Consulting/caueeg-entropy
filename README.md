# Not All Entropy Is Equal

Analysis code and results for **"Not All Entropy Is Equal: Parameter Sensitivity and Cross-Dataset Fragility of Band-Filtered Entropy in Dementia EEG."**

Preprint: *TODO — link once posted.*

## What this does

This repository analyzes entropy and complexity measures as EEG biomarkers for dementia, on 1,177 clinical recordings from the CAUEEG dataset, with a cross-dataset check on the independent ds004504 cohort. It began as a validation study and became a methodological one. The central results:

- **Permutation entropy (PE) is parameter-dependent to the point of reversal.** On the *same* alpha-band data, four PE parameterizations give effect sizes from *d* = −0.70 to *d* = +0.71 for dementia vs. normal — a direction reversal, plus two complete nulls. The timescale-appropriate parameterization (order 5, delay 5; one full alpha cycle) is a null (*d* = −0.025).
- **Sub-cycle PE is a phase measure, not a complexity measure.** We prove analytically that order-3/delay-1 PE on a narrowband signal reduces to the entropy of the instantaneous phase, and confirm it on the real EEG: the ordinal pattern is fixed by the Hilbert phase on ~90% of samples, and PE equals the phase-predicted entropy (Pearson *r* = 0.95).
- **Sample entropy (SE) is not exempt.** Sweeping the embedding dimension *m* ∈ {1, 2, 3}, the alpha-band SE effect runs from null (*m* = 1) to *d* = +0.86 (*m* = 3). Within *m* ≥ 2 it is tolerance-robust and spectrally independent (*r* = −0.04), but the embedding dimension is a genuine, usually-unexamined degree of freedom.
- **Nothing entropy-based generalizes across cohorts; the spectral baseline does.** Under a harmonized pipeline with matched physical timescale (ds004504 downsampled 500 → 200 Hz), the PE parameter-sensitivity *profile* reproduces across datasets (Pearson *r* = 0.997), but no entropy measure retains its *direction* out of sample — both PE and SE reverse. The relative alpha/theta **power ratio** is the largest single effect (*d* = −0.73, AUC = 0.739) and the only measure whose direction replicates across cohorts.

The practical message: on band-filtered EEG, entropy measures are parameter- and cohort-fragile, and a simple spectral power ratio is the baseline they should be benchmarked against.

## Repository layout

```
analysis_code/     analysis + statistics scripts (see Pipeline below)
data_results/      computed outputs (summaries, reports, per-subject CSVs)
figures/           publication figures (PNG)
pre_registration/  dataset access request with a priori predictions
requirements.txt   Python dependencies
LICENSE            MIT
```

Raw EEG datasets are **not** included (see Data access). The manuscript and writing materials are maintained separately and are not part of this repository.

## Requirements

Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies: `mne`, `antropy`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.

## Data access

Obtain the datasets separately and place each folder at the repository root (or in a `data/` subfolder); the scripts search both locations automatically.

- **CAUEEG** (required for most scripts) — request access via the official form and see [Kim et al., 2023](https://doi.org/10.1016/j.neuroimage.2023.120054). Expected as `caueeg-dataset/` (with `signal/edf/`, `event/`, `annotation.json`). Publicly available upon request for academic use; "upon request" is not the same as openly downloadable.
- **ds004504** (discovery + cross-dataset) — available on [OpenNeuro](https://openneuro.org/datasets/ds004504); see [Miltiadous et al., 2023](https://doi.org/10.3390/data8060095). Expected as `ds004504/`.

## Pipeline

### Core CAUEEG analysis

| # | Script | Purpose | Output | Runtime |
|---|--------|---------|--------|---------|
| 1 | `caueeg_entropy_v3.py` | Main pipeline. Eyes-closed extraction, artifact exclusion, per-segment entropy. Computes PE (4 parameterizations), SE (*m* = 2), LZC, and spectral power per channel, averaged over 19 channels. | `data_results/caueeg_entropy_v3.csv`, `..._summary.txt` | ~60 min |
| 2 | `generate_figures.py` | The four publication figures from the v3 CSV. | `figures/fig1–4.png` | ~30 s |
| 3 | `cross_validate_combined_model.py` | 10×10 stratified CV of logistic models (power ratio, SE, LZC, and combinations). | `cross_validation_combined_model_results.txt` | ~2 min |
| 4 | `test_interaction_term.py` | Tests a Power × SE interaction term (non-significant). | `test_interaction_term_results.txt` | ~2 min |

### Symmetric-rigor analyses (entropy parameter dependence)

| # | Script | Purpose | Output | Runtime |
|---|--------|---------|--------|---------|
| 5 | `se_parameter_sweep.py` | Sweeps SE over *m* ∈ {1,2,3} × *r* ∈ {0.10–0.30}×SD, whole-scalp and posterior (occipital/parietal) subsets. | `se_sweep_values.csv` | ~45–90 min |
| 6 | `se_sweep_stats.py` | Effect sizes/AUCs and **one** Benjamini–Hochberg FDR across the full measure × parameter grid; SE-vs-PE stability; whole vs. posterior; cross-dataset SE. | `se_sweep_report.txt` | <1 min |
| 7 | `phase_entropy_validation.py` | Empirical test of the §4.2 reduction: shows sub-cycle PE is fixed by instantaneous (Hilbert) phase on the real EEG. | `phase_entropy_validation.csv` | ~10 min |

### Cross-dataset analysis (ds004504)

| # | Script | Purpose | Output | Runtime |
|---|--------|---------|--------|---------|
| 8 | `crossdataset_harmonize.py` | Brings ds004504 up to the CAUEEG pipeline and downsamples 500 → 200 Hz, so identical integer (order, delay) match identical physical timescales. | `ds004504_harmonized_values.csv` | ~minutes |
| 9 | `crossdataset_convergence.py` | Compares the harmonized, timescale-matched PE profile across datasets (effect sizes, bootstrap CIs, d-profile concordance). | `crossdataset_convergence_report.txt` | <1 min |

Typical order: **1** → (2, 3, 4) → **5 → 6**, **7**, **8 → 9**. Scripts 6, 7, and 9 reuse outputs from earlier scripts (and the CAUEEG loader), so run the core pipeline first.

### Retired / superseded

These are retained for transparency but are superseded by the scripts above:

- `test_pe_cross_srate.py` — a **preliminary** cross-sampling-rate PE test (native 500 Hz, first-60 s, no harmonization). Superseded by `crossdataset_harmonize.py` + `crossdataset_convergence.py`, which harmonize the pipeline and match the physical timescale by downsampling.
- `test_se_tolerance.py` — the original SE robustness check varying only the tolerance *r* at *m* = 2. Subsumed by `se_parameter_sweep.py`, which additionally sweeps the embedding dimension *m*.
- Earlier v1/v2 pipelines and exploratory scripts are not included in this repository; `caueeg_entropy_v3.py` is the current, corrected pipeline.

## Results

`data_results/` holds the computed outputs (summary text reports and per-subject CSVs) so results can be inspected without re-running. `figures/` holds the four publication PNGs. The large per-subject CSVs regenerate from scripts 1, 5, 7, and 8.

## Pre-registration and scope

`pre_registration/` contains the dataset access request written before any CAUEEG data were examined. It specifies *a priori* predictions for the alpha/theta **LZC ratio** only. The permutation-entropy and sample-entropy parameterization analyses are exploratory; this distinction is stated throughout the manuscript.

## Reproduction

```bash
pip install -r requirements.txt
# place caueeg-dataset/ and ds004504/ at the repo root (or under data/)
python analysis_code/caueeg_entropy_v3.py          # main CAUEEG pipeline
python analysis_code/generate_figures.py
python analysis_code/se_parameter_sweep.py         # then:
python analysis_code/se_sweep_stats.py
python analysis_code/phase_entropy_validation.py
python analysis_code/crossdataset_harmonize.py     # then:
python analysis_code/crossdataset_convergence.py
```

Heavy scripts print a live progress bar; several accept `--max-subjects N` for a quick test run, and the stats scripts accept `--self-test` to validate their logic on synthetic data.

## Citation

If you use this code, please cite the preprint (TODO) and the datasets:

```bibtex
@article{kim2023deep,
  title={Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia: algorithms and dataset},
  author={Kim, Min-jae and Youn, Young Chul and Paik, Joonki},
  journal={NeuroImage}, volume={272}, pages={120054}, year={2023}, publisher={Elsevier}
}

@misc{miltiadous2024ds004504,
  title={A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects},
  author={Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Evripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
  year={2024}, publisher={OpenNeuro}, note={[Dataset], version 1.0.8 (version analyzed)}, doi={10.18112/openneuro.ds004504.v1.0.8}
}

% Companion data descriptor:
@article{miltiadous2023dataset,
  title={A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG},
  author={Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Evripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
  journal={Data}, volume={8}, number={6}, pages={95}, year={2023}, doi={10.3390/data8060095}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

Victor Edmonds
