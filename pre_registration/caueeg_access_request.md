# Research Plan for CAUEEG Dataset Access Request

**Title:** Testing a Band-Specific Entropy Ratio for Early Detection of Mild Cognitive Impairment

**Principal Investigator:** Victor Edmonds  
**Affiliation:** Final Stop Consulting (Independent Researcher)  
**Date:** January 2026  
**Requested Dataset:** CAUEEG (Chung-Ang University Hospital EEG)

---

## 1. Background and Rationale

Early detection of cognitive decline—particularly at the mild cognitive impairment (MCI) stage—is critical for intervention and care planning. Electroencephalography (EEG) offers a non-invasive, low-cost screening method, but existing EEG biomarkers show limited sensitivity to MCI.

In preliminary analysis of the OpenNeuro ds004504 dataset (Miltiadous et al., 2023; N=88), we identified a novel pattern in dementia patients: opposite-direction complexity changes in the alpha (8-12 Hz) and theta (4-8 Hz) bands. Specifically, patients showed decreased theta complexity (p=0.025) and increased alpha complexity (p=0.004) compared to healthy controls. A ratio of these measures (alpha/theta entropy) improved discrimination (p=0.008).

Critically, this ratio did not correlate with dementia severity (r≈-0.08 with MMSE within patients), suggesting it may index a discrete state change rather than gradual decline. This raises a key question: **does this state change occur at the MCI stage, before frank dementia?**

The preliminary dataset lacked MCI patients, preventing us from testing this hypothesis. The CAUEEG dataset, with its large sample size and well-characterized MCI group, provides an ideal opportunity to determine whether this marker has utility for early detection.

---

## 2. Research Objectives

**Primary Objective:**  
To determine whether the alpha/theta entropy ratio is elevated in MCI patients compared to healthy controls.

**Secondary Objectives:**
1. To replicate the ratio's discrimination of dementia from healthy controls in an independent cohort
2. To assess whether the ratio differs between MCI and dementia (staging vs. detection)
3. To characterize the distribution of ratio scores across diagnostic groups (testing for bimodality consistent with a state-change model)

**Specific Predictions:**
Based on preliminary data showing the ratio does not track severity within dementia patients, we predict:
- The ratio will be elevated in MCI patients relative to controls
- The ratio will not differ significantly between MCI and dementia groups
- This pattern would indicate the marker detects a state transition that occurs early in cognitive decline

---

## 3. Study Design

**Design Type:** Cross-sectional observational study (secondary analysis of existing data)

**Comparison Groups:**
- Normal controls
- Mild Cognitive Impairment (MCI) — *primary group of interest*
- Dementia

**Key Comparisons:**
1. MCI vs. Normal (primary analysis — early detection)
2. Dementia vs. Normal (replication of preliminary findings)
3. MCI vs. Dementia (staging: does the ratio differ between stages?)

---

## 4. Methods

### 4.1 Data Selection

From the CAUEEG dataset, we will include all subjects with diagnostic labels of "Normal," "MCI," or "Dementia" who have resting-state EEG recordings of sufficient quality for analysis.

**Exclusion criteria:**
- Recordings with excessive artifact contamination (>50% of epochs rejected)
- Incomplete electrode coverage
- Comorbid neurological conditions (e.g., epilepsy) if annotated

### 4.2 EEG Processing

All processing will use open-source tools (MNE-Python, Antropy) to ensure reproducibility.

1. **Preprocessing:**
   - Bandpass filter: 1-45 Hz
   - Notch filter: 50/60 Hz (line noise)
   - Re-reference to common average (consistent with CAUEEG format)
   - Artifact rejection via amplitude threshold and visual inspection

2. **Frequency Band Extraction:**
   - Theta: 4-8 Hz
   - Alpha: 8-12 Hz
   - Additional bands (delta, beta, gamma) for control analyses

3. **Complexity Measurement:**
   - Primary metric: Lempel-Ziv Complexity (LZC)
   - Validation metrics: Permutation Entropy, Sample Entropy
   - Calculated per channel, then averaged by region

### 4.3 Outcome Measures

**Primary outcome:**
- Alpha/Theta Entropy Ratio: MCI vs. Normal (AUC, sensitivity, specificity)

**Secondary outcomes:**
- Alpha/Theta Entropy Ratio: Dementia vs. Normal (replication of preliminary finding)
- Alpha/Theta Entropy Ratio: MCI vs. Dementia (does ratio differ between stages?)
- Distribution of ratio scores across groups (testing for bimodality)
- Correlation between ratio and cognitive scores within MCI group

### 4.4 Statistical Analysis

- **Group comparisons:** Kruskal-Wallis test with Dunn's post-hoc correction
- **Effect sizes:** Cohen's d for pairwise comparisons
- **Classification:** Receiver Operating Characteristic (ROC) analysis with leave-one-out cross-validation
- **Correlation:** Spearman correlation between entropy measures and available cognitive scores
- **Covariates:** Age and sex will be examined as potential confounders

**Significance threshold:** p < 0.05 (two-tailed), with correction for multiple comparisons where appropriate.

---

## 5. Expected Outcomes and Significance

**Scenario A: Ratio elevated in MCI (primary hypothesis supported)**
- The marker detects cognitive impairment at the MCI stage, before dementia onset
- This would support development as an early screening tool
- Clinical significance: EEG-based screening could identify at-risk individuals for earlier intervention

**Scenario B: Ratio elevated in Dementia but not MCI**
- The state change occurs later in disease progression
- Marker would have diagnostic but not early detection utility
- Still valuable for confirming/ruling out dementia in symptomatic patients

**Scenario C: Ratio elevated in both MCI and Dementia at similar levels**
- Supports the "state change" model—transition happens early and persists
- Strong evidence for early detection utility
- Would motivate longitudinal studies to identify the transition point

**Scenario D: Findings do not replicate**
- Preliminary results were dataset-specific or population-specific
- Differences between cohorts would be systematically examined
- Negative results would be transparently reported

Regardless of outcome, this study will clarify the potential clinical utility of band-specific entropy measures for early detection of cognitive impairment.

---

## 6. Data Security and Ethical Considerations

- All data will be stored on encrypted, password-protected local storage
- No attempt will be made to re-identify individual participants
- Data will be used solely for the stated research purposes
- Data will not be shared with third parties
- Upon completion of the study, raw data files will be deleted; only aggregate results will be retained

---

## 7. Dissemination Plan

Results will be submitted for peer-reviewed publication regardless of outcome (positive or negative). The CAUEEG dataset and the original authors (Kim et al., 2023) will be acknowledged and cited per the dataset's requirements.

**Target journals:**
- Clinical Neurophysiology
- Journal of Alzheimer's Disease
- NeuroImage: Clinical

Analysis code will be made publicly available via GitHub upon publication.

---

## 8. Timeline

| Phase | Activity | Duration |
|-------|----------|----------|
| 1 | Data download and quality assessment | 1 week |
| 2 | Preprocessing and entropy calculation | 2 weeks |
| 3 | Statistical analysis and validation | 2 weeks |
| 4 | Manuscript preparation | 4 weeks |

**Estimated total duration:** 9 weeks from data access

---

## 9. Investigator Qualifications

Victor Edmonds is an independent researcher with experience in signal processing and biomarker development. Preliminary work on this project has been conducted using publicly available data (OpenNeuro ds004504), demonstrating technical competence in EEG analysis and complexity measures.

---

## 10. References

Kim, M. J., Youn, Y. C., & Paik, J. (2023). Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia: Algorithms and dataset. *NeuroImage*, 272, 120054.

Miltiadous, A., Tzimourta, K. D., Afrantou, T., et al. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8(6), 95.

Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. *IEEE Transactions on Information Theory*, 22(1), 75-81.

---

## Contact Information

**Victor Edmonds**  
Final Stop Consulting  
Email: [your email]  
Phone: [your phone]

---

*I confirm that this research will be conducted in accordance with ethical principles and that the CAUEEG dataset will be used solely for academic research purposes.*

Signature: ________________________  
Date: ________________________
