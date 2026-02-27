"""
CAUEEG Entropy Analysis — Main Pipeline
========================================
Processes all CAUEEG EDF files to compute entropy, complexity, and spectral
power measures on eyes-closed, artifact-excluded EEG segments.

Pipeline:
  1. Eyes-closed extraction using event markers
  2. Artifact exclusion (±1s buffer around marked events)
  3. Bandpass filtering: alpha (8-12 Hz) and theta (4-8 Hz)
  4. Per-segment entropy computation with weighted averaging
  5. Spectral power via Welch PSD (absolute/relative alpha, theta, ratio)
  6. Statistical summary with group comparisons, AUCs, age correction

Entropy measures:
  - Permutation entropy: 4 parameterizations (o5d5, o3d10, o7d3, o3d1)
  - Sample entropy: order=2, Chebyshev metric, r=0.2*SD
  - Lempel-Ziv complexity: median-threshold, normalized

Usage:
    python caueeg_entropy_v3.py

Requirements: pip install -r requirements.txt
Runtime: ~60-90 minutes (1,388 EDF files)
Output: data_results/caueeg_entropy_v3.csv, data_results/caueeg_entropy_v3_summary.txt
"""

import mne
import numpy as np
import pandas as pd
import antropy as ant
from pathlib import Path
from scipy import stats
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
import json
import math
import warnings
import time
import sys
import os

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# --- CONFIGURATION ---
# Search for dataset relative to both CWD and script location
SCRIPT_DIR = Path(__file__).parent
CANDIDATES = [
    Path("caueeg-dataset"),
    Path("old/caueeg-dataset"),
    Path("../old/caueeg-dataset"),
    SCRIPT_DIR.parent / "caueeg-dataset",
    SCRIPT_DIR.parent.parent / "caueeg-dataset",
]
DATA_DIR = None
for c in CANDIDATES:
    if c.exists():
        DATA_DIR = c
        break

if DATA_DIR is None:
    print("ERROR: Cannot find caueeg-dataset/ directory.")
    print("Searched:", [str(c) for c in CANDIDATES])
    print("Place the CAUEEG dataset in one of these locations, or run from the project root.")
    sys.exit(1)

ANNOTATION_FILE = DATA_DIR / "annotation.json"
EVENT_DIR = DATA_DIR / "event"

# Output
OUTPUT_DIR = Path(__file__).parent.parent / "data_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "caueeg_entropy_v3.csv"
SUMMARY_FILE = OUTPUT_DIR / "caueeg_entropy_v3_summary.txt"

# --- PE PARAMETER SETS ---
PE_CONFIGS = {
    'pe_o5d5': {'order': 5, 'delay': 5},   # PRIMARY: 100ms, 120 states
    'pe_o3d10': {'order': 3, 'delay': 10},  # 100ms, 6 states (coarse)
    'pe_o7d3': {'order': 7, 'delay': 3},    # 90ms, 5040 states (rich)
    'pe_o3d1': {'order': 3, 'delay': 1},    # ORIGINAL: 10ms, 6 states
}

# Minimum samples needed for each PE config (order * delay + some margin)
PE_MIN_SAMPLES = {}
for name, cfg in PE_CONFIGS.items():
    # Need at least order * delay samples for one pattern, but practically
    # need many more for a meaningful entropy estimate
    PE_MIN_SAMPLES[name] = max(cfg['order'] * cfg['delay'] * 10, 200)

# --- ARTIFACT EVENT LABELS ---
ARTIFACT_LABELS = {
    'artifact', 'Move', 'chewing', 'swallowing', 'Talk', 'cough', 'COUGH',
    'eye blinking -->>', 'eyelid move', 'Drowsy ++', 'Seizure', 'sleep Try',
}

EC_END_LABELS = {'Eyes Open', 'Paused', 'Photic Off', 'Recording Resumed'}
EC_END_PREFIXES = ('Photic On', 'HV -')


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def get_group(symptom_list):
    """Map CAUEEG symptom labels to analysis groups."""
    symptoms = set(symptom_list)
    if 'dementia' in symptoms or 'ad' in symptoms or 'load' in symptoms:
        return 'Dementia'
    elif 'mci' in symptoms or any('mci' in s for s in symptoms):
        return 'MCI'
    elif 'normal' in symptoms or 'cb_normal' in symptoms:
        return 'Normal'
    return None


def get_dementia_subtype(symptom_list):
    """Extract dementia subtype if available."""
    symptoms = set(symptom_list)
    if 'ad' in symptoms:
        return 'AD'
    elif 'vd' in symptoms or 'vascular' in symptoms:
        return 'VaD'
    elif 'ftd' in symptoms:
        return 'FTD'
    elif 'dementia' in symptoms or 'load' in symptoms:
        return 'Dementia_other'
    return None


def is_artifact_event(label):
    """Check if an event label indicates an artifact."""
    return label in ARTIFACT_LABELS


def is_ec_end_event(label):
    """Check if event ends a eyes-closed segment."""
    if label in EC_END_LABELS:
        return True
    for prefix in EC_END_PREFIXES:
        if label.startswith(prefix):
            return True
    return False


def extract_eyes_closed_segments(events, sfreq, total_samples):
    """
    Extract eyes-closed segments from event list, excluding artifacts.
    Returns list of (start_sample, end_sample) tuples for clean EC data.
    """
    ec_segments = []
    ec_start = None

    events_sorted = sorted(events, key=lambda x: x[0])

    for sample, label in events_sorted:
        if label == 'Eyes Closed':
            ec_start = sample
        elif is_ec_end_event(label) and ec_start is not None:
            ec_segments.append((ec_start, sample))
            ec_start = None

    if ec_start is not None:
        ec_segments.append((ec_start, total_samples))

    if not ec_segments:
        return []

    # Exclude artifact-contaminated intervals (±1s buffer)
    artifact_buffer = int(1.0 * sfreq)
    artifact_intervals = []
    for sample, label in events_sorted:
        if is_artifact_event(label):
            artifact_intervals.append((sample - artifact_buffer, sample + artifact_buffer))

    if artifact_intervals:
        artifact_intervals.sort()
        merged = [artifact_intervals[0]]
        for start, end in artifact_intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        artifact_intervals = merged

    clean_segments = []
    for ec_s, ec_e in ec_segments:
        remaining = [(ec_s, ec_e)]
        for art_s, art_e in artifact_intervals:
            new_remaining = []
            for seg_s, seg_e in remaining:
                if art_e <= seg_s or art_s >= seg_e:
                    new_remaining.append((seg_s, seg_e))
                else:
                    if art_s > seg_s:
                        new_remaining.append((seg_s, art_s))
                    if art_e < seg_e:
                        new_remaining.append((art_e, seg_e))
            remaining = new_remaining
        clean_segments.extend(remaining)

    min_samples = int(2.0 * sfreq)
    clean_segments = [(s, e) for s, e in clean_segments if (e - s) >= min_samples]

    return clean_segments


def collect_segments(raw, segments, max_seconds=60):
    """
    Extract clean segments as a LIST of arrays (not concatenated).
    Each element is (n_channels, n_samples_in_segment).
    Returns list of arrays and total seconds collected.
    """
    sfreq = raw.info['sfreq']
    max_samples = int(max_seconds * sfreq)
    total_samples = int(raw.n_times)

    collected = []
    total_collected = 0

    for start, end in segments:
        start = max(0, int(start))
        end = min(total_samples, int(end))
        if start >= end:
            continue

        needed = max_samples - total_collected
        if needed <= 0:
            break

        take = min(end - start, needed)
        chunk = raw.get_data(start=start, stop=start + take)
        collected.append(chunk)
        total_collected += take

    return collected, total_collected / sfreq


def compute_pe(signal, order=3, delay=1):
    """Permutation Entropy via antropy."""
    try:
        return ant.perm_entropy(signal, order=order, delay=delay, normalize=True)
    except Exception:
        return np.nan


def compute_se(signal, order=2, metric='chebyshev'):
    """Sample Entropy via antropy."""
    try:
        return ant.sample_entropy(signal, order=order, metric=metric)
    except Exception:
        return np.nan


def compute_lzc(signal):
    """Lempel-Ziv Complexity via antropy."""
    try:
        return ant.lziv_complexity(signal > np.median(signal), normalize=True)
    except Exception:
        return np.nan


def weighted_mean_segments(segments, func):
    """
    Apply func to each segment, return weighted average by segment length.
    segments: list of 1D arrays (single channel each).
    func: callable that takes a 1D array and returns a scalar.
    """
    values = []
    weights = []
    for seg in segments:
        if len(seg) < 400:  # Skip segments < 2s (avoids finite-sample bias + filter edge artifacts)
            continue
        val = func(seg)
        if not np.isnan(val):
            values.append(val)
            weights.append(len(seg))
    if not values:
        return np.nan
    weights = np.array(weights, dtype=float)
    values = np.array(values)
    return np.average(values, weights=weights)


def compute_spectral_power(segments, sfreq):
    """
    Compute absolute and relative spectral power from unfiltered EC segments.
    Returns dict with alpha/theta absolute power, relative power, and ratio.
    """
    # Concatenate for PSD (concatenation is fine for Welch — it uses
    # overlapping windows and discontinuities at boundaries are negligible
    # relative to the many windows in 60s of data)
    if not segments:
        return {}

    all_data = np.concatenate(segments, axis=1)  # (n_channels, n_samples)

    alpha_abs_list = []
    theta_abs_list = []
    total_power_list = []

    for ch_data in all_data:
        # Welch PSD: 2-second windows, 50% overlap
        nperseg = min(int(2.0 * sfreq), len(ch_data))
        if nperseg < int(0.5 * sfreq):
            continue
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)

        # Band masks
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        theta_mask = (freqs >= 4) & (freqs < 8)
        total_mask = (freqs >= 1) & (freqs <= 45)

        alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])
        theta_power = np.trapz(psd[theta_mask], freqs[theta_mask])
        total_power = np.trapz(psd[total_mask], freqs[total_mask])

        alpha_abs_list.append(alpha_power)
        theta_abs_list.append(theta_power)
        total_power_list.append(total_power)

    if not alpha_abs_list:
        return {}

    alpha_abs = np.mean(alpha_abs_list)
    theta_abs = np.mean(theta_abs_list)
    total = np.mean(total_power_list)

    alpha_rel = alpha_abs / total if total > 0 else np.nan
    theta_rel = theta_abs / total if total > 0 else np.nan
    power_ratio = alpha_rel / theta_rel if theta_rel > 0 else np.nan

    return {
        'Alpha_Power_Abs': alpha_abs,
        'Theta_Power_Abs': theta_abs,
        'Alpha_Power_Rel': alpha_rel,
        'Theta_Power_Rel': theta_rel,
        'Power_Ratio': power_ratio,
    }


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0


def compute_auc(labels, scores):
    """Compute AUC, handling direction (flips if AUC < 0.5)."""
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)  # Direction-corrected
    except Exception:
        return np.nan


# =====================================================================
# MAIN PROCESSING LOOP
# =====================================================================

print("=" * 70)
print("CAUEEG ENTROPY ANALYSIS v3 — FINAL PIPELINE")
print("=" * 70)
print(f"Data directory: {DATA_DIR}")
print(f"Output: {OUTPUT_CSV}")
print()
print("v3 improvements over v2:")
print("  1. Per-segment entropy (no concatenation boundary artifacts)")
print("  2. Spectral power via Welch PSD (alpha, theta, relative, ratio)")
print("  3. AUCs computed in summary")
print("  (Plus all v2 corrections: EC extraction, artifact rejection, 4 PE params)")
print()

# Load annotations
with open(ANNOTATION_FILE, 'r') as f:
    raw_annotations = json.load(f)

annotations = {item['serial']: item for item in raw_annotations['data']}
print(f"Loaded {len(annotations)} annotation records")

# Process
edf_dir = DATA_DIR / "signal" / "edf"
edf_files = sorted(edf_dir.glob("*.edf"))
print(f"Found {len(edf_files)} EDF files. Processing...\n")

results = []
t_start = time.time()
processed = 0
skipped = 0
skip_reasons = {}

for i, edf_path in enumerate(edf_files):
    serial = edf_path.stem

    if serial not in annotations:
        skipped += 1
        skip_reasons['no_annotation'] = skip_reasons.get('no_annotation', 0) + 1
        continue

    anno = annotations[serial]
    group = get_group(anno.get('symptom', []))

    if group is None:
        skipped += 1
        skip_reasons['no_group'] = skip_reasons.get('no_group', 0) + 1
        continue

    try:
        # Load event file
        event_file = EVENT_DIR / f"{serial}.json"
        if not event_file.exists():
            skipped += 1
            skip_reasons['no_events'] = skip_reasons.get('no_events', 0) + 1
            continue

        with open(event_file) as ef:
            events = json.load(ef)

        # Load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        sfreq = raw.info['sfreq']
        if sfreq != 200.0:
            print(f"\n  WARNING: {serial} has sfreq={sfreq}, expected 200.0")

        # EEG channels only
        eeg_channels = [ch for ch in raw.ch_names
                        if ch not in ['EKG', 'Photic', 'EDF Annotations']]
        raw.pick(eeg_channels)

        total_samples = raw.n_times

        # Extract clean eyes-closed segments
        ec_segments = extract_eyes_closed_segments(events, sfreq, total_samples)

        if not ec_segments:
            skipped += 1
            skip_reasons['no_ec_data'] = skip_reasons.get('no_ec_data', 0) + 1
            continue

        total_ec_seconds = sum(e - s for s, e in ec_segments) / sfreq

        # Collect segments as LIST (not concatenated) — up to 60s
        seg_list, actual_seconds = collect_segments(raw, ec_segments, max_seconds=60)

        if not seg_list or actual_seconds < 10:
            skipped += 1
            skip_reasons['too_short'] = skip_reasons.get('too_short', 0) + 1
            continue

        n_channels = seg_list[0].shape[0]
        n_segments = len(seg_list)
        seg_lengths = [s.shape[1] for s in seg_list]

        # ---- SPECTRAL POWER (on unfiltered EC data) ----
        power_results = compute_spectral_power(seg_list, sfreq)

        # ---- BANDPASS FILTER each segment separately ----
        # Filter each segment individually to avoid boundary effects
        info = mne.create_info(
            ch_names=eeg_channels[:n_channels],
            sfreq=sfreq,
            ch_types='eeg'
        )

        alpha_segments = []  # list of (n_channels, n_samples)
        theta_segments = []

        for seg in seg_list:
            # Alpha
            raw_a = mne.io.RawArray(seg.copy(), info, verbose=False)
            raw_a.filter(8, 12, verbose=False)
            alpha_segments.append(raw_a.get_data())

            # Theta
            raw_t = mne.io.RawArray(seg.copy(), info, verbose=False)
            raw_t.filter(4, 8, verbose=False)
            theta_segments.append(raw_t.get_data())

        # ---- COMPUTE ENTROPY MEASURES (per-segment, weighted average) ----
        row = {
            'ID': serial,
            'Group': group,
            'Subtype': get_dementia_subtype(anno.get('symptom', [])),
            'Age': anno.get('age', np.nan),
            'EC_seconds': round(actual_seconds, 1),
            'EC_total_available': round(total_ec_seconds, 1),
            'n_segments': n_segments,
            'sfreq': sfreq,
            'n_channels': n_channels,
        }

        # Add spectral power
        row.update(power_results)

        # LZC — per channel, per segment, weighted average
        lzc_alpha_per_ch = []
        lzc_theta_per_ch = []
        for ch_idx in range(n_channels):
            ch_segs_a = [seg[ch_idx] for seg in alpha_segments]
            ch_segs_t = [seg[ch_idx] for seg in theta_segments]
            lzc_alpha_per_ch.append(weighted_mean_segments(ch_segs_a, compute_lzc))
            lzc_theta_per_ch.append(weighted_mean_segments(ch_segs_t, compute_lzc))

        row['LZC_Alpha'] = np.nanmean(lzc_alpha_per_ch)
        row['LZC_Theta'] = np.nanmean(lzc_theta_per_ch)
        row['LZC_Ratio'] = (row['LZC_Alpha'] / row['LZC_Theta']
                            if row['LZC_Theta'] > 0 else np.nan)

        # SE — per channel, per segment, weighted average
        se_alpha_per_ch = []
        se_theta_per_ch = []
        for ch_idx in range(n_channels):
            ch_segs_a = [seg[ch_idx] for seg in alpha_segments]
            ch_segs_t = [seg[ch_idx] for seg in theta_segments]
            se_alpha_per_ch.append(weighted_mean_segments(ch_segs_a, compute_se))
            se_theta_per_ch.append(weighted_mean_segments(ch_segs_t, compute_se))

        row['SE_Alpha'] = np.nanmean(se_alpha_per_ch)
        row['SE_Theta'] = np.nanmean(se_theta_per_ch)
        row['SE_Ratio'] = (row['SE_Alpha'] / row['SE_Theta']
                           if row['SE_Theta'] > 0 else np.nan)

        # PE — per channel, per segment, weighted average, for each config
        for config_name, config in PE_CONFIGS.items():
            order = config['order']
            delay = config['delay']
            min_samp = PE_MIN_SAMPLES[config_name]

            pe_alpha_per_ch = []
            pe_theta_per_ch = []

            for ch_idx in range(n_channels):
                # Filter out segments too short for this PE config
                ch_segs_a = [seg[ch_idx] for seg in alpha_segments
                             if seg.shape[1] >= min_samp]
                ch_segs_t = [seg[ch_idx] for seg in theta_segments
                             if seg.shape[1] >= min_samp]

                pe_func = lambda s, o=order, d=delay: compute_pe(s, order=o, delay=d)
                pe_alpha_per_ch.append(weighted_mean_segments(ch_segs_a, pe_func))
                pe_theta_per_ch.append(weighted_mean_segments(ch_segs_t, pe_func))

            pe_a = np.nanmean(pe_alpha_per_ch)
            pe_t = np.nanmean(pe_theta_per_ch)

            row[f'{config_name}_Alpha'] = pe_a
            row[f'{config_name}_Theta'] = pe_t
            row[f'{config_name}_Ratio'] = pe_a / pe_t if pe_t > 0 else np.nan

        results.append(row)
        processed += 1

        elapsed = time.time() - t_start
        rate = processed / elapsed
        remaining = (len(edf_files) - i - 1) / rate if rate > 0 else 0

        pe_primary = row.get('pe_o5d5_Alpha', 0)
        pe_orig = row.get('pe_o3d1_Alpha', 0)
        pr = row.get('Power_Ratio', 0)
        print(f"  [{processed:4d}/{len(edf_files)}] {serial} ({group}) "
              f"| PE_o5d5={pe_primary:.4f} PE_orig={pe_orig:.4f} PwrR={pr:.3f} "
              f"| EC={actual_seconds:.0f}s ({n_segments}seg) "
              f"| ETA: {remaining / 60:.0f}min", end='\r')
        sys.stdout.flush()

    except Exception as e:
        print(f"\n  ERROR {serial}: {e}")
        skipped += 1
        skip_reasons['error'] = skip_reasons.get('error', 0) + 1
        continue

# =====================================================================
# SAVE RESULTS
# =====================================================================

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

elapsed_total = time.time() - t_start
print(f"\n\nProcessed: {processed}, Skipped: {skipped}")
print(f"Skip reasons: {skip_reasons}")
print(f"Time: {elapsed_total / 60:.1f} minutes")
print(f"Saved to: {OUTPUT_CSV}")


# =====================================================================
# SUMMARY ANALYSIS
# =====================================================================

summary_lines = []

def log(msg):
    print(msg)
    summary_lines.append(msg)


log("\n" + "=" * 70)
log("CAUEEG ENTROPY v3 — RESULTS SUMMARY")
log("=" * 70)

log(f"\nN = {len(df)}")
log(f"Groups: {df['Group'].value_counts().to_dict()}")
log(f"\nPipeline (v3 final):")
log(f"  - Eyes-closed extraction: yes (event markers)")
log(f"  - Artifact exclusion: yes (±1s buffer around marked events)")
log(f"  - Entropy computation: per-segment, weighted average by length")
log(f"  - Spectral power: Welch PSD (2s windows, 50% overlap)")
log(f"  - Sampling frequency: {df['sfreq'].unique()} Hz")
log(f"  - Filter: MNE FIR (firwin), zero-phase")
log(f"  - EC data per subject: mean={df['EC_seconds'].mean():.1f}s, "
    f"min={df['EC_seconds'].min():.1f}s, max={df['EC_seconds'].max():.1f}s")
log(f"  - Segments per subject: mean={df['n_segments'].mean():.1f}, "
    f"min={df['n_segments'].min()}, max={df['n_segments'].max()}")

# --- Demographics ---
log("\n--- DEMOGRAPHICS ---")
for g in ['Normal', 'MCI', 'Dementia']:
    sub = df[df['Group'] == g]
    log(f"  {g}: n={len(sub)}, age={sub['Age'].mean():.1f} ± {sub['Age'].std():.1f}")

# --- PE PARAMETER COMPARISON ---
log("\n" + "=" * 70)
log("PE PARAMETER COMPARISON (Alpha band)")
log("=" * 70)

comparisons = [('Dementia', 'Normal'), ('MCI', 'Normal')]

for config_name, config in PE_CONFIGS.items():
    order = config['order']
    delay = config['delay']
    window_ms = (order - 1) * delay / 200 * 1000
    n_states = math.factorial(order)

    log(f"\n--- {config_name}: order={order}, delay={delay}, "
        f"window={window_ms:.0f}ms, states={n_states} ---")

    col = f'{config_name}_Alpha'
    for g1, g2 in comparisons:
        v1 = df[df['Group'] == g1][col].dropna().values
        v2 = df[df['Group'] == g2][col].dropna().values
        if len(v1) > 5 and len(v2) > 5:
            t, p = stats.ttest_ind(v1, v2)
            d = cohens_d(v1, v2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            log(f"  {g1} vs {g2}: d={d:.3f}, t={t:.3f}, p={p:.6f} {sig}")

# --- HEAD-TO-HEAD: ALL MEASURES ---
log("\n" + "=" * 70)
log("HEAD-TO-HEAD: ALL MEASURES (v3 Final Pipeline)")
log("=" * 70)

all_measures = [
    ('pe_o5d5_Alpha', 'PE_Alpha (o5d5, PRIMARY)'),
    ('pe_o3d1_Alpha', 'PE_Alpha (o3d1, ORIGINAL)'),
    ('pe_o7d3_Alpha', 'PE_Alpha (o7d3)'),
    ('pe_o3d10_Alpha', 'PE_Alpha (o3d10)'),
    ('LZC_Alpha', 'LZC_Alpha'),
    ('SE_Alpha', 'SE_Alpha'),
    ('LZC_Ratio', 'LZC Ratio'),
    ('pe_o5d5_Ratio', 'PE Ratio (o5d5)'),
    ('pe_o3d1_Ratio', 'PE Ratio (o3d1, ORIGINAL)'),
    ('SE_Ratio', 'SE Ratio'),
    ('Power_Ratio', 'Power Ratio (rel a/t)'),
    ('Alpha_Power_Rel', 'Alpha Power (relative)'),
    ('Theta_Power_Rel', 'Theta Power (relative)'),
]

log(f"\n{'Measure':<30} {'Comparison':<20} {'d':>7} {'t':>8} {'p':>10} {'AUC':>6}")
log("-" * 85)

for col, label in all_measures:
    if col not in df.columns:
        continue
    for g1, g2 in comparisons:
        mask1 = df['Group'] == g1
        mask2 = df['Group'] == g2
        v1 = df.loc[mask1, col].dropna().values
        v2 = df.loc[mask2, col].dropna().values
        if len(v1) > 5 and len(v2) > 5:
            t, p = stats.ttest_ind(v1, v2)
            d = cohens_d(v1, v2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            # AUC
            sub = df[mask1 | mask2][[col, 'Group']].dropna()
            labels_bin = (sub['Group'] == g1).astype(int).values
            scores = sub[col].values
            auc = compute_auc(labels_bin, scores)

            log(f"  {label:<28} {g1+' vs '+g2:<18} {d:>7.3f} {t:>8.3f} "
                f"{p:>10.6f} {auc:>5.3f} {sig}")

# --- AGE CORRECTION ---
log("\n" + "=" * 70)
log("AGE CORRECTION (primary measures)")
log("=" * 70)

measures_to_correct = [
    'pe_o5d5_Alpha', 'pe_o3d1_Alpha', 'LZC_Ratio', 'SE_Alpha',
    'pe_o5d5_Ratio', 'pe_o3d1_Ratio', 'Power_Ratio', 'Alpha_Power_Rel',
]

age = df['Age'].values
X_age = np.column_stack([np.ones(len(age)), age])
XtX_inv = np.linalg.pinv(X_age.T @ X_age)

log(f"\n{'Measure':<25} {'Comparison':<20} {'d_raw':>7} {'d_ageR':>7} "
    f"{'p_ageR':>10} {'AUC_ageR':>8}")
log("-" * 85)

for col in measures_to_correct:
    if col not in df.columns:
        continue

    vals = df[col].values
    valid = ~np.isnan(vals)
    betas = XtX_inv @ X_age[valid].T @ vals[valid]
    resid = np.full_like(vals, np.nan)
    resid[valid] = vals[valid] - X_age[valid] @ betas
    df[f'{col}_age_resid'] = resid

    for g1, g2 in comparisons:
        mask1 = (df['Group'] == g1).values
        mask2 = (df['Group'] == g2).values

        v1_raw = vals[mask1 & valid]
        v2_raw = vals[mask2 & valid]
        d_raw = cohens_d(v1_raw, v2_raw) if len(v1_raw) > 5 and len(v2_raw) > 5 else np.nan

        v1_r = resid[mask1 & valid]
        v2_r = resid[mask2 & valid]
        if len(v1_r) > 5 and len(v2_r) > 5:
            d_r = cohens_d(v1_r, v2_r)
            t_r, p_r = stats.ttest_ind(v1_r, v2_r)
            sig = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else ""

            # AUC on residuals
            sub_mask = (mask1 | mask2) & valid
            if sub_mask.sum() > 10:
                labels_bin = mask1[sub_mask].astype(int)
                scores_r = resid[sub_mask]
                auc_r = compute_auc(labels_bin, scores_r)
            else:
                auc_r = np.nan
        else:
            d_r, p_r, sig, auc_r = np.nan, np.nan, "", np.nan

        label = col.replace('_Alpha', '').replace('_Ratio', '_R').replace('_Power_Rel', '_Pwr')
        log(f"  {label:<23} {g1+' vs '+g2:<18} {d_raw:>7.3f} {d_r:>7.3f} "
            f"{p_r:>10.6f} {auc_r:>7.3f} {sig}")

# --- AGE-MATCHED SUBGROUP (70-80) ---
log("\n" + "=" * 70)
log("AGE-MATCHED SUBGROUP (ages 70-80)")
log("=" * 70)

df_matched = df[(df['Age'] >= 70) & (df['Age'] <= 80)]
log(f"\nn = {len(df_matched)}")
for g in ['Normal', 'MCI', 'Dementia']:
    sub = df_matched[df_matched['Group'] == g]
    log(f"  {g}: n={len(sub)}, age={sub['Age'].mean():.1f} ± {sub['Age'].std():.1f}")

log(f"\n{'Measure':<25} {'Comparison':<20} {'d':>7} {'p':>10} {'AUC':>6}")
log("-" * 72)

for col in ['pe_o5d5_Alpha', 'pe_o3d1_Alpha', 'LZC_Ratio', 'SE_Alpha',
            'Power_Ratio', 'Alpha_Power_Rel']:
    if col not in df_matched.columns:
        continue
    for g1, g2 in comparisons:
        v1 = df_matched[df_matched['Group'] == g1][col].dropna().values
        v2 = df_matched[df_matched['Group'] == g2][col].dropna().values
        if len(v1) > 5 and len(v2) > 5:
            t, p = stats.ttest_ind(v1, v2)
            d = cohens_d(v1, v2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            sub = df_matched[df_matched['Group'].isin([g1, g2])][[col, 'Group']].dropna()
            labels_bin = (sub['Group'] == g1).astype(int).values
            scores = sub[col].values
            auc = compute_auc(labels_bin, scores)

            label = col.replace('_Alpha', '').replace('_Ratio', '_R').replace('_Power_Rel', '_Pwr')
            log(f"  {label:<23} {g1+' vs '+g2:<18} {d:>7.3f} {p:>10.6f} {auc:>5.3f} {sig}")


# --- CORRELATION BETWEEN PE PARAMETERIZATIONS ---
log("\n" + "=" * 70)
log("CORRELATIONS BETWEEN PE PARAMETERIZATIONS (Alpha band)")
log("=" * 70)

pe_cols = [f'{c}_Alpha' for c in PE_CONFIGS.keys()]
pe_cols_present = [c for c in pe_cols if c in df.columns]

header = f"\n{'':>18}" + "".join(f" {c.replace('_Alpha',''):>10}" for c in pe_cols_present)
log(header)

for c1 in pe_cols_present:
    row_str = f"  {c1.replace('_Alpha',''):<16}"
    for c2 in pe_cols_present:
        v1 = df[c1].dropna().values
        v2 = df[c2].dropna().values
        n = min(len(v1), len(v2))
        r = np.corrcoef(v1[:n], v2[:n])[0, 1]
        row_str += f" {r:>10.3f}"
    log(row_str)


# --- CORRELATION: PE vs SPECTRAL POWER ---
log("\n" + "=" * 70)
log("CORRELATIONS: ENTROPY vs SPECTRAL POWER")
log("=" * 70)

corr_pairs = [
    ('pe_o5d5_Alpha', 'Alpha_Power_Rel'),
    ('pe_o3d1_Alpha', 'Alpha_Power_Rel'),
    ('LZC_Alpha', 'Alpha_Power_Rel'),
    ('SE_Alpha', 'Alpha_Power_Rel'),
    ('pe_o5d5_Alpha', 'Power_Ratio'),
    ('pe_o3d1_Alpha', 'Power_Ratio'),
    ('LZC_Ratio', 'Power_Ratio'),
]

for c1, c2 in corr_pairs:
    if c1 in df.columns and c2 in df.columns:
        valid = df[[c1, c2]].dropna()
        if len(valid) > 10:
            r = np.corrcoef(valid[c1], valid[c2])[0, 1]
            log(f"  {c1:<20} vs {c2:<20}: r = {r:+.3f} (shared var = {r**2:.1%})")


# --- PIPELINE DOCUMENTATION ---
log("\n" + "=" * 70)
log("PIPELINE DOCUMENTATION")
log("=" * 70)
log(f"""
Sampling frequency: 200 Hz (consistent across all CAUEEG EDF files)
Filter: MNE default FIR (firwin design), zero-phase application
  - Equivalent to scipy.signal.filtfilt with FIR coefficients
  - No phase distortion
  - Alpha band: 8-12 Hz bandpass
  - Theta band: 4-8 Hz bandpass

Entropy computation: PER-SEGMENT with weighted averaging
  - Each clean eyes-closed segment is filtered independently
  - Entropy computed within each segment (no cross-boundary artifacts)
  - Segment-level values combined via weighted average (weight = segment length)
  - Segments shorter than 400 samples (2s at 200 Hz) excluded from entropy computation
  - This addresses the concatenation boundary artifact identified in review

Spectral power: Welch PSD on unfiltered EC data
  - Window: 2 seconds, 50% overlap (Hann window)
  - Concatenation acceptable for Welch (windowed, overlap handles boundaries)
  - Absolute power: integral of PSD within band (trapz)
  - Relative power: band power / total power (1-45 Hz)
  - Power Ratio: relative alpha / relative theta

PE Primary parameters: order=5, delay=5
  - Embedding window: (5-1) * 5 / 200 = 100 ms = 1 alpha cycle at 10 Hz
  - Number of ordinal states: 5! = 120
  - Sufficient state-space richness for 60s epochs (12,000 samples)

PE Original parameters (for comparison): order=3, delay=1
  - Embedding window: (3-1) * 1 / 200 = 10 ms = 0.1 alpha cycles
  - Number of ordinal states: 3! = 6
  - Note: measures local slope/curvature of bandpass-filtered signal,
    not ordinal complexity at the alpha timescale

PE o3d10: order=3, delay=10
  - Effective sampling rate: 200/10 = 20 Hz (at Nyquist limit for alpha)
  - Phase-aliasing of ordinal patterns makes this a frequency discriminator

PE o7d3: order=7, delay=3
  - Number of ordinal states: 7! = 5,040
  - WARNING: subjects with <12s of EC data have ~2,400 samples
    insufficient to populate 5,040-bin distribution (finite-sample bias)

Eyes-closed extraction:
  - Event markers 'Eyes Closed' and 'Eyes Open' used to delimit segments
  - Segments also terminated by photic stimulation and HV events
  - Up to 60s of clean EC data extracted per recording

Artifact exclusion:
  - Events labeled as artifact, Move, chewing, swallowing, Talk, cough,
    eye blinking, eyelid move, Drowsy, Seizure, sleep Try
  - ±1 second buffer around each artifact marker
  - Segments < 2 seconds excluded after artifact removal
""")

log("=" * 70)
log("DONE — Review caueeg_entropy_v3.csv for full results")
log("=" * 70)

# Save summary
with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_lines))
print(f"\nSummary saved to: {SUMMARY_FILE}")
