"""
Sample Entropy Tolerance Sensitivity Test
==========================================
Tests whether SE_alpha results are robust across tolerance values (r = 0.15,
0.20, 0.25 × SD). Reports effect sizes, AUCs, and cross-validated combined
model performance for each tolerance to demonstrate that SE is not parameter-
fragile in the way PE is.

Usage: python test_se_tolerance.py
Requires: caueeg_entropy_v3.csv (run caueeg_entropy_v3.py first) AND
          raw CAUEEG EDF files (for recomputation with alternative r values)

Note: This script recomputes SE from raw EEG data because the tolerance
parameter must be applied during entropy computation, not post-hoc. It uses
the same pipeline as caueeg_entropy_v3.py (eyes-closed extraction, artifact
exclusion, per-segment computation, weighted averaging).
"""

import numpy as np
import pandas as pd
import mne
import json
import warnings
import time
import os
import sys
from pathlib import Path
from scipy import stats
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
CANDIDATES = [
    Path("caueeg-dataset"),
    Path("old/caueeg-dataset"),
    Path("../old/caueeg-dataset"),
    SCRIPT_DIR.parent / "caueeg-dataset",
    SCRIPT_DIR.parent.parent / "caueeg-dataset",
]

DATASET_PATH = None
for c in CANDIDATES:
    if c.is_dir() and (c / "annotation.json").exists():
        DATASET_PATH = c
        break

if DATASET_PATH is None:
    print("ERROR: caueeg-dataset/ not found. See README for download instructions.")
    sys.exit(1)

print(f"Using dataset: {DATASET_PATH.resolve()}")

# Also load the precomputed CSV for Power_Ratio (we only recompute SE)
csv_path = SCRIPT_DIR.parent / "data_results" / "caueeg_entropy_v3.csv"
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found. Run caueeg_entropy_v3.py first.")
    sys.exit(1)

df_existing = pd.read_csv(csv_path)
print(f"Loaded precomputed CSV: {len(df_existing)} subjects")

# Tolerance values to test
R_VALUES = [0.15, 0.20, 0.25]

# Signal processing parameters (must match caueeg_entropy_v3.py exactly)
ALPHA_LOW, ALPHA_HIGH = 8, 12
EEG_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5', 'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
]
ARTIFACT_EVENTS = {'artifact', 'Move', 'chewing', 'swallowing', 'Talk', 'cough', 'eye blinking'}
ARTIFACT_BUFFER = 1.0
EC_START_PREFIXES = ('Eyes Closed',)
EC_END_PREFIXES = ('Eyes Open', 'Photic On', 'HV -', 'Paused', 'Stop Recording')
MIN_SEGMENT_SECONDS = 2.0
MIN_SEGMENT_SAMPLES = 400


# ============================================================
# Helper functions (copied from caueeg_entropy_v3.py)
# ============================================================

def parse_events(event_path):
    """Parse event JSON file, return list of (sample_idx, label)."""
    with open(event_path, 'r') as f:
        events = json.load(f)
    parsed = []
    for ev in events:
        if isinstance(ev, (list, tuple)) and len(ev) >= 2:
            parsed.append((int(ev[0]), str(ev[1])))
        elif isinstance(ev, dict):
            idx = int(ev.get('sample', ev.get('onset', 0)))
            label = str(ev.get('label', ev.get('description', '')))
            parsed.append((idx, label))
    return parsed


def get_eyes_closed_intervals(events, total_samples, sfreq):
    """Extract eyes-closed intervals from events, excluding artifact regions."""
    ec_intervals = []
    artifact_zones = []

    for idx, label in events:
        if any(label.startswith(p) for p in EC_START_PREFIXES):
            ec_intervals.append([idx, total_samples])
        elif any(label.startswith(p) for p in EC_END_PREFIXES):
            if ec_intervals and ec_intervals[-1][1] == total_samples:
                ec_intervals[-1][1] = idx
        if label.strip().lower() in {a.lower() for a in ARTIFACT_EVENTS}:
            buf = int(ARTIFACT_BUFFER * sfreq)
            artifact_zones.append((max(0, idx - buf), min(total_samples, idx + buf)))

    # Subtract artifact zones from EC intervals
    clean = []
    for start, end in ec_intervals:
        segments = [(start, end)]
        for az_start, az_end in artifact_zones:
            new_segments = []
            for s, e in segments:
                if az_end <= s or az_start >= e:
                    new_segments.append((s, e))
                else:
                    if s < az_start:
                        new_segments.append((s, az_start))
                    if az_end < e:
                        new_segments.append((az_end, e))
            segments = new_segments
        clean.extend(segments)

    min_samples = int(MIN_SEGMENT_SECONDS * sfreq)
    clean = [(s, e) for s, e in clean if (e - s) >= min_samples]
    return clean


def collect_ec_data(raw, intervals, max_seconds=60):
    """Collect eyes-closed data from intervals up to max_seconds total."""
    sfreq = raw.info['sfreq']
    needed = int(max_seconds * sfreq)
    collected = []
    total_collected = 0

    for start, end in intervals:
        if total_collected >= needed:
            break
        take = min(end - start, needed - total_collected)
        chunk = raw.get_data(start=start, stop=start + take)
        collected.append(chunk)
        total_collected += take

    return collected, total_collected / sfreq


def compute_se_with_r(signal, order=2, r_factor=0.2):
    """
    Compute sample entropy with a specific tolerance factor.
    r = r_factor * SD of the signal.
    Uses antropy's fast KDTree implementation with explicit tolerance.
    """
    try:
        import antropy as ant
        r_abs = r_factor * np.std(signal, ddof=1)
        if r_abs == 0:
            return np.nan
        # Try using the tolerance parameter (antropy >= 0.1.6)
        try:
            return ant.sample_entropy(signal, order=order, metric='chebyshev',
                                      tolerance=r_abs)
        except TypeError:
            # Older antropy without tolerance parameter — use workaround:
            # antropy internally does r = 0.2 * std(x).
            # We want r = r_factor * std(original_x).
            # Since SE is NOT scale-invariant when tolerance is computed
            # from the transformed signal, we need to normalize x to unit
            # variance first, then antropy will use r = 0.2 * 1.0 = 0.2,
            # and we want r_factor * 1.0 = r_factor.
            # So scale x such that 0.2 * std(scaled) = r_factor * std(original)
            # With x_norm = x / std(x): std(x_norm) = 1
            # antropy uses r = 0.2 * 1 = 0.2
            # We want r_factor relative to original std=1, so we need:
            # 0.2 * std(scaled) = r_factor => std(scaled) = r_factor / 0.2
            # scaled = x_norm * (r_factor / 0.2)
            # But then distances also scale by (r_factor/0.2) and r scales same way
            # => dist * k < 0.2 * k => dist < 0.2 -- SAME. Scale-invariant!
            #
            # The only way is to pass an absolute tolerance. If antropy
            # doesn't support it, we must patch or use nolds/neurokit2.
            # As a last resort, use a pure-numpy vectorized version:
            return _sample_entropy_vectorized(signal, m=order, r=r_abs)
    except Exception:
        return np.nan


def _sample_entropy_vectorized(x, m=2, r=None):
    """
    Vectorized sample entropy using KDTree (Chebyshev metric).
    Much faster than naive nested loops.
    """
    from sklearn.neighbors import KDTree
    N = len(x)
    if N < m + 2 or r is None or r == 0:
        return np.nan

    def _count_matches_kdtree(template_len):
        templates = np.array([x[i:i + template_len] for i in range(N - template_len)])
        tree = KDTree(templates, metric='chebyshev')
        # Count neighbors within radius r (including self)
        counts = tree.query_radius(templates, r=r, count_only=True)
        # Subtract self-matches and divide by 2 (each pair counted once)
        total = (np.sum(counts) - len(templates)) / 2
        return total

    B = _count_matches_kdtree(m)
    if B == 0:
        return np.nan
    A = _count_matches_kdtree(m + 1)
    if A == 0:
        return np.nan

    return -np.log(A / B)


def weighted_mean_segments(segments, func):
    """Apply func to each segment, return weighted average by segment length."""
    values = []
    weights = []
    for seg in segments:
        if len(seg) < MIN_SEGMENT_SAMPLES:
            continue
        val = func(seg)
        if not np.isnan(val):
            values.append(val)
            weights.append(len(seg))
    if not values:
        return np.nan
    return np.average(values, weights=np.array(weights, dtype=float))


# ============================================================
# Main processing
# ============================================================

def process_subject(serial, r_values):
    """Process one subject, return SE_Alpha for each r value."""
    edf_path = DATASET_PATH / "signal" / "edf" / f"{serial}.edf"
    event_path = DATASET_PATH / "event" / f"{serial}.json"

    if not edf_path.exists() or not event_path.exists():
        return {f"SE_Alpha_r{int(r*100):02d}": np.nan for r in r_values}

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    # Map channel names
    ch_map = {}
    for ch in raw.ch_names:
        base = ch.split('-')[0].strip().upper()
        for target in EEG_CHANNELS:
            if base == target.upper():
                ch_map[ch] = target
                break

    if len(ch_map) < 10:
        return {f"SE_Alpha_r{int(r*100):02d}": np.nan for r in r_values}

    raw.rename_channels(ch_map)
    available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available)

    events = parse_events(event_path)
    total_samples = raw.n_times
    ec_intervals = get_eyes_closed_intervals(events, total_samples, sfreq)

    if not ec_intervals:
        return {f"SE_Alpha_r{int(r*100):02d}": np.nan for r in r_values}

    ec_segments, ec_seconds = collect_ec_data(raw, ec_intervals)
    if not ec_segments:
        return {f"SE_Alpha_r{int(r*100):02d}": np.nan for r in r_values}

    # Alpha bandpass filter each segment independently
    # Skip segments too short for MNE's FIR filter (need at least ~0.5s)
    alpha_segments = []
    min_filter_samples = int(0.5 * sfreq)
    for seg in ec_segments:
        if seg.shape[1] < min_filter_samples:
            continue
        try:
            raw_seg = mne.io.RawArray(seg, raw.info, verbose=False)
            raw_seg.filter(ALPHA_LOW, ALPHA_HIGH, verbose=False)
            alpha_segments.append(raw_seg.get_data())
        except Exception:
            continue

    if not alpha_segments:
        return {f"SE_Alpha_r{int(r*100):02d}": np.nan for r in r_values}

    results = {}
    for r_val in r_values:
        key = f"SE_Alpha_r{int(r_val*100):02d}"
        ch_values = []
        for ch_idx in range(len(available)):
            ch_segs = [seg[ch_idx] for seg in alpha_segments]
            se_val = weighted_mean_segments(
                ch_segs,
                lambda s, rv=r_val: compute_se_with_r(s, order=2, r_factor=rv)
            )
            ch_values.append(se_val)

        valid = [v for v in ch_values if not np.isnan(v)]
        results[key] = np.mean(valid) if valid else np.nan

    return results


# ============================================================
# Run analysis
# ============================================================

print(f"\nRecomputing SE_alpha with r = {R_VALUES} on all subjects...")
print("This will take ~30-60 minutes.\n")

# Get the list of serials from the existing CSV
serials = df_existing['ID'].tolist()
groups = df_existing['Group'].tolist()
ages = df_existing['Age'].tolist()
power_ratios = df_existing['Power_Ratio'].tolist()

results_list = []
t0 = time.time()

for i, (serial, group, age, pr) in enumerate(zip(serials, groups, ages, power_ratios)):
    serial_str = str(serial).zfill(5) if str(serial).isdigit() else str(serial)

    se_results = process_subject(serial_str, R_VALUES)
    row = {'ID': serial, 'Group': group, 'Age': age, 'Power_Ratio': pr}
    row.update(se_results)
    results_list.append(row)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(serials) - i - 1) / rate
        pct = 100 * (i + 1) / len(serials)
        mins_left = eta / 60
        print(f"  {i+1}/{len(serials)} ({pct:.0f}%) - {elapsed/60:.1f}min elapsed, ~{mins_left:.1f}min remaining")

elapsed = time.time() - t0
print(f"\nProcessed {len(results_list)} subjects in {elapsed:.1f}s")

df = pd.DataFrame(results_list)

# ============================================================
# Statistical analysis
# ============================================================

output_lines = []
def log(msg=""):
    print(msg)
    output_lines.append(msg)

log("\n" + "=" * 70)
log("SAMPLE ENTROPY TOLERANCE SENSITIVITY TEST")
log("=" * 70)
log(f"Tolerance values tested: r = {R_VALUES} × SD")
log(f"N = {len(df)} subjects")
log(f"Groups: {df['Group'].value_counts().to_dict()}\n")

# --- Group comparisons for each r value ---
for comparison_name, group_pair, pos_label in [
    ("DEMENTIA vs NORMAL", ['Normal', 'Dementia'], 'Dementia'),
    ("MCI vs NORMAL", ['Normal', 'MCI'], 'MCI'),
]:
    log("-" * 70)
    log(f"{comparison_name}")
    log("-" * 70)

    subset = df[df['Group'].isin(group_pair)].copy()
    y = (subset['Group'] == pos_label).astype(int)

    log(f"\n{'Measure':<20s}  {'d':>7s}  {'t':>8s}  {'p':>12s}  {'AUC':>6s}")
    log(f"{'-'*58}")

    for r_val in R_VALUES:
        key = f"SE_Alpha_r{int(r_val*100):02d}"
        group_a = subset[subset['Group'] != pos_label][key].dropna()
        group_b = subset[subset['Group'] == pos_label][key].dropna()

        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
        pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
        d = (group_b.mean() - group_a.mean()) / pooled_std if pooled_std > 0 else 0

        valid_mask = subset[key].notna()
        auc = roc_auc_score(y[valid_mask], subset.loc[valid_mask, key])
        if auc < 0.5:
            auc = 1 - auc

        p_str = f"{p_val:.6f}" if p_val >= 0.0001 else "< 0.0001"
        label = f"SE_α (r={r_val})"
        log(f"{label:<20s}  {d:>+7.3f}  {t_stat:>+8.2f}  {p_str:>12s}  {auc:>6.3f}")

    log()

# --- Cross-validated combined models for each r ---
log("\n" + "=" * 70)
log("CROSS-VALIDATED COMBINED MODELS (Power Ratio + SE_α)")
log("=" * 70)

def run_cv(X_df, y, feature_names, n_splits=10, n_repeats=10):
    """10×10 repeated stratified k-fold CV."""
    all_aucs = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X_df, y):
            X_train = X_df.iloc[train_idx][feature_names].values
            X_test = X_df.iloc[test_idx][feature_names].values
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, probs))
        all_aucs.append(np.mean(fold_aucs))
    return np.mean(all_aucs), np.percentile(all_aucs, 2.5), np.percentile(all_aucs, 97.5)

for comparison_name, group_pair, pos_label in [
    ("DEMENTIA vs NORMAL", ['Normal', 'Dementia'], 'Dementia'),
    ("MCI vs NORMAL", ['Normal', 'MCI'], 'MCI'),
]:
    log(f"\n{comparison_name}")
    log(f"{'Model':<40s}  {'AUC':>6s}  {'95% CI':>16s}")
    log("-" * 66)

    subset = df[df['Group'].isin(group_pair)].copy().dropna(subset=['Power_Ratio'])
    y = (subset['Group'] == pos_label).astype(int)

    # Power ratio alone
    mean_auc, ci_lo, ci_hi = run_cv(subset, y, ['Power_Ratio'])
    log(f"{'Power Ratio alone':<40s}  {mean_auc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    for r_val in R_VALUES:
        key = f"SE_Alpha_r{int(r_val*100):02d}"
        valid = subset.dropna(subset=[key])
        y_valid = (valid['Group'] == pos_label).astype(int)

        # SE alone
        mean_auc, ci_lo, ci_hi = run_cv(valid, y_valid, [key])
        log(f"{f'SE_α (r={r_val}) alone':<40s}  {mean_auc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

        # Combined
        mean_auc, ci_lo, ci_hi = run_cv(valid, y_valid, ['Power_Ratio', key])
        log(f"{f'Power + SE_α (r={r_val})':<40s}  {mean_auc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    log()

# --- Correlation between r values ---
log("\n" + "=" * 70)
log("CORRELATION BETWEEN SE VALUES AT DIFFERENT TOLERANCES")
log("=" * 70)

for i, r1 in enumerate(R_VALUES):
    for r2 in R_VALUES[i+1:]:
        k1 = f"SE_Alpha_r{int(r1*100):02d}"
        k2 = f"SE_Alpha_r{int(r2*100):02d}"
        valid = df[[k1, k2]].dropna()
        r_corr, p_corr = stats.pearsonr(valid[k1], valid[k2])
        log(f"r={r1} vs r={r2}: Pearson r = {r_corr:.4f} (p {f'= {p_corr:.6f}' if p_corr >= 0.0001 else '< 0.0001'})")

log("\n--- Interpretation ---")
log("If SE results are robust across tolerance values, we expect:")
log("  - Similar effect sizes (d) and AUCs across r = 0.15, 0.20, 0.25")
log("  - High correlations (r > 0.95) between SE values at different tolerances")
log("  - Similar cross-validated AUCs for combined models")
log("This contrasts with PE, where parameter changes produce d from -0.70 to +0.71.")

# Save results
output_dir = SCRIPT_DIR.parent / "data_results"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "test_se_tolerance_results.txt"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
print(f"\nResults saved to {output_path}")

# Also save the raw data
csv_out = output_dir / "se_tolerance_values.csv"
df.to_csv(csv_out, index=False)
print(f"SE values saved to {csv_out}")

print("\nDone.")
