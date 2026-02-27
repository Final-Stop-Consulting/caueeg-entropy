"""
Cross-Sampling-Rate PE Parameterization Test
=============================================
Replicates the PE parameter sensitivity analysis on ds004504 (500 Hz) to test
whether the effects map to physical timescale rather than parameter values.

At 500 Hz, the same physical timescales require different parameters:
  - 100 ms alpha window: order=5, delay=13 (vs delay=5 at 200 Hz)
  - 10 ms sub-cycle window: order=3, delay=3 (vs delay=1 at 200 Hz)

If PE sensitivity is a timescale phenomenon (not a parameter artifact), then
matched-timescale parameterizations should produce matching effects across
datasets regardless of sampling rate.

Usage: python test_pe_cross_srate.py
Requires: ds004504 dataset (download from OpenNeuro)
"""

import numpy as np
import pandas as pd
import mne
import antropy as ant
import warnings
import time
import os
import sys
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent

# Search for ds004504 dataset
DS_CANDIDATES = [
    Path("ds004504"),
    Path("../ds004504"),
    Path("../old/ds004504"),
    Path("../../old/ds004504"),
    SCRIPT_DIR.parent / "ds004504",
    SCRIPT_DIR.parent.parent / "ds004504",
    SCRIPT_DIR.parent.parent / "old" / "ds004504",
]

DATASET_PATH = None
for c in DS_CANDIDATES:
    if c.is_dir() and (c / "participants.tsv").exists():
        DATASET_PATH = c
        break

if DATASET_PATH is None:
    print("ERROR: ds004504/ not found.")
    print("Download from: https://openneuro.org/datasets/ds004504")
    print("Place ds004504/ in the project root or parent directory.")
    sys.exit(1)

print(f"Using dataset: {DATASET_PATH.resolve()}")

# EEG channels (10-20 system)
EEG_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5', 'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
]
# Alternative names used in ds004504
ALT_NAMES = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
}

ALPHA_LOW, ALPHA_HIGH = 8, 12
SFREQ_TARGET = 500  # ds004504 native rate

# PE parameterizations matched to physical timescale
# At 500 Hz: window_ms = (order-1) * delay * (1000/500) = (order-1) * delay * 2
PE_PARAMS_500HZ = {
    # Physical timescale matched to CAUEEG parameterizations
    'pe_100ms_o5':  {'order': 5, 'delay': 13, 'window_ms': 104, 'states': 120,
                     'note': 'Proper alpha (matches pe_o5d5 at 200Hz: 100ms)'},
    'pe_100ms_o3':  {'order': 3, 'delay': 25, 'window_ms': 100, 'states': 6,
                     'note': 'Coarse alpha (matches pe_o3d10 at 200Hz: 100ms)'},
    'pe_10ms_o3':   {'order': 3, 'delay': 3,  'window_ms': 12,  'states': 6,
                     'note': 'Sub-cycle (matches pe_o3d1 at 200Hz: 10ms)'},
    'pe_90ms_o7':   {'order': 7, 'delay': 8,  'window_ms': 96,  'states': 5040,
                     'note': 'Rich state space (matches pe_o7d3 at 200Hz: 90ms)'},
    # Also test same parameter values as CAUEEG (different physical timescale!)
    'pe_o5d5_raw':  {'order': 5, 'delay': 5,  'window_ms': 40,  'states': 120,
                     'note': 'Same params as CAUEEG o5d5, but 40ms window at 500Hz'},
    'pe_o3d1_raw':  {'order': 3, 'delay': 1,  'window_ms': 4,   'states': 6,
                     'note': 'Same params as CAUEEG o3d1, but 4ms window at 500Hz'},
}


def compute_pe(signal, order=3, delay=1):
    """Permutation entropy via antropy."""
    try:
        return ant.perm_entropy(signal, order=order, delay=delay, normalize=True)
    except Exception:
        return np.nan


def weighted_mean_segments(segments, func, min_samples=400):
    """Apply func to each segment, weighted average by length."""
    values, weights = [], []
    for seg in segments:
        if len(seg) < min_samples:
            continue
        val = func(seg)
        if not np.isnan(val):
            values.append(val)
            weights.append(len(seg))
    if not values:
        return np.nan
    return np.average(values, weights=np.array(weights, dtype=float))


# ============================================================
# Load participants info
# ============================================================
participants = pd.read_csv(DATASET_PATH / "participants.tsv", sep='\t')
print(f"Participants: {len(participants)}")
print(f"Groups: {participants['Group'].value_counts().to_dict()}")

# Map group labels
GROUP_MAP = {'A': 'AD', 'C': 'Control', 'F': 'FTD'}

results_list = []
t0 = time.time()

for i, row in participants.iterrows():
    sub_id = row['participant_id']
    group = GROUP_MAP.get(row.get('Group', ''), row.get('Group', ''))

    # Find EEG file
    eeg_dir = DATASET_PATH / sub_id / "eeg"
    if not eeg_dir.exists():
        continue

    # Look for .set or .edf files
    eeg_files = list(eeg_dir.glob("*.set")) + list(eeg_dir.glob("*.edf"))
    if not eeg_files:
        continue

    eeg_file = eeg_files[0]

    try:
        if eeg_file.suffix == '.set':
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_edf(str(eeg_file), preload=True, verbose=False)
    except Exception as e:
        print(f"  Skipping {sub_id}: {e}")
        continue

    sfreq = raw.info['sfreq']

    # Map channel names to standard 10-20
    ch_map = {}
    for ch in raw.ch_names:
        base = ch.split('-')[0].strip().upper()
        # Direct match
        for target in EEG_CHANNELS:
            if base == target.upper():
                ch_map[ch] = target
                break
        # Alternative name match
        if ch not in ch_map:
            for std_name, alt_name in ALT_NAMES.items():
                if base == alt_name.upper():
                    ch_map[ch] = std_name
                    break

    if len(ch_map) < 10:
        print(f"  Skipping {sub_id}: only {len(ch_map)} channels matched")
        continue

    raw.rename_channels(ch_map)
    available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available)

    # Use first 60 seconds (ds004504 has no event markers for eyes-closed)
    max_samples = min(int(60 * sfreq), raw.n_times)
    data = raw.get_data(start=0, stop=max_samples)

    # Alpha bandpass filter
    raw_clip = mne.io.RawArray(data, raw.info, verbose=False)
    raw_clip.filter(ALPHA_LOW, ALPHA_HIGH, verbose=False)
    alpha_data = raw_clip.get_data()

    # Compute PE for each parameterization
    subject_results = {
        'Subject': sub_id,
        'Group': group,
        'Age': row.get('Age', np.nan),
        'sfreq': sfreq,
    }

    for pe_name, params in PE_PARAMS_500HZ.items():
        ch_values = []
        for ch_idx in range(len(available)):
            ch_signal = alpha_data[ch_idx]
            # Use full signal as one segment (no eyes-closed extraction for ds004504)
            pe_val = compute_pe(ch_signal, order=params['order'], delay=params['delay'])
            ch_values.append(pe_val)

        valid = [v for v in ch_values if not np.isnan(v)]
        subject_results[pe_name] = np.mean(valid) if valid else np.nan

    results_list.append(subject_results)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(participants)} ({elapsed:.0f}s elapsed)")

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
log("CROSS-SAMPLING-RATE PE PARAMETERIZATION TEST")
log("=" * 70)
log(f"Dataset: ds004504 (OpenNeuro)")
log(f"Sampling rate: {SFREQ_TARGET} Hz")
log(f"N = {len(df)} subjects")
log(f"Groups: {df['Group'].value_counts().to_dict()}\n")

# Print parameterization table
log("PE Parameterizations at 500 Hz:")
log(f"{'Label':<20s}  {'Order':>5s}  {'Delay':>5s}  {'Window':>8s}  {'States':>6s}  Note")
log("-" * 85)
for name, params in PE_PARAMS_500HZ.items():
    log(f"{name:<20s}  {params['order']:>5d}  {params['delay']:>5d}  {params['window_ms']:>5d} ms  {params['states']:>6d}  {params['note']}")
log()

# Group comparisons: AD vs Control
log("=" * 70)
log("AD vs CONTROL (timescale-matched parameterizations)")
log("=" * 70)

subset = df[df['Group'].isin(['AD', 'Control'])].copy()
y = (subset['Group'] == 'AD').astype(int)
log(f"N = {len(subset)} (AD: {(y==1).sum()}, Control: {(y==0).sum()})\n")

log(f"{'Parameterization':<20s}  {'d':>7s}  {'t':>8s}  {'p':>12s}  {'AUC':>6s}  {'Window':>8s}")
log("-" * 70)

for pe_name, params in PE_PARAMS_500HZ.items():
    group_ctrl = subset[subset['Group'] == 'Control'][pe_name].dropna()
    group_ad = subset[subset['Group'] == 'AD'][pe_name].dropna()

    if len(group_ctrl) < 5 or len(group_ad) < 5:
        log(f"{pe_name:<20s}  insufficient data")
        continue

    t_stat, p_val = stats.ttest_ind(group_ctrl, group_ad, equal_var=False)
    pooled_std = np.sqrt((group_ctrl.std()**2 + group_ad.std()**2) / 2)
    d = (group_ad.mean() - group_ctrl.mean()) / pooled_std if pooled_std > 0 else 0

    valid_mask = subset[pe_name].notna()
    try:
        auc = roc_auc_score(y[valid_mask], subset.loc[valid_mask, pe_name])
        if auc < 0.5:
            auc = 1 - auc
    except Exception:
        auc = np.nan

    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
    log(f"{pe_name:<20s}  {d:>+7.3f}  {t_stat:>+8.2f}  {p_str:>12s}  {auc:>6.3f}  {params['window_ms']:>5d} ms")

log()

# Comparison table: CAUEEG vs ds004504 at matched timescales
log("\n" + "=" * 70)
log("INTERPRETATION: TIMESCALE vs PARAMETER COMPARISON")
log("=" * 70)
log("""
If PE sensitivity maps to PHYSICAL TIMESCALE:
  - pe_100ms_o5 (500Hz) should match pe_o5d5 (200Hz): both null
  - pe_10ms_o3 (500Hz) should match pe_o3d1 (200Hz): both large negative d
  - pe_100ms_o3 (500Hz) should match pe_o3d10 (200Hz): both large positive d

If PE sensitivity maps to PARAMETER VALUES:
  - pe_o5d5_raw (500Hz) should match pe_o5d5 (200Hz)
  - pe_o3d1_raw (500Hz) should match pe_o3d1 (200Hz)
  But these have DIFFERENT physical timescales (40ms vs 100ms, 4ms vs 10ms)

The timescale hypothesis predicts that matching physical windows across sampling
rates will reproduce the effect pattern, while matching parameter values will NOT.
This would confirm that the parameter sensitivity is fundamentally a timescale
phenomenon, not an artifact of specific parameter values.

NOTE: ds004504 is a smaller dataset (N=88) with different demographics, no
eyes-closed extraction, and mixed dementia types (AD + FTD). Effect sizes
will differ from CAUEEG, but the DIRECTION PATTERN should be consistent
if the timescale hypothesis holds.
""")

# Save results
output_dir = SCRIPT_DIR.parent / "data_results"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "test_pe_cross_srate_results.txt"
with open(output_path, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"Results saved to {output_path}")

csv_out = output_dir / "pe_cross_srate_values.csv"
df.to_csv(csv_out, index=False)
print(f"PE values saved to {csv_out}")

print("\nDone.")
