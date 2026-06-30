"""
Harmonized ds004504 pipeline (cross-dataset PE under physical-timescale matching)
================================================================================
Brings ds004504 (500 Hz) UP TO the CAUEEG v3 pipeline so that PE/SE/LZC are
computed identically across datasets, at IDENTICAL physical timescales:

  1. read .set/.edf, map channels to the 19-channel 10-20 montage (T7/T8/P7/P8 -> T3/T4/T5/T6)
  2. DOWNSAMPLE 500 -> 200 Hz (MNE FIR anti-alias)  [pre-committed resample direction]
  3. automatic artifact rejection -> clean contiguous segments (ds004504 has no EC markers;
     the whole recording is resting eyes-closed -> auto-clean is the closest harmonization)
  4. per clean segment: bandpass alpha (8-12) and theta (4-8), FIR zero-phase
  5. per-segment PE/SE/LZC, weighted by segment length (segments < 2 s dropped)
  6. spectral power (Welch) on UNFILTERED segments -> relative alpha/theta power ratio

After downsampling, the canonical CAUEEG parameterizations use the SAME integer
(order, delay) and therefore the SAME physical windows -- this is the whole point.

Output: data_results/ds004504_harmonized_values.csv  (one row per subject)
CAUEEG side is NOT recomputed here: the convergence step reuses the published
v3 per-subject values (caueeg_entropy_v3.csv).

Run (~minutes; N=88):
    python crossdataset_harmonize.py
    python crossdataset_harmonize.py --max-subjects 3   # quick test on a few subjects

Requires: mne, antropy, numpy, pandas, scipy  (see requirements.txt)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

import mne
import antropy as ant

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ----------------------------------------------------------------------------
# CONFIG  (kept identical to caueeg_entropy_v3.py where it matters)
# ----------------------------------------------------------------------------
TARGET_SFREQ = 200.0          # downsample ds004504 500 -> 200
ALPHA = (8.0, 12.0)
THETA = (4.0, 8.0)
MAX_SECONDS = 60              # cap clean data per subject (matches v3)
MIN_SEG_SAMPLES = 400        # 2 s at 200 Hz (matches v3; avoids finite-sample/edge bias)

# Automatic artifact rejection (ds004504 has no markers) -- pre-committed defaults
EPOCH_S = 1.0                # epoch length for rejection scan
PTP_MAX_UV = 150.0           # reject epoch if peak-to-peak > 150 uV on any kept channel
VAR_MULT = 5.0               # reject epoch if channel variance > VAR_MULT * median channel variance
CH_DROP_FRAC = 0.50          # drop a channel for a subject if > 50% of its epochs rejected
MIN_CLEAN_S = 20.0           # require >= 20 s clean or exclude subject

EEG_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4", "T6",
    "Fz", "Cz", "Pz",
]
ALT_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # std_name -> ds004504 name
GROUP_MAP = {"A": "AD", "C": "Control", "F": "FTD"}

# PE grid: 4 canonical (identical integers to CAUEEG v3) + density grid for component-C robustness.
CANONICAL_PE = {
    "pe_o5d5": (5, 5),   # proper alpha, 100 ms, 120 states  (HEADLINE)
    "pe_o3d10": (3, 10),  # coarse,       100 ms, 6 states
    "pe_o7d3": (7, 3),    # rich,         90 ms,  5040 states
    "pe_o3d1": (3, 1),    # sub-cycle,    10 ms,  6 states
}
DENSITY_ORDERS = [3, 5, 7]
DENSITY_DELAYS = [1, 2, 3, 5, 8, 10]
MAX_WINDOW_MS = 150.0


def window_ms(order, delay, sfreq=TARGET_SFREQ):
    return (order - 1) * delay * (1000.0 / sfreq)


def build_pe_grid():
    grid = dict(CANONICAL_PE)
    for o in DENSITY_ORDERS:
        for d in DENSITY_DELAYS:
            if window_ms(o, d) <= MAX_WINDOW_MS:
                grid[f"pe_o{o}d{d}"] = (o, d)
    return grid


PE_GRID = build_pe_grid()


# ----------------------------------------------------------------------------
# Measure functions  (identical antropy calls to caueeg_entropy_v3.py)
# ----------------------------------------------------------------------------
def compute_pe(signal, order, delay):
    try:
        return ant.perm_entropy(signal, order=order, delay=delay, normalize=True)
    except Exception:
        return np.nan


def compute_se(signal, order=2, metric="chebyshev"):
    try:
        return ant.sample_entropy(signal, order=order, metric=metric)
    except Exception:
        return np.nan


def compute_lzc(signal):
    try:
        return ant.lziv_complexity(signal > np.median(signal), normalize=True)
    except Exception:
        return np.nan


def weighted_mean_segments(segments, func):
    """Apply func per segment (1D arrays), weighted average by length; segs < MIN_SEG_SAMPLES dropped."""
    vals, wts = [], []
    for seg in segments:
        if len(seg) < MIN_SEG_SAMPLES:
            continue
        v = func(seg)
        if not np.isnan(v):
            vals.append(v)
            wts.append(len(seg))
    if not vals:
        return np.nan
    return np.average(vals, weights=np.asarray(wts, dtype=float))


# ----------------------------------------------------------------------------
# Artifact rejection -> clean contiguous segments (per subject)
# ----------------------------------------------------------------------------
def clean_segments(data, sfreq):
    """
    data: (n_channels, n_samples) UNFILTERED, already downsampled.
    Returns (kept_channel_idx, list_of_(start,end)_clean_spans, total_clean_seconds).
    Scans 1 s epochs; rejects epochs with ptp>PTP_MAX_UV or var>VAR_MULT*median(var);
    drops channels rejected in >CH_DROP_FRAC of epochs; merges consecutive good epochs.
    Assumes data is in volts (MNE) -> convert to uV for the ptp threshold.
    """
    n_ch, n_samp = data.shape
    ep = int(EPOCH_S * sfreq)
    n_ep = n_samp // ep
    if n_ep == 0:
        return [], [], 0.0

    data_uv = data * 1e6
    # per-epoch, per-channel stats
    ptp = np.zeros((n_ch, n_ep))
    var = np.zeros((n_ch, n_ep))
    for e in range(n_ep):
        seg = data_uv[:, e * ep:(e + 1) * ep]
        ptp[:, e] = seg.max(axis=1) - seg.min(axis=1)
        var[:, e] = seg.var(axis=1)

    med_var = np.median(var) if np.median(var) > 0 else var.mean()
    bad_epoch_ch = (ptp > PTP_MAX_UV) | (var > VAR_MULT * med_var)

    # drop chronically-bad channels
    ch_bad_frac = bad_epoch_ch.mean(axis=1)
    kept = [i for i in range(n_ch) if ch_bad_frac[i] <= CH_DROP_FRAC]
    if len(kept) < 10:
        return [], [], 0.0

    # an epoch is good if no KEPT channel is bad in it
    epoch_good = ~bad_epoch_ch[kept, :].any(axis=0)

    # merge consecutive good epochs into spans
    spans = []
    run_start = None
    for e in range(n_ep):
        if epoch_good[e] and run_start is None:
            run_start = e
        elif not epoch_good[e] and run_start is not None:
            spans.append((run_start * ep, e * ep))
            run_start = None
    if run_start is not None:
        spans.append((run_start * ep, n_ep * ep))

    total_clean = sum((e - s) for s, e in spans) / sfreq
    return kept, spans, total_clean


def bandpass(data, sfreq, lo, hi):
    """FIR zero-phase bandpass on (n_channels, n_samples)."""
    info = mne.create_info([f"ch{i}" for i in range(data.shape[0])], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(lo, hi, method="fir", phase="zero", fir_design="firwin", verbose=False)
    return raw.get_data()


# ----------------------------------------------------------------------------
# Per-subject processing
# ----------------------------------------------------------------------------
def process_subject(eeg_file):
    """Returns dict of measures for one subject, or None if unusable."""
    try:
        if eeg_file.suffix == ".set":
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_edf(str(eeg_file), preload=True, verbose=False)
    except Exception as e:
        print(f"    read fail: {e}")
        return None

    # channel mapping -> standard 10-20 names used by v3
    ch_map = {}
    alt_to_std = {v.upper(): k for k, v in ALT_NAMES.items()}
    for ch in raw.ch_names:
        base = ch.split("-")[0].strip().upper()
        for t in EEG_CHANNELS:
            if base == t.upper():
                ch_map[ch] = t
                break
        if ch not in ch_map and base in alt_to_std:
            ch_map[ch] = alt_to_std[base]
    if len(ch_map) < 10:
        print(f"    only {len(ch_map)} channels matched")
        return None
    raw.rename_channels(ch_map)
    avail = [c for c in EEG_CHANNELS if c in raw.ch_names]
    raw.pick_channels(avail)

    # DOWNSAMPLE 500 -> 200 (anti-aliased) -- the load-bearing step
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-6:
        raw.resample(TARGET_SFREQ, npad="auto", verbose=False)
    sfreq = raw.info["sfreq"]

    data = raw.get_data()  # (n_ch, n_samp) volts, downsampled
    kept_idx, spans, clean_s = clean_segments(data, sfreq)
    if clean_s < MIN_CLEAN_S or not spans:
        print(f"    insufficient clean data ({clean_s:.1f}s)")
        return None

    # cap at MAX_SECONDS of clean data
    capped, used = [], 0.0
    for s, e in spans:
        if used >= MAX_SECONDS:
            break
        span_s = (e - s) / sfreq
        if used + span_s > MAX_SECONDS:
            e = s + int((MAX_SECONDS - used) * sfreq)
        capped.append((s, e))
        used += (e - s) / sfreq

    kept = np.asarray(kept_idx)
    raw_segs = [data[np.ix_(kept, range(s, e))] for s, e in capped]  # unfiltered, kept channels

    # bandpass each segment, per band
    alpha_segs = [bandpass(seg, sfreq, *ALPHA) for seg in raw_segs]
    theta_segs = [bandpass(seg, sfreq, *THETA) for seg in raw_segs]

    out = {"n_segments": len(raw_segs), "EC_seconds": round(used, 1),
           "n_channels": len(kept), "sfreq_native": 500.0, "sfreq_used": sfreq}

    # channel-then-segment averaging: for each measure, average over channels per segment,
    # then weighted-average over segments (matches v3 structure)
    def measure_over(segs_band, func):
        per_seg_ch_vals, lengths = [], []
        for seg in segs_band:
            ch_vals = [func(seg[ci]) for ci in range(seg.shape[0])]
            ch_vals = [v for v in ch_vals if not np.isnan(v)]
            if ch_vals:
                per_seg_ch_vals.append(np.mean(ch_vals))
                lengths.append(seg.shape[1])
        if not per_seg_ch_vals:
            return np.nan
        return np.average(per_seg_ch_vals, weights=np.asarray(lengths, dtype=float))

    # PE grid (alpha band is the locus of the test; theta computed for the ratio)
    for name, (o, d) in PE_GRID.items():
        out[f"{name}_Alpha"] = measure_over(alpha_segs, lambda x: compute_pe(x, o, d))
    # canonical also on theta for ratio parity with v3
    for name, (o, d) in CANONICAL_PE.items():
        out[f"{name}_Theta"] = measure_over(theta_segs, lambda x: compute_pe(x, o, d))
        a, t = out.get(f"{name}_Alpha"), out.get(f"{name}_Theta")
        out[f"{name}_Ratio"] = (a / t) if (a and t and t != 0) else np.nan

    # SE, LZC (alpha + theta + ratio)
    out["SE_Alpha"] = measure_over(alpha_segs, compute_se)
    out["SE_Theta"] = measure_over(theta_segs, compute_se)
    out["SE_Ratio"] = (out["SE_Alpha"] / out["SE_Theta"]) if out["SE_Theta"] else np.nan
    out["LZC_Alpha"] = measure_over(alpha_segs, compute_lzc)
    out["LZC_Theta"] = measure_over(theta_segs, compute_lzc)
    out["LZC_Ratio"] = (out["LZC_Alpha"] / out["LZC_Theta"]) if out["LZC_Theta"] else np.nan

    # spectral power on UNFILTERED segments
    def band_power(psd, freqs, lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return np.trapz(psd[m], freqs[m])
    a_abs, t_abs, tot = [], [], []
    for seg in raw_segs:
        for ci in range(seg.shape[0]):
            nper = min(int(2.0 * sfreq), seg.shape[1])
            if nper < int(0.5 * sfreq):
                continue
            f, p = welch(seg[ci], fs=sfreq, nperseg=nper, noverlap=nper // 2)
            a_abs.append(band_power(p, f, *ALPHA))
            t_abs.append(band_power(p, f, *THETA))
            tot.append(band_power(p, f, 0.5, 45.0))
    if tot:
        A, T, TOT = np.mean(a_abs), np.mean(t_abs), np.mean(tot)
        out["Alpha_Power_Abs"], out["Theta_Power_Abs"] = A, T
        out["Alpha_Power_Rel"] = A / TOT if TOT else np.nan
        out["Theta_Power_Rel"] = T / TOT if TOT else np.nan
        out["Power_Ratio"] = (A / TOT) / (T / TOT) if (T and TOT) else np.nan
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def project_root():
    """Walk up from this file to the caueeg_paper root (dir containing data_results)."""
    p = Path(__file__).resolve()
    for anc in p.parents:
        if (anc / "data_results").is_dir() or (anc / "analysis_code").is_dir():
            return anc
    return p.parents[3]  # caueeg_paper, by known layout


def show_progress(done, total, t0, kept=None, label=""):
    """Live single-line progress bar: count, %, elapsed, ETA, current subject."""
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    pct = 100 * done / total if total else 0
    fill = int(28 * done / total) if total else 0
    bar = "#" * fill + "." * (28 - fill)
    kept_s = f" kept {kept}" if kept is not None else ""
    print(f"\r  [{bar}] {done}/{total} {pct:4.0f}% | {el/60:4.1f}m elapsed | "
          f"ETA {eta/60:4.1f}m |{kept_s} {label:<10}", end="", flush=True)


def find_dataset():
    root = project_root()
    candidates = [
        root / "ds004504",
        root / "data" / "ds004504",
        Path("ds004504"), Path("data/ds004504"),
    ]
    for c in candidates:
        if (c / "participants.tsv").exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=0, help="limit number of subjects (for a quick test run)")
    args = ap.parse_args()

    ds = find_dataset()
    if ds is None:
        print("ERROR: ds004504 not found (looked for data/ds004504/participants.tsv).")
        sys.exit(1)
    print(f"Dataset: {ds.resolve()}")
    print(f"PE grid ({len(PE_GRID)} params): {sorted(PE_GRID)}")

    parts = pd.read_csv(ds / "participants.tsv", sep="\t")
    print(f"Participants: {len(parts)}  groups={parts['Group'].value_counts().to_dict()}")

    total = args.max_subjects if args.max_subjects else len(parts)
    rows, t0, seen = [], time.time(), 0
    for i, r in parts.iterrows():
        if args.max_subjects and seen >= args.max_subjects:
            break
        seen += 1
        sub = r["participant_id"]
        show_progress(seen, total, t0, kept=len(rows), label=sub)  # live: which subject, % done, ETA
        eeg_dir = ds / sub / "eeg"
        if not eeg_dir.exists():
            continue
        files = list(eeg_dir.glob("*.set")) + list(eeg_dir.glob("*.edf"))
        if not files:
            continue
        res = process_subject(files[0])
        if res is None:
            continue
        res.update({"Subject": sub, "Group": GROUP_MAP.get(r.get("Group"), r.get("Group")),
                    "Age": r.get("Age", np.nan), "MMSE": r.get("MMSE", np.nan)})
        rows.append(res)
    print()  # newline after the progress bar

    df = pd.DataFrame(rows)
    if df.empty:
        print("No subjects processed.")
        sys.exit(1)
    lead = ["Subject", "Group", "Age", "MMSE", "n_segments", "EC_seconds", "n_channels",
            "sfreq_native", "sfreq_used"]
    df = df[[c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]]

    out_dir = project_root() / "data_results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "ds004504_harmonized_values.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nProcessed {len(df)} subjects in {time.time()-t0:.0f}s -> {out_csv}")
    print(df.groupby("Group")[["pe_o5d5_Alpha", "pe_o3d1_Alpha", "SE_Alpha", "Power_Ratio"]].mean())


if __name__ == "__main__":
    main()
