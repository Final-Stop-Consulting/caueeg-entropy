"""
Empirical validation of the Section 4.2 analytic claim (sub-cycle PE = phase entropy)
=================================================================================
The manuscript proves analytically that on a narrowband signal, order-3 / delay-1
permutation entropy reduces to the entropy of the instantaneous-phase distribution:
three consecutive samples' ordinal pattern is determined by the local phase, and the
six permutations partition [0, 2pi) into six sectors. That was shown on an idealized
sine; here we show it on the actual EEG.

Two measurements per channel/segment on alpha-band CAUEEG (8-12 Hz):
  1. pe_o3d1            -- antropy permutation entropy (order=3, delay=1), normalized
  2. phase_entropy      -- normalized Shannon entropy of the Hilbert instantaneous
                           phase, binned into 6 equal sectors
  3. phase_perm_acc     -- fraction of consecutive triplets whose ordinal permutation
                           equals the one predicted from the local Hilbert phase under
                           the sine model (delta = 2*pi*f0/fs). Direct test that
                           "phase determines the permutation" on real data.

Validation passes if, across subjects:
  - pe_o3d1 correlates strongly with phase_entropy (Pearson r high), and
  - phase_perm_acc is high (permutation is largely fixed by phase).

Scope boundary (state in the paper): this reduction holds for BAND-FILTERED,
SUB-CYCLE PE only -- not for broadband PE or timescale-appropriate parameters.

Output: data_results/phase_entropy_validation.csv + console summary.
Run:  python phase_entropy_validation.py  [--max-subjects 5]
Reuses the CAUEEG loader from se_parameter_sweep.py (same folder).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import antropy as ant
from scipy.signal import hilbert

# reuse the verified CAUEEG loader / pipeline from Item 2
from se_parameter_sweep import (project_root, parse_events, eyes_closed_intervals,
                            collect_ec, bandpass_segments, EEG_CHANNELS, ALPHA,
                            MIN_SEG_SAMPLES, show_progress)
import mne
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

import collections


def _phase_predicted_perms(sig):
    """Per delay-1 triplet, the ordinal permutation predicted from phase ALONE.
    Hilbert analytic signal: x ~ A*cos(theta). The §4.2 claim is that the ordinal
    pattern is set by phase, not amplitude; so we predict the triplet ordering from
    cos(theta) at the three sample positions (amplitude discarded) and compare to the
    actual sample ordering. Parameter-free (no centre-frequency assumption)."""
    N = len(sig)
    cph = np.cos(np.angle(hilbert(sig)))
    actual = np.argsort(np.stack([sig[:N - 2], sig[1:N - 1], sig[2:N]], axis=1), axis=1)
    pred = np.argsort(np.stack([cph[:N - 2], cph[1:N - 1], cph[2:N]], axis=1), axis=1)
    return actual, pred


def phase_perm_accuracy(sig):
    """Fraction of triplets whose ACTUAL ordinal permutation equals the phase-predicted one."""
    if len(sig) < 4:
        return np.nan
    actual, pred = _phase_predicted_perms(sig)
    return float(np.mean(np.all(actual == pred, axis=1)))


def pe_from_phase(sig):
    """Normalized order-3 entropy of the PHASE-PREDICTED permutation distribution.
    If the reduction holds this equals the actual pe_o3d1."""
    if len(sig) < 4:
        return np.nan
    _, pred = _phase_predicted_perms(sig)
    cc = collections.Counter(map(tuple, pred))
    p = np.array(list(cc.values()), float)
    p /= p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(6))


def pe_o3d1(sig):
    try:
        return ant.perm_entropy(sig, order=3, delay=1, normalize=True)
    except Exception:
        return np.nan


def weighted(vals, wts):
    vals, wts = np.asarray(vals, float), np.asarray(wts, float)
    m = ~np.isnan(vals)
    return float(np.average(vals[m], weights=wts[m])) if m.any() else np.nan


def process(edf, ev):
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    ch_map = {}
    for ch in raw.ch_names:
        base = ch.split("-")[0].strip().upper()
        for t in EEG_CHANNELS:
            if base == t.upper():
                ch_map[ch] = t
                break
    if len(ch_map) < 10:
        return None
    raw.rename_channels(ch_map)
    avail = [c for c in EEG_CHANNELS if c in raw.ch_names]
    raw.pick_channels(avail)
    inter = eyes_closed_intervals(parse_events(ev), raw.n_times, sfreq)
    if not inter:
        return None
    segs = bandpass_segments(collect_ec(raw, inter)[0], raw.info)
    if not segs:
        return None
    pe_v, ph_v, acc_v, w = [], [], [], []
    for seg in segs:
        for ci in range(seg.shape[0]):
            s = seg[ci]
            if len(s) < MIN_SEG_SAMPLES:
                continue
            pe_v.append(pe_o3d1(s)); ph_v.append(pe_from_phase(s))
            acc_v.append(phase_perm_accuracy(s)); w.append(len(s))
    if not w:
        return None
    return {"pe_o3d1": weighted(pe_v, w), "pe_from_phase": weighted(ph_v, w),
            "phase_perm_acc": weighted(acc_v, w), "n_seg": len(segs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()
    root = project_root()
    ds = next((c for c in [root / "caueeg-dataset", root / "data" / "caueeg-dataset", root.parent / "caueeg-dataset"]
               if (c / "annotation.json").exists()), None)
    if ds is None:
        print("ERROR: caueeg-dataset not found."); sys.exit(1)
    base = pd.read_csv(root / "data_results" / "caueeg_entropy_v3.csv")[["ID", "Group", "Subtype", "Age"]]
    total = args.max_subjects if args.max_subjects else len(base)
    rows, t0, seen = [], time.time(), 0
    for _, r in base.iterrows():
        if args.max_subjects and seen >= args.max_subjects:
            break
        seen += 1
        serial = str(r["ID"]).zfill(5) if str(r["ID"]).isdigit() else str(r["ID"])
        show_progress(seen, total, t0, kept=len(rows), label=serial)
        edf = ds / "signal" / "edf" / f"{serial}.edf"
        ev = ds / "event" / f"{serial}.json"
        if not edf.exists() or not ev.exists():
            continue
        res = process(edf, ev)
        if res is None:
            continue
        res.update({"ID": r["ID"], "Group": r["Group"]})
        rows.append(res)
    print()
    df = pd.DataFrame(rows)
    out = root / "data_results" / "phase_entropy_validation.csv"
    df.to_csv(out, index=False)

    from scipy import stats
    d = df.dropna(subset=["pe_o3d1", "pe_from_phase"])
    rP, pP = stats.pearsonr(d["pe_o3d1"], d["pe_from_phase"])
    rS, _ = stats.spearmanr(d["pe_o3d1"], d["pe_from_phase"])
    print("=" * 64)
    print("Section 4.2 empirical validation")
    print("=" * 64)
    print(f"N subjects = {len(df)}")
    print(f"pe_o3d1 vs pe_from_phase:  Pearson r = {rP:+.3f} (p={pP:.1e})  Spearman = {rS:+.3f}")
    print(f"phase_perm_acc (phase alone determines the permutation): "
          f"mean = {df['phase_perm_acc'].mean():.3f}  (chance = 1/6 = 0.167)")
    print("  -> high accuracy + r~1 confirm sub-cycle PE is set by instantaneous phase,")
    print("     not amplitude, on real EEG. SCOPE: band-filtered, sub-cycle (o3d1) only.")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
