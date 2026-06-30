"""
Sample Entropy parameter sweep (CAUEEG)
=======================================
PE has been criticized for parameter sensitivity while sample entropy is typically
run at a single embedding dimension. Here we sweep SE symmetrically across:
    m (embedding order) in {1, 2, 3}
    r (tolerance)       in {0.10, 0.15, 0.20, 0.25, 0.30} x SD
on alpha-band EEG, per-segment weighted, using the SAME v3 pipeline
(eyes-closed extraction, artifact exclusion, segments >= 2 s).

Two channel definitions are produced from the SAME per-channel values:
    whole  = 19-channel scalp average (as in the paper)
    post   = occipital/parietal subset where alpha lives (O1,O2,P3,P4,Pz,T5,T6)
            -> also reports a posterior (occipital/parietal) channel subset.

SE is computed with antropy.sample_entropy(tolerance=r*SD) so every value matches
the published v3 pipeline exactly (m=2, r=0.20 reproduces the paper's SE_Alpha).

Output: data_results/se_sweep_values.csv  (one row per subject)
Then run se_sweep_stats.py for effect sizes + one BH-FDR across the full grid.

Run (~45-90 min, N~1,187):
    python se_parameter_sweep.py
    python se_parameter_sweep.py --max-subjects 5   # quick test on a few subjects
Requires: mne, antropy, numpy, pandas, scipy, scikit-learn
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import antropy as ant

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# --- sweep grid ---
M_VALUES = [1, 2, 3]
R_FACTORS = [0.10, 0.15, 0.20, 0.25, 0.30]

# --- pipeline params (match caueeg_entropy_v3.py / test_se_tolerance.py) ---
ALPHA = (8.0, 12.0)
MAX_SECONDS = 60
MIN_SEG_SAMPLES = 400
ARTIFACT_BUFFER = 1.0
EEG_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1", "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4", "T6", "Fz", "Cz", "Pz",
]
POSTERIOR = ["O1", "O2", "P3", "P4", "Pz", "T5", "T6"]  # occipital/parietal (alpha generators)
ARTIFACT_EVENTS = {"artifact", "move", "chewing", "swallowing", "talk", "cough", "eye blinking"}
EC_START_PREFIXES = ("Eyes Closed",)
EC_END_PREFIXES = ("Eyes Open", "Photic On", "HV -", "Paused", "Stop Recording", "Photic Off")


def project_root():
    p = Path(__file__).resolve()
    for anc in p.parents:
        if (anc / "data_results").is_dir() or (anc / "analysis_code").is_dir():
            return anc
    return p.parents[3]


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
          f"ETA {eta/60:4.1f}m |{kept_s} {label:<8}", end="", flush=True)


# ----------------------------------------------------------------------------
def parse_events(event_path):
    with open(event_path) as f:
        events = json.load(f)
    out = []
    for ev in events:
        if isinstance(ev, (list, tuple)) and len(ev) >= 2:
            out.append((int(ev[0]), str(ev[1])))
        elif isinstance(ev, dict):
            out.append((int(ev.get("sample", ev.get("onset", 0))),
                        str(ev.get("label", ev.get("description", "")))))
    return out


def eyes_closed_intervals(events, total, sfreq):
    ec, art = [], []
    for idx, label in events:
        if any(label.startswith(p) for p in EC_START_PREFIXES):
            ec.append([idx, total])
        elif any(label.startswith(p) for p in EC_END_PREFIXES):
            if ec and ec[-1][1] == total:
                ec[-1][1] = idx
        if label.strip().lower() in ARTIFACT_EVENTS:
            buf = int(ARTIFACT_BUFFER * sfreq)
            art.append((max(0, idx - buf), min(total, idx + buf)))
    clean = []
    for s, e in ec:
        segs = [(s, e)]
        for as_, ae in art:
            new = []
            for a, b in segs:
                if ae <= a or as_ >= b:
                    new.append((a, b))
                else:
                    if a < as_:
                        new.append((a, as_))
                    if ae < b:
                        new.append((ae, b))
            segs = new
        clean.extend(segs)
    return [(s, e) for s, e in clean if (e - s) >= MIN_SEG_SAMPLES]


def collect_ec(raw, intervals, max_seconds=MAX_SECONDS):
    sfreq = raw.info["sfreq"]
    need = int(max_seconds * sfreq)
    out, tot = [], 0
    for s, e in intervals:
        if tot >= need:
            break
        take = min(e - s, need - tot)
        out.append(raw.get_data(start=s, stop=s + take))
        tot += take
    return out, tot / sfreq


def se_grid_channel(x, m_values=M_VALUES, r_factors=R_FACTORS):
    """SE for one 1D signal across the (m, r) grid.

    Uses antropy.sample_entropy(tolerance=r*SD) so values match the published v3
    pipeline exactly (which used antropy). tolerance is the absolute radius;
    r is expressed as a fraction of the signal SD.
    """
    x = np.asarray(x, float)
    sd = np.std(x, ddof=1)
    res = {(m, rf): np.nan for m in m_values for rf in r_factors}
    if sd == 0 or len(x) < max(m_values) + 2:
        return res
    for m in m_values:
        for rf in r_factors:
            try:
                res[(m, rf)] = ant.sample_entropy(x, order=m, metric="chebyshev",
                                                  tolerance=rf * sd)
            except Exception:
                res[(m, rf)] = np.nan
    return res


def bandpass_segments(ec_segments, info):
    """Alpha bandpass each EC segment (FIR zero-phase); drop too-short ones."""
    out = []
    minf = int(0.5 * info["sfreq"])
    for seg in ec_segments:
        if seg.shape[1] < minf:
            continue
        r = mne.io.RawArray(seg, info, verbose=False)
        r.filter(*ALPHA, method="fir", phase="zero", fir_design="firwin", verbose=False)
        out.append(r.get_data())
    return out


def process_subject(edf_path, event_path):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
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

    events = parse_events(event_path)
    intervals = eyes_closed_intervals(events, raw.n_times, sfreq)
    if not intervals:
        return None
    ec_segs, ec_s = collect_ec(raw, intervals)
    alpha_segs = bandpass_segments(ec_segs, raw.info)
    if not alpha_segs:
        return None

    # per-channel weighted-mean SE across segments, for each (m,r)
    ch_se = {ch: {(m, rf): np.nan for m in M_VALUES for rf in R_FACTORS} for ch in avail}
    for ci, ch in enumerate(avail):
        acc = {(m, rf): [0.0, 0.0] for m in M_VALUES for rf in R_FACTORS}  # [wsum, w]
        for seg in alpha_segs:
            sig = seg[ci]
            if len(sig) < MIN_SEG_SAMPLES:
                continue
            g = se_grid_channel(sig)
            w = len(sig)
            for k, v in g.items():
                if not np.isnan(v):
                    acc[k][0] += v * w
                    acc[k][1] += w
        for k, (ws, w) in acc.items():
            ch_se[ch][k] = ws / w if w > 0 else np.nan

    def region_mean(channels, key):
        vals = [ch_se[ch][key] for ch in channels if ch in ch_se and not np.isnan(ch_se[ch][key])]
        return np.mean(vals) if vals else np.nan

    out = {"EC_seconds": round(ec_s, 1), "n_channels": len(avail), "sfreq": sfreq}
    post = [c for c in POSTERIOR if c in avail]
    for m in M_VALUES:
        for rf in R_FACTORS:
            tag = f"m{m}_r{int(rf*100):02d}"
            out[f"SE_Alpha_{tag}_whole"] = region_mean(avail, (m, rf))
            out[f"SE_Alpha_{tag}_post"] = region_mean(post, (m, rf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()

    root = project_root()
    ds = None
    for c in [root / "caueeg-dataset", root / "data" / "caueeg-dataset", root.parent / "caueeg-dataset"]:
        if (c / "annotation.json").exists():
            ds = c
            break
    if ds is None:
        print("ERROR: caueeg-dataset not found (data/caueeg-dataset/annotation.json).")
        sys.exit(1)
    v3 = root / "data_results" / "caueeg_entropy_v3.csv"
    if not v3.exists():
        print(f"ERROR: {v3} not found.")
        sys.exit(1)
    base = pd.read_csv(v3)[["ID", "Group", "Subtype", "Age"]]
    print(f"Dataset: {ds}  |  subjects in v3: {len(base)}")
    print(f"Sweep: m={M_VALUES} x r={R_FACTORS}  (whole + posterior)")

    total = args.max_subjects if args.max_subjects else len(base)
    rows, t0, seen = [], time.time(), 0
    for i, r in base.iterrows():
        if args.max_subjects and seen >= args.max_subjects:
            break
        seen += 1
        serial = str(r["ID"]).zfill(5) if str(r["ID"]).isdigit() else str(r["ID"])
        show_progress(seen, total, t0, kept=len(rows), label=serial)  # live: which subject, % done, ETA
        edf = ds / "signal" / "edf" / f"{serial}.edf"
        ev = ds / "event" / f"{serial}.json"
        if not edf.exists() or not ev.exists():
            continue
        res = process_subject(edf, ev)
        if res is None:
            continue
        res.update({"ID": r["ID"], "Group": r["Group"], "Subtype": r["Subtype"], "Age": r["Age"]})
        rows.append(res)
    print()  # newline after the progress bar

    df = pd.DataFrame(rows)
    lead = ["ID", "Group", "Subtype", "Age", "EC_seconds", "n_channels", "sfreq"]
    df = df[[c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]]
    out = root / "data_results" / "se_sweep_values.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} subjects in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
