"""
SE sweep statistics + one BH-FDR across the full grid
=====================================================
Consumes:
  data_results/caueeg_entropy_v3.csv         (PE / LZC / power per subject)
  data_results/se_sweep_values.csv           (SE over m x r, whole + posterior)
  data_results/ds004504_harmonized_values.csv (optional; cross-dataset SE)

Produces, for each comparison (Dementia vs Normal, MCI vs Normal, AD vs Normal):
  - Cohen's d, Welch p, direction-corrected AUC for EVERY measure in the unified
    Table-3 grid (PE params + LZC + power + the full SE m x r grid)
  - ONE Benjamini-Hochberg FDR (q=0.05) across that whole grid
  - SE stability across m and r (d range, sign, correlations) vs PE's d swing
  - whole-scalp vs occipital/parietal channel subset
  - m = 2 justification numbers
  - cross-dataset SE: CAUEEG vs ds004504

Run:
    python se_sweep_stats.py
    python se_sweep_stats.py --self-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

M_VALUES = [1, 2, 3]
R_FACTORS = [0.10, 0.15, 0.20, 0.25, 0.30]
PE_PARAMS = ["pe_o5d5_Alpha", "pe_o3d10_Alpha", "pe_o7d3_Alpha", "pe_o3d1_Alpha"]
OTHER_V3 = ["Power_Ratio", "LZC_Ratio", "LZC_Alpha", "SE_Alpha"]


def project_root():
    p = Path(__file__).resolve()
    for anc in p.parents:
        if (anc / "data_results").is_dir() or (anc / "analysis_code").is_dir():
            return anc
    return p.parents[3]


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def welch_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def auc_dir(labels, scores):
    m = ~np.isnan(scores)
    if m.sum() < 4 or len(np.unique(labels[m])) < 2:
        return np.nan
    a = roc_auc_score(labels[m], scores[m])
    return max(a, 1 - a)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    pv = p[idx]
    m = len(pv)
    order = np.argsort(pv)
    prev = 1.0
    adj = np.empty(m)
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pv[i] * m / (rank + 1))
        adj[i] = prev
    out[idx] = adj
    return out


def measure_table(df, pos_grp, neg_grp, measures, pos_is_subtype=False):
    """Return DataFrame: measure, d, p, AUC over given measures for pos vs neg."""
    if pos_is_subtype:
        pos = df[df["Subtype"] == pos_grp]
    else:
        pos = df[df["Group"] == pos_grp]
    neg = df[df["Group"] == neg_grp]
    rows = []
    for mname in measures:
        if mname not in df.columns:
            continue
        a, b = pos[mname], neg[mname]
        d = cohens_d(a, b)
        p = welch_p(a, b)
        sub = pd.concat([pos, neg])
        y = (sub.index.isin(pos.index)).astype(int)
        au = auc_dir(y, sub[mname].values)
        rows.append(dict(measure=mname, d=d, p=p, AUC=au, n_pos=a.notna().sum(), n_neg=b.notna().sum()))
    t = pd.DataFrame(rows)
    t["q_BH"] = bh_fdr(t["p"].values)
    t["sig_q05"] = t["q_BH"] < 0.05
    return t


def run_real(root):
    v3p = root / "data_results" / "caueeg_entropy_v3.csv"
    swp = root / "data_results" / "se_sweep_values.csv"
    if not v3p.exists() or not swp.exists():
        print("Missing inputs:")
        print(f"  {v3p} exists={v3p.exists()}")
        print(f"  {swp} exists={swp.exists()}  (run se_parameter_sweep.py first)")
        sys.exit(1)
    v3 = pd.read_csv(v3p)
    sw = pd.read_csv(swp)
    sweep_cols = [c for c in sw.columns if c.startswith("SE_Alpha_m")]
    df = v3.merge(sw[["ID"] + sweep_cols], on="ID", how="inner")
    print(f"merged N={len(df)}; SE sweep cols={len(sweep_cols)}")

    whole = [c for c in sweep_cols if c.endswith("_whole")]
    post = [c for c in sweep_cols if c.endswith("_post")]
    grid = OTHER_V3 + PE_PARAMS + whole + post  # the unified Table-3 grid

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    for pos, neg, sub in [("Dementia", "Normal", False), ("MCI", "Normal", False), ("AD", "Normal", True)]:
        t = measure_table(df, pos, neg, grid, pos_is_subtype=sub)
        log("\n" + "=" * 78)
        log(f"{pos} vs {neg}   (one BH-FDR across {t['p'].notna().sum()} measures)")
        log("=" * 78)
        log(f"{'measure':<26}{'d':>8}{'p':>12}{'q_BH':>10}{'AUC':>7}{'sig':>5}")
        for _, r in t.sort_values('measure').iterrows():
            log(f"{r['measure']:<26}{r['d']:>+8.3f}{r['p']:>12.2g}{r['q_BH']:>10.2g}{r['AUC']:>7.3f}{'  *' if r['sig_q05'] else '':>5}")

        # SE stability vs PE swing (whole-scalp)
        se_ds = t[t["measure"].isin(whole)]["d"].dropna()
        pe_ds = t[t["measure"].isin(PE_PARAMS)]["d"].dropna()
        if len(se_ds):
            log(f"\n  SE grid (whole) d range: [{se_ds.min():+.3f}, {se_ds.max():+.3f}]  "
                f"span={se_ds.max()-se_ds.min():.3f}  all same sign={np.all(np.sign(se_ds)==np.sign(se_ds.iloc[0]))}")
        if len(pe_ds):
            log(f"  PE params   d range: [{pe_ds.min():+.3f}, {pe_ds.max():+.3f}]  span={pe_ds.max()-pe_ds.min():.3f}")
        # whole vs posterior at headline m2r20
        for tag in ["SE_Alpha_m2_r20_whole", "SE_Alpha_m2_r20_post"]:
            row = t[t["measure"] == tag]
            if len(row):
                log(f"  {tag}: d={row['d'].values[0]:+.3f} AUC={row['AUC'].values[0]:.3f}")

    # m=2 justification: d at m1/m2/m3 (r=0.20), Dementia vs Normal
    log("\n" + "=" * 78)
    log("m justification (r=0.20, whole, Dementia vs Normal)")
    log("=" * 78)
    pos = df[df["Group"] == "Dementia"]; neg = df[df["Group"] == "Normal"]
    for m in M_VALUES:
        c = f"SE_Alpha_m{m}_r20_whole"
        if c in df.columns:
            log(f"  m={m}: d={cohens_d(pos[c], neg[c]):+.3f}  (embedding spans {m} samples = {m*5} ms at 200 Hz)")

    # cross-dataset SE
    dsp = root / "data_results" / "ds004504_harmonized_values.csv"
    log("\n" + "=" * 78)
    log("Cross-dataset SE check")
    log("=" * 78)
    cae_d = cohens_d(df[df["Group"] == "Dementia"]["SE_Alpha"], df[df["Group"] == "Normal"]["SE_Alpha"])
    cae_ad = cohens_d(df[df["Subtype"] == "AD"]["SE_Alpha"], df[df["Group"] == "Normal"]["SE_Alpha"])
    log(f"  CAUEEG SE_Alpha  Dementia vs Normal d={cae_d:+.3f} ; AD vs Normal d={cae_ad:+.3f}")
    if dsp.exists():
        ds = pd.read_csv(dsp)
        if "SE_Alpha" in ds.columns:
            d = cohens_d(ds[ds["Group"] == "AD"]["SE_Alpha"], ds[ds["Group"] == "Control"]["SE_Alpha"])
            log(f"  ds004504 SE_Alpha  AD vs Control d={d:+.3f}  (downsampled-harmonized, N small)")
            log("  -> if directions disagree, state plainly: SE's CAUEEG effect does not")
            log("     replicate on the smaller, more severely affected ds004504 cohort.")
    else:
        log("  ds004504_harmonized_values.csv not found (run crossdataset_harmonize.py).")

    out = root / "data_results" / "se_sweep_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {out}")


def self_test():
    print(">>> SELF-TEST: synthetic grid, check d/AUC/BH-FDR plumbing")
    rng = np.random.default_rng(0)
    n = 200
    grp = np.array(["Normal"] * n + ["Dementia"] * n + ["MCI"] * n)
    sub = np.where(grp == "Dementia", rng.choice(["AD", "VaD"], 3 * n)[:3 * n], "")
    df = pd.DataFrame({"ID": range(3 * n), "Group": grp, "Subtype": sub, "Age": rng.normal(70, 8, 3 * n)})
    # one real effect (Power_Ratio), one null (LZC_Ratio), SE grid mild effect
    eff = (grp == "Dementia").astype(float) * 0.8 + (grp == "MCI").astype(float) * 0.3
    df["Power_Ratio"] = -eff + rng.standard_normal(3 * n)
    df["LZC_Ratio"] = rng.standard_normal(3 * n)
    df["LZC_Alpha"] = rng.standard_normal(3 * n)
    df["SE_Alpha"] = 0.5 * eff + rng.standard_normal(3 * n)
    for p in PE_PARAMS:
        df[p] = rng.standard_normal(3 * n)
    for m in M_VALUES:
        for rf in R_FACTORS:
            df[f"SE_Alpha_m{m}_r{int(rf*100):02d}_whole"] = 0.5 * eff + rng.standard_normal(3 * n)
            df[f"SE_Alpha_m{m}_r{int(rf*100):02d}_post"] = 0.5 * eff + rng.standard_normal(3 * n)
    grid = OTHER_V3 + PE_PARAMS + [c for c in df.columns if c.startswith("SE_Alpha_m")]
    t = measure_table(df, "Dementia", "Normal", grid)
    assert t["q_BH"].notna().all() and (t["q_BH"] >= t["p"] - 1e-9).all(), "BH-FDR sanity failed"
    assert bool(t[t.measure == "Power_Ratio"]["sig_q05"].values[0]), "real effect should survive FDR"
    assert not bool(t[t.measure == "LZC_Ratio"]["sig_q05"].values[0]), "null should not survive FDR"
    print("  Power_Ratio survives FDR:", bool(t[t.measure == 'Power_Ratio']['sig_q05'].values[0]))
    print("  LZC_Ratio (null) survives FDR:", bool(t[t.measure == 'LZC_Ratio']['sig_q05'].values[0]))
    print("  SE grid all significant:", bool(t[t.measure.str.startswith('SE_Alpha_m')]['sig_q05'].all()))
    print(">>> SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run_real(project_root())


if __name__ == "__main__":
    main()
