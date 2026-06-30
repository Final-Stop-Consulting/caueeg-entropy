"""
Cross-dataset convergence test
==============================
Compares harmonized, physical-timescale-matched PE across CAUEEG (200 Hz) and
ds004504 (downsampled 500->200 Hz) and applies the pre-committed decision rule.

CAUEEG side reuses published v3 per-subject values:
    probable-AD = Subtype == "AD"      (n=226)
    Normal      = Group   == "Normal"  (n=457)
ds004504 side from crossdataset_harmonize.py:
    AD = Group == "AD"   Control = Group == "Control"

Headline parameterization = pe_o5d5_Alpha (proper alpha, 100 ms).
Direction of d: (AD mean - control/normal mean) / pooled SD.

Decision thresholds (headline parameterization = pe_o5d5):
    A  sign/null agreement on pe_o5d5:  signs agree OR max(|d_C|,|d_d|) <= 0.20
    B  effect-size overlap  on pe_o5d5:  95% bootstrap CIs overlap OR |dC-dd| <= 0.30
    C  d-profile concordance:            Spearman rho >= 0.60 over the 4 canonical params
    DECISION: P1 iff A and B and C, else P0.

Note: the binary P1/P0 label is a heuristic decision aid. The substantive outputs are
the per-parameter effect sizes, their bootstrap CIs, and the cross-dataset d-profile
concordance (Pearson/Spearman). Interpret those directly; the binary label can pass
degenerately when one dataset is null and the other is small-N (e.g. a sign match
between a near-zero effect and a large one, or a CI overlap driven by small N).

Run:
    python crossdataset_convergence.py
    python crossdataset_convergence.py --self-test   # validate logic on synthetic data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260629)
N_BOOT = 10000
CANONICAL = ["pe_o5d5", "pe_o3d10", "pe_o7d3", "pe_o3d1"]
HEADLINE = "pe_o5d5"

# locked thresholds
A_NULL_BAND = 0.20
B_DELTA = 0.30
C_RHO = 0.60


# ----------------------------------------------------------------------------
def cohens_d(ad, ctrl):
    """(AD mean - control mean) / pooled SD."""
    ad, ctrl = np.asarray(ad, float), np.asarray(ctrl, float)
    ad, ctrl = ad[~np.isnan(ad)], ctrl[~np.isnan(ctrl)]
    if len(ad) < 2 or len(ctrl) < 2:
        return np.nan
    sp = np.sqrt((ad.var(ddof=1) + ctrl.var(ddof=1)) / 2)
    return (ad.mean() - ctrl.mean()) / sp if sp > 0 else np.nan


def welch_p(ad, ctrl):
    ad, ctrl = np.asarray(ad, float), np.asarray(ctrl, float)
    ad, ctrl = ad[~np.isnan(ad)], ctrl[~np.isnan(ctrl)]
    if len(ad) < 2 or len(ctrl) < 2:
        return np.nan
    return stats.ttest_ind(ad, ctrl, equal_var=False).pvalue


def bca_ci(ad, ctrl, alpha=0.05):
    """BCa bootstrap CI for Cohen's d (resample each group independently)."""
    ad = np.asarray(ad, float); ad = ad[~np.isnan(ad)]
    ctrl = np.asarray(ctrl, float); ctrl = ctrl[~np.isnan(ctrl)]
    if len(ad) < 3 or len(ctrl) < 3:
        return (np.nan, np.nan)
    theta_hat = cohens_d(ad, ctrl)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = cohens_d(RNG.choice(ad, len(ad), replace=True),
                            RNG.choice(ctrl, len(ctrl), replace=True))
    boots = boots[~np.isnan(boots)]
    if boots.size < 100:
        return (np.nan, np.nan)
    # bias correction
    z0 = stats.norm.ppf((boots < theta_hat).mean()) if 0 < (boots < theta_hat).mean() < 1 else 0.0
    # acceleration via jackknife over the pooled sample
    pooled = np.concatenate([ad, ctrl])
    n_ad = len(ad)
    jack = []
    for i in range(len(pooled)):
        m = np.ones(len(pooled), bool); m[i] = False
        s = pooled[m]
        jack.append(cohens_d(s[:n_ad - (i < n_ad)], s[n_ad - (i < n_ad):]))
    jack = np.asarray(jack, float); jack = jack[~np.isnan(jack)]
    jbar = jack.mean()
    denom = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = (((jbar - jack) ** 3).sum()) / denom if denom != 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.percentile(boots, 100 * adj(zl))
    hi = np.percentile(boots, 100 * adj(zu))
    return (lo, hi)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    pv = p[idx]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pv[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = prev
    out[idx] = adj
    return out


def age_residualized_d(df_ad, df_ctrl, col, age="Age"):
    """Regress measure on age across pooled groups, compare residual means as d."""
    a = df_ad[[col, age]].dropna(); c = df_ctrl[[col, age]].dropna()
    if len(a) < 3 or len(c) < 3:
        return np.nan
    pooled = pd.concat([a, c])
    b1, b0 = np.polyfit(pooled[age], pooled[col], 1)
    ra = a[col] - (b0 + b1 * a[age])
    rc = c[col] - (b0 + b1 * c[age])
    sp = np.sqrt((ra.var(ddof=1) + rc.var(ddof=1)) / 2)
    return (ra.mean() - rc.mean()) / sp if sp > 0 else np.nan


# ----------------------------------------------------------------------------
def load_caueeg(path):
    df = pd.read_csv(path)
    ad = df[df["Subtype"] == "AD"].copy()
    norm = df[df["Group"] == "Normal"].copy()
    return ad, norm


def load_ds004504(path):
    df = pd.read_csv(path)
    ad = df[df["Group"] == "AD"].copy()
    ctrl = df[df["Group"] == "Control"].copy()
    return ad, ctrl


def analyze(ad, ctrl, label):
    rows = {}
    for p in CANONICAL:
        col = f"{p}_Alpha"
        if col not in ad.columns:
            rows[p] = dict(d=np.nan, p=np.nan, lo=np.nan, hi=np.nan, d_age=np.nan)
            continue
        d = cohens_d(ad[col], ctrl[col])
        pv = welch_p(ad[col], ctrl[col])
        lo, hi = bca_ci(ad[col], ctrl[col])
        d_age = age_residualized_d(ad, ctrl, col) if "Age" in ad.columns else np.nan
        rows[p] = dict(d=d, p=pv, lo=lo, hi=hi, d_age=d_age)
    print(f"\n[{label}]  AD n={len(ad)}  control n={len(ctrl)}")
    print(f"  {'param':<10}{'d':>8}{'95% CI':>20}{'p':>10}{'d_age':>8}")
    for p in CANONICAL:
        r = rows[p]
        ci = f"[{r['lo']:+.2f},{r['hi']:+.2f}]" if not np.isnan(r['lo']) else "   --   "
        print(f"  {p:<10}{r['d']:>+8.3f}{ci:>20}{r['p']:>10.4g}{r['d_age']:>+8.3f}")
    return rows


def evaluate(cae, dsr):
    dC = cae[HEADLINE]["d"]; dD = dsr[HEADLINE]["d"]
    # A
    sign_agree = np.sign(dC) == np.sign(dD)
    shared_null = max(abs(dC), abs(dD)) <= A_NULL_BAND
    A = bool(sign_agree or shared_null)
    # B
    ci_overlap = not (cae[HEADLINE]["hi"] < dsr[HEADLINE]["lo"] or
                      dsr[HEADLINE]["hi"] < cae[HEADLINE]["lo"])
    delta_ok = abs(dC - dD) <= B_DELTA
    B = bool(ci_overlap or delta_ok)
    # C
    vC = [cae[p]["d"] for p in CANONICAL]
    vD = [dsr[p]["d"] for p in CANONICAL]
    rho, _ = stats.spearmanr(vC, vD)
    pear, _ = stats.pearsonr(vC, vD)
    C = bool(rho >= C_RHO)
    decision = "P1 (convergence)" if (A and B and C) else "P0 (discrepancy survives)"

    print("\n" + "=" * 64)
    print("CONVERGENCE EVALUATION  (locked thresholds)")
    print("=" * 64)
    print(f"  headline pe_o5d5:  d_CAUEEG={dC:+.3f}   d_ds004504={dD:+.3f}")
    print(f"  A sign/null  : {'PASS' if A else 'FAIL'}  (sign_agree={sign_agree}, shared_null|d|<= {A_NULL_BAND}={shared_null})")
    print(f"  B overlap    : {'PASS' if B else 'FAIL'}  (CI_overlap={ci_overlap}, |dC-dD|={abs(dC-dD):.3f}<= {B_DELTA}={delta_ok})")
    print(f"  C d-profile  : {'PASS' if C else 'FAIL'}  (Spearman rho={rho:+.3f}>= {C_RHO}; Pearson r={pear:+.3f})")
    print(f"\n  DECISION: {decision}")
    print("=" * 64)
    return decision, dict(A=A, B=B, C=C, rho=rho, pearson=pear, dC=dC, dD=dD)


# ----------------------------------------------------------------------------
def self_test():
    """Synthetic check of the stats + decision rule (no real data needed)."""
    print(">>> SELF-TEST: fabricating data for a known P0 and a known P1 scenario\n")
    def make(d_target, n_ad=36, n_ctrl=29, seed=1):
        r = np.random.default_rng(seed)
        return r.normal(d_target, 1, n_ad), r.normal(0, 1, n_ctrl)

    # P0: CAUEEG null on headline, ds004504 strongly positive; profiles discordant
    cae = {p: None for p in CANONICAL}; dsr = {p: None for p in CANONICAL}
    targC = {"pe_o5d5": -0.02, "pe_o3d10": 0.71, "pe_o7d3": 0.01, "pe_o3d1": -0.70}
    targD = {"pe_o5d5": 0.72, "pe_o3d10": 1.11, "pe_o7d3": 0.74, "pe_o3d1": 0.45}
    for i, p in enumerate(CANONICAL):
        a, c = make(targC[p], 226, 457, seed=10 + i)
        cae[p] = dict(d=cohens_d(a, c), p=welch_p(a, c), **dict(zip(("lo", "hi"), bca_ci(a, c))), d_age=np.nan)
        a, c = make(targD[p], 36, 29, seed=50 + i)
        dsr[p] = dict(d=cohens_d(a, c), p=welch_p(a, c), **dict(zip(("lo", "hi"), bca_ci(a, c))), d_age=np.nan)
    print("Scenario P0 (expected): sign flip on headline, low profile concordance")
    dec, _ = evaluate(cae, dsr)
    assert dec.startswith("P0"), "self-test P0 scenario failed"

    # P1: both null on headline, concordant profiles
    targD2 = {"pe_o5d5": -0.05, "pe_o3d10": 0.68, "pe_o7d3": 0.04, "pe_o3d1": -0.66}
    for i, p in enumerate(CANONICAL):
        a, c = make(targD2[p], 36, 29, seed=80 + i)
        dsr[p] = dict(d=cohens_d(a, c), p=welch_p(a, c), **dict(zip(("lo", "hi"), bca_ci(a, c))), d_age=np.nan)
    print("\nScenario P1 (expected): shared null headline, concordant profile")
    dec, _ = evaluate(cae, dsr)
    assert dec.startswith("P1"), "self-test P1 scenario failed"
    print("\n>>> SELF-TEST PASSED: both scenarios decided correctly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caueeg", default=None)
    ap.add_argument("--ds004504", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    def project_root():
        p = Path(__file__).resolve()
        for anc in p.parents:
            if (anc / "data_results").is_dir() or (anc / "analysis_code").is_dir():
                return anc
        return p.parents[3]
    base = project_root()  # caueeg_paper
    cae_path = Path(args.caueeg) if args.caueeg else base / "data_results" / "caueeg_entropy_v3.csv"
    ds_path = Path(args.ds004504) if args.ds004504 else base / "data_results" / "ds004504_harmonized_values.csv"
    if not cae_path.exists() or not ds_path.exists():
        print("Missing input(s):")
        print(f"  CAUEEG v3 csv : {cae_path}  exists={cae_path.exists()}")
        print(f"  ds004504 csv  : {ds_path}  exists={ds_path.exists()}  (run crossdataset_harmonize.py first)")
        print("\nTip: python crossdataset_convergence.py --self-test  to validate the logic without data.")
        sys.exit(1)

    ad_c, norm_c = load_caueeg(cae_path)
    ad_d, ctrl_d = load_ds004504(ds_path)
    cae = analyze(ad_c, norm_c, "CAUEEG 200 Hz  (probable-AD vs Normal)")
    dsr = analyze(ad_d, ctrl_d, "ds004504 ->200 Hz  (AD vs Control)")
    decision, summ = evaluate(cae, dsr)

    out = base / "data_results" / "crossdataset_convergence_report.txt"
    with open(out, "w") as f:
        f.write("Cross-dataset convergence (binary label is a heuristic; read the effect sizes)\n")
        f.write(f"DECISION (heuristic): {decision}\n")
        f.write(f"headline pe_o5d5: d_CAUEEG={summ['dC']:+.3f} d_ds004504={summ['dD']:+.3f}\n")
        f.write(f"A={summ['A']} B={summ['B']} C={summ['C']} "
                f"spearman={summ['rho']:+.3f} pearson={summ['pearson']:+.3f}\n")
    print(f"\nReport -> {out}")
    print(">>> Interpret the per-parameter effect sizes and the d-profile concordance above.")


if __name__ == "__main__":
    main()
