"""
Generate Publication Figures
============================
Produces 4 figures in figures/:

  1. PE parameterization sensitivity (Cohen's d with bootstrap 95% CIs)
  2. Embedding window schematic (sub-cycle vs proper alpha on synthetic waveform)
  3. Head-to-head measure comparison (all entropy + spectral measures)
  4. Age correction impact (raw vs age-residualized effect sizes)

Usage: python generate_figures.py
Requires: data_results/caueeg_entropy_v3.csv (run caueeg_entropy_v3.py first)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

# --- Config ---
DATA_DIR = Path(__file__).parent.parent / "data_results"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "caueeg_entropy_v3.csv"
if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} not found. Run caueeg_entropy_v3.py first.")
    exit(1)

print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"N = {len(df)}, columns: {len(df.columns)}")


# --- Helper functions ---
def cohens_d(a, b):
    na, nb = len(a), len(b)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0


def compute_d_and_ci(df, col, g1='Dementia', g2='Normal', n_boot=1000):
    """Compute Cohen's d with bootstrap 95% CI."""
    v1 = df[df['Group'] == g1][col].dropna().values
    v2 = df[df['Group'] == g2][col].dropna().values
    d = cohens_d(v1, v2)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_ds = []
    for _ in range(n_boot):
        b1 = rng.choice(v1, size=len(v1), replace=True)
        b2 = rng.choice(v2, size=len(v2), replace=True)
        boot_ds.append(cohens_d(b1, b2))
    ci_lo, ci_hi = np.percentile(boot_ds, [2.5, 97.5])
    return d, ci_lo, ci_hi


# =====================================================================
# FIGURE 1: PE Parameterization Sensitivity
# =====================================================================
print("\nGenerating Figure 1: PE Parameterization Sensitivity...")

pe_configs = [
    ('pe_o3d1_Alpha', 'o3d1\n(10ms, 6 states)', '#d62728'),
    ('pe_o5d5_Alpha', 'o5d5\n(100ms, 120 states)', '#2ca02c'),
    ('pe_o3d10_Alpha', 'o3d10\n(100ms, 6 states)', '#ff7f0e'),
    ('pe_o7d3_Alpha', 'o7d3\n(90ms, 5040 states)', '#1f77b4'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, (g1, g2, title) in zip(axes, [
    ('Dementia', 'Normal', 'Dementia vs Normal'),
    ('MCI', 'Normal', 'MCI vs Normal'),
]):
    ds = []
    ci_los = []
    ci_his = []
    labels = []
    colors = []

    for col, label, color in pe_configs:
        if col in df.columns:
            d, ci_lo, ci_hi = compute_d_and_ci(df, col, g1, g2)
            ds.append(d)
            ci_los.append(d - ci_lo)
            ci_his.append(ci_hi - d)
            labels.append(label)
            colors.append(color)

    x = np.arange(len(ds))
    bars = ax.bar(x, ds, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.errorbar(x, ds, yerr=[ci_los, ci_his], fmt='none', ecolor='black',
                capsize=5, linewidth=1.5)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axhline(y=0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=-0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel("Cohen's d" if ax == axes[0] else "", fontsize=11)

    # Annotate d values
    for i, d in enumerate(ds):
        offset = 0.05 if d >= 0 else -0.05
        va = 'bottom' if d >= 0 else 'top'
        ax.text(i, d + offset, f'd={d:.2f}', ha='center', va=va, fontsize=8,
                fontweight='bold')

axes[0].set_ylim(-1.0, 1.0)
fig.suptitle('PE Alpha-Band Effect Sizes by Parameterization\n'
             '(Same data, same band, different parameters)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_pe_sensitivity.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'fig1_pe_sensitivity.png'}")


# =====================================================================
# FIGURE 2: Embedding Window Schematic
# =====================================================================
print("Generating Figure 2: Embedding Window Schematic...")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})

t = np.linspace(0, 0.3, 3000)  # 300 ms of signal
sfreq = 200
# Simulated alpha wave: 10 Hz sine with slight amplitude modulation
alpha_wave = np.sin(2 * np.pi * 10 * t) * (1 + 0.15 * np.sin(2 * np.pi * 1.5 * t))

# Panel A: pe_o3d1 (sub-cycle, 10ms window)
ax = axes[0]
ax.plot(t * 1000, alpha_wave, 'b-', linewidth=1.5, alpha=0.7)
ax.set_title('A) pe_o3d1: order=3, delay=1 → 10 ms window (sub-cycle)',
             fontsize=11, fontweight='bold')

# Show embedding window at a sample point
t_center = 0.125  # seconds
sample_times = [t_center, t_center + 1/sfreq, t_center + 2/sfreq]
sample_vals = np.sin(2 * np.pi * 10 * np.array(sample_times)) * \
              (1 + 0.15 * np.sin(2 * np.pi * 1.5 * np.array(sample_times)))

# Highlight the 10ms window
ax.axvspan(sample_times[0]*1000, sample_times[-1]*1000, alpha=0.3, color='red',
           label='10 ms embedding window')
ax.plot(np.array(sample_times)*1000, sample_vals, 'ro', markersize=8, zorder=5)
for i, (st, sv) in enumerate(zip(sample_times, sample_vals)):
    ax.annotate(f's{i+1}', (st*1000, sv), textcoords="offset points",
                xytext=(5, 10), fontsize=9, color='red', fontweight='bold')

ax.set_ylabel('Amplitude (a.u.)', fontsize=10)
ax.set_xlim(50, 250)
ax.legend(loc='upper right', fontsize=9)
ax.text(0.02, 0.95, 'Only 10% of one alpha cycle\n→ Measures local curvature, not complexity',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel B: pe_o5d5 (full-cycle, 100ms window)
ax = axes[1]
ax.plot(t * 1000, alpha_wave, 'b-', linewidth=1.5, alpha=0.7)
ax.set_title('B) pe_o5d5: order=5, delay=5 → 100 ms window (full alpha cycle)',
             fontsize=11, fontweight='bold')

# 5 sample points with delay=5 (every 25ms)
sample_times_5 = [t_center + i * 5/sfreq for i in range(5)]
sample_vals_5 = np.sin(2 * np.pi * 10 * np.array(sample_times_5)) * \
                (1 + 0.15 * np.sin(2 * np.pi * 1.5 * np.array(sample_times_5)))

ax.axvspan(sample_times_5[0]*1000, sample_times_5[-1]*1000, alpha=0.2, color='green',
           label='100 ms embedding window')
ax.plot(np.array(sample_times_5)*1000, sample_vals_5, 'go', markersize=8, zorder=5)
for i, (st, sv) in enumerate(zip(sample_times_5, sample_vals_5)):
    ax.annotate(f's{i+1}', (st*1000, sv), textcoords="offset points",
                xytext=(5, 10 if i % 2 == 0 else -15), fontsize=9,
                color='green', fontweight='bold')

ax.set_xlabel('Time (ms)', fontsize=10)
ax.set_ylabel('Amplitude (a.u.)', fontsize=10)
ax.set_xlim(50, 250)
ax.legend(loc='upper right', fontsize=9)
ax.text(0.02, 0.95, 'Spans 1 full alpha cycle (10 Hz)\n→ Measures true ordinal complexity',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

fig.suptitle('What PE Actually Measures Depends on the Embedding Window',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_embedding_windows.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'fig2_embedding_windows.png'}")


# =====================================================================
# FIGURE 3: Head-to-Head Measure Comparison
# =====================================================================
print("Generating Figure 3: Head-to-Head Measure Comparison...")

# Determine which measures are available
candidate_measures = [
    ('pe_o3d1_Alpha', 'PE (o3d1)', '#d62728'),
    ('pe_o5d5_Alpha', 'PE (o5d5)', '#2ca02c'),
    ('LZC_Ratio', 'LZC Ratio', '#9467bd'),
    ('SE_Alpha', 'SE Alpha', '#8c564b'),
    ('LZC_Alpha', 'LZC Alpha', '#e377c2'),
]

# Add spectral if available
if 'Power_Ratio' in df.columns:
    candidate_measures.append(('Power_Ratio', 'Power Ratio', '#17becf'))
if 'Alpha_Power_Rel' in df.columns:
    candidate_measures.append(('Alpha_Power_Rel', 'Rel. Alpha Power', '#bcbd22'))

measures = [(col, lab, c) for col, lab, c in candidate_measures if col in df.columns]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, (g1, g2, title) in zip(axes, [
    ('Dementia', 'Normal', 'Dementia vs Normal'),
    ('MCI', 'Normal', 'MCI vs Normal'),
]):
    ds = []
    ci_los = []
    ci_his = []
    labels = []
    colors = []

    for col, label, color in measures:
        d, ci_lo, ci_hi = compute_d_and_ci(df, col, g1, g2)
        ds.append(d)
        ci_los.append(d - ci_lo)
        ci_his.append(ci_hi - d)
        labels.append(label)
        colors.append(color)

    x = np.arange(len(ds))
    bars = ax.barh(x, ds, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                   height=0.6)
    ax.errorbar(ds, x, xerr=[ci_los, ci_his], fmt='none', ecolor='black',
                capsize=4, linewidth=1.2)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Cohen's d (with 95% bootstrap CI)", fontsize=10)

    # Annotate
    for i, d in enumerate(ds):
        offset = 0.03 if d >= 0 else -0.03
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + offset, i, f'{d:.2f}', va='center', ha=ha, fontsize=8,
                fontweight='bold')

fig.suptitle('Effect Sizes for All Measures (v3 Corrected Pipeline)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_headtohead.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'fig3_headtohead.png'}")


# =====================================================================
# FIGURE 4: Age Correction Impact
# =====================================================================
print("Generating Figure 4: Age Correction Impact...")

# Age-residualize
age = df['Age'].values
X_age = np.column_stack([np.ones(len(age)), age])
XtX_inv = np.linalg.pinv(X_age.T @ X_age)

age_measures = [
    ('pe_o3d1_Alpha', 'PE (o3d1)'),
    ('pe_o5d5_Alpha', 'PE (o5d5)'),
    ('LZC_Ratio', 'LZC Ratio'),
    ('SE_Alpha', 'SE Alpha'),
]

if 'Power_Ratio' in df.columns:
    age_measures.append(('Power_Ratio', 'Power Ratio'))
if 'Alpha_Power_Rel' in df.columns:
    age_measures.append(('Alpha_Power_Rel', 'Rel. Alpha Pwr'))

# Filter to available
age_measures = [(c, l) for c, l in age_measures if c in df.columns]

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(age_measures))
width = 0.35

d_raw_list = []
d_corr_list = []
labels = []

for col, label in age_measures:
    vals = df[col].values
    valid = ~np.isnan(vals)
    betas = XtX_inv @ X_age[valid].T @ vals[valid]
    resid = np.full_like(vals, np.nan)
    resid[valid] = vals[valid] - X_age[valid] @ betas

    v1_raw = vals[(df['Group'] == 'Dementia').values & valid]
    v2_raw = vals[(df['Group'] == 'Normal').values & valid]
    v1_r = resid[(df['Group'] == 'Dementia').values & valid]
    v2_r = resid[(df['Group'] == 'Normal').values & valid]

    d_raw = cohens_d(v1_raw, v2_raw)
    d_corr = cohens_d(v1_r, v2_r)

    d_raw_list.append(d_raw)
    d_corr_list.append(d_corr)
    labels.append(label)

bars1 = ax.bar(x - width/2, d_raw_list, width, label='Raw', color='#4c72b0',
               alpha=0.85, edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, d_corr_list, width, label='Age-corrected', color='#dd8452',
               alpha=0.85, edgecolor='black', linewidth=0.8)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Cohen's d (Dementia vs Normal)", fontsize=11)
ax.set_title('Impact of Age Correction on Effect Sizes', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# Annotate
for i, (dr, dc) in enumerate(zip(d_raw_list, d_corr_list)):
    for d_val, x_off in [(dr, -width/2), (dc, width/2)]:
        offset = 0.03 if d_val >= 0 else -0.03
        va = 'bottom' if d_val >= 0 else 'top'
        ax.text(i + x_off, d_val + offset, f'{d_val:.2f}', ha='center', va=va,
                fontsize=8, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_age_correction.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'fig4_age_correction.png'}")


print("\nAll figures generated successfully!")
print(f"Output directory: {FIG_DIR}")
