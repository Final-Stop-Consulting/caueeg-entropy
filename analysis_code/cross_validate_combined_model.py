"""
Cross-Validation of Combined Logistic Models
=============================================
10-fold stratified CV (repeated 10 times) for logistic regression models
combining Power Ratio, SE_Alpha, LZC Ratio, and Age as predictors.
Reports mean AUC with 95% CI for Dementia vs Normal and MCI vs Normal.

Usage: python cross_validate_combined_model.py
Requires: data_results/caueeg_entropy_v3.csv (run caueeg_entropy_v3.py first)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

# --- Load data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'data_results', 'caueeg_entropy_v3.csv')
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} subjects")
print(f"Groups: {df['Group'].value_counts().to_dict()}\n")

# --- Define models to test ---
models = {
    'Power_Ratio alone':       ['Power_Ratio'],
    'SE_Alpha alone':          ['SE_Alpha'],
    'LZC_Ratio alone':         ['LZC_Ratio'],
    'Power_Ratio + SE_Alpha':  ['Power_Ratio', 'SE_Alpha'],
    'Power_Ratio + LZC_Ratio': ['Power_Ratio', 'LZC_Ratio'],
    'Power + SE + LZC':        ['Power_Ratio', 'SE_Alpha', 'LZC_Ratio'],
    'Power + SE + Age':        ['Power_Ratio', 'SE_Alpha', 'Age'],
}

def run_cv(df_subset, y, model_features, n_splits=10, n_repeats=10):
    """Run repeated stratified k-fold CV, return mean AUC and 95% CI."""
    all_aucs = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        fold_aucs = []

        for train_idx, test_idx in skf.split(df_subset, y):
            X_train = df_subset.iloc[train_idx][model_features].values
            X_test = df_subset.iloc[test_idx][model_features].values
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            fold_aucs.append(auc)

        all_aucs.append(np.mean(fold_aucs))

    mean_auc = np.mean(all_aucs)
    ci_lo = np.percentile(all_aucs, 2.5)
    ci_hi = np.percentile(all_aucs, 97.5)
    return mean_auc, ci_lo, ci_hi

# === DEMENTIA vs NORMAL ===
print("=" * 70)
print("DEMENTIA vs NORMAL (10-fold CV, 10 repeats)")
print("=" * 70)

dn = df[df['Group'].isin(['Normal', 'Dementia'])].copy().reset_index(drop=True)
y_dn = (dn['Group'] == 'Dementia').astype(int)
print(f"N = {len(dn)} (Normal: {(y_dn==0).sum()}, Dementia: {(y_dn==1).sum()})\n")

print(f"{'Model':<30s}  {'AUC':>6s}  {'95% CI':>16s}")
print("-" * 56)

for name, features in models.items():
    mean_auc, ci_lo, ci_hi = run_cv(dn, y_dn, features)
    print(f"{name:<30s}  {mean_auc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

# === MCI vs NORMAL ===
print()
print("=" * 70)
print("MCI vs NORMAL (10-fold CV, 10 repeats)")
print("=" * 70)

mn = df[df['Group'].isin(['Normal', 'MCI'])].copy().reset_index(drop=True)
y_mn = (mn['Group'] == 'MCI').astype(int)
print(f"N = {len(mn)} (Normal: {(y_mn==0).sum()}, MCI: {(y_mn==1).sum()})\n")

print(f"{'Model':<30s}  {'AUC':>6s}  {'95% CI':>16s}")
print("-" * 56)

for name, features in models.items():
    mean_auc, ci_lo, ci_hi = run_cv(mn, y_mn, features)
    print(f"{name:<30s}  {mean_auc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

# === Apparent (training) AUCs for comparison ===
print()
print("=" * 70)
print("APPARENT (TRAINING) AUCs FOR COMPARISON")
print("=" * 70)

for comparison, (subset, y) in [("Dem vs Norm", (dn, y_dn)), ("MCI vs Norm", (mn, y_mn))]:
    print(f"\n{comparison}:")
    for name, features in models.items():
        X = subset[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(X_scaled, y.values)
        probs = clf.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y.values, probs)
        print(f"  {name:<30s}  AUC = {auc:.3f}")

print("\nDone.")
