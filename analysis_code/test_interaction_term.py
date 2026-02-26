"""
Interaction Term Test
=====================
Tests whether a Power_Ratio × SE_Alpha multiplicative interaction improves
the combined logistic model. Cross-validated AUCs with and without interaction,
plus Wald test for coefficient significance.

Usage: python test_interaction_term.py
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

def run_cv(df_subset, y, feature_names, n_splits=10, n_repeats=10):
    """Run repeated stratified k-fold CV, return mean AUC and 95% CI."""
    all_aucs = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        fold_aucs = []

        for train_idx, test_idx in skf.split(df_subset, y):
            X_train = df_subset.iloc[train_idx][feature_names].values
            X_test = df_subset.iloc[test_idx][feature_names].values
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

def get_coefficients(df_subset, y, feature_names):
    """Fit on full data and return coefficients + p-values via Wald test."""
    X = df_subset[feature_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_scaled, y.values)

    # Wald test for coefficient significance
    # Compute Hessian-based standard errors
    probs = clf.predict_proba(X_scaled)[:, 1]
    W = np.diag(probs * (1 - probs))

    # Add intercept column
    X_design = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

    try:
        # Fisher information matrix
        fisher = X_design.T @ W @ X_design
        cov = np.linalg.inv(fisher)

        # Standard errors (skip intercept)
        se = np.sqrt(np.diag(cov)[1:])

        # Wald z-statistics
        z = clf.coef_[0] / se

        # Two-tailed p-values
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z)))
    except np.linalg.LinAlgError:
        se = np.full(len(feature_names), np.nan)
        z = np.full(len(feature_names), np.nan)
        p_values = np.full(len(feature_names), np.nan)

    auc = roc_auc_score(y.values, probs)

    return clf.coef_[0], clf.intercept_[0], se, z, p_values, auc

# === DEMENTIA vs NORMAL ===
for comparison_name, group_pair, pos_label in [
    ("DEMENTIA vs NORMAL", ['Normal', 'Dementia'], 'Dementia'),
    ("MCI vs NORMAL", ['Normal', 'MCI'], 'MCI'),
]:
    print("=" * 70)
    print(f"{comparison_name}")
    print("=" * 70)

    subset = df[df['Group'].isin(group_pair)].copy().reset_index(drop=True)
    y = (subset['Group'] == pos_label).astype(int)
    print(f"N = {len(subset)}\n")

    # Create interaction term
    subset['Power_x_SE'] = subset['Power_Ratio'] * subset['SE_Alpha']

    models = {
        'Power + SE (no interaction)':    ['Power_Ratio', 'SE_Alpha'],
        'Power + SE + interaction':       ['Power_Ratio', 'SE_Alpha', 'Power_x_SE'],
        'Power + SE + Age':               ['Power_Ratio', 'SE_Alpha', 'Age'],
        'Power + SE + Age + interaction': ['Power_Ratio', 'SE_Alpha', 'Age', 'Power_x_SE'],
    }

    # Cross-validated AUCs
    print("Cross-validated AUCs:")
    print(f"{'Model':<35s}  {'AUC':>6s}  {'95% CI':>16s}")
    print("-" * 62)
    for name, features in models.items():
        mean_auc, ci_lo, ci_hi = run_cv(subset, y, features)
        print(f"{name:<35s}  {mean_auc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Coefficient analysis for the interaction model
    print(f"\nCoefficient analysis (full-data fit):")
    for name, features in models.items():
        coefs, intercept, se, z, pvals, auc = get_coefficients(subset, y, features)
        print(f"\n  {name} (apparent AUC = {auc:.3f}):")
        print(f"    {'Feature':<20s}  {'Coef':>8s}  {'SE':>8s}  {'z':>8s}  {'p':>10s}")
        print(f"    {'-'*58}")
        for i, feat in enumerate(features):
            p_str = f"{pvals[i]:.4f}" if pvals[i] >= 0.0001 else "< 0.0001"
            print(f"    {feat:<20s}  {coefs[i]:>8.4f}  {se[i]:>8.4f}  {z[i]:>8.2f}  {p_str:>10s}")

    print()

print("Done.")
