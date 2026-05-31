#!/usr/bin/env python3
"""
Rebuttal Exp 7: Risk-coverage curve and AURC for selective evaluation.
Replaces the tau-table headline in the YU66 R2 response, which exposed a
marginal-vs-conditional coverage anomaly at small accept rates.

Setup: rank test items by a selector score (lower = more confident). For
each coverage c in (0, 1], accept the top c-fraction (most confident) and
measure mean judge error on the accepted subset. The risk-coverage curve
is risk(c) vs c; AURC = integral_0^1 risk(c) dc, lower is better.

Selectors compared:
  1. CP-width (proposed): rank by R2CCP interval width, ascending.
  2. Judge top-token margin (baseline): rank by -max(s2_lp_k), ascending
     (i.e., higher max logprob = more confident).
  3. Negative entropy of judge score distribution (baseline): rank by
     entropy of softmax(s2_lp_1..s2_lp_5), ascending.
  4. Random (reference): permuted ranking.

We also report:
  - E-AURC = AURC - AURC_oracle (oracle = rank by true error).
  - Coverage at risk thresholds, and risk at coverage thresholds.

The risk-coverage framing replaces the threshold-tau table because it
asks the correct question (does the selector RANK reliability better
than chance?), and does not depend on maintaining marginal CP coverage
on subsets, which conformal prediction does not guarantee.

Pooled across 3 judges and 10 seeds on MLLM-Judge (using per_instance.csv
from exp4) plus features_s2.csv for judge-confidence baselines.
"""
import os, sys, math
import numpy as np
import pandas as pd

ROOT = '/ssd_4TB/divake/VLM_Judge_Uncertainty'
os.chdir(ROOT)

FEATURE_FILES = {
    'LLaVA-Critic': 'results/v2/features_s2.csv',
    'Phi-4':        'results/v2_phi4/features_s2.csv',
    'Gemini':       'results/v2_gemini/features_s2.csv',
}
LP_COLS = ['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']

os.makedirs('results/rebuttal/exp7', exist_ok=True)

# Load per-instance CP records
per_inst = pd.read_csv('results/rebuttal/exp4/per_instance.csv')
per_inst = per_inst[per_inst.judge_err.notna()].copy()
print(f"Loaded {len(per_inst)} per-instance records")

# Build judge-confidence features keyed by (judge, sample_id) -> top_lp, entropy
conf_rows = []
for judge, fpath in FEATURE_FILES.items():
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    df = df[df.gt_score >= 1]
    lps = df[LP_COLS].to_numpy(dtype=np.float64)
    # softmax
    m = lps.max(axis=1, keepdims=True)
    e = np.exp(lps - m)
    p = e / e.sum(axis=1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    top_lp = lps.max(axis=1)
    for sid, t, h in zip(df['sample_id'].to_numpy(), top_lp, ent):
        conf_rows.append({'judge': judge, 'sample_id': int(sid),
                          'top_lp': float(t), 'entropy': float(h)})
conf = pd.DataFrame(conf_rows)
# Merge: each (judge, sample_id) has the same top_lp/entropy across seeds
df = per_inst.merge(conf, on=['judge', 'sample_id'], how='left')
print(f"After confidence merge: {len(df)} records ({df.top_lp.isna().sum()} missing top_lp)")
df = df[df.top_lp.notna()].copy()

# Selectors: lower selector = more confident (accept first)
df['sel_width'] = df['width_raw']
df['sel_margin'] = -df['top_lp']  # higher top_lp => more confident => smaller selector
df['sel_entropy'] = df['entropy']  # lower entropy => more confident

def risk_coverage_curve(selector, err, n_points=200):
    """Sort ascending by selector; for coverage c in (0, 1], compute mean err
    on accepted top-c items. Returns (cov_grid, risk_grid)."""
    order = np.argsort(selector, kind='mergesort')
    err_sorted = err[order]
    cum = np.cumsum(err_sorted)
    n = len(err_sorted)
    # Coverage at k items = k/n, risk at coverage = cum[k-1] / k.
    cov_full = np.arange(1, n + 1) / n
    risk_full = cum / np.arange(1, n + 1)
    # Subsample to n_points evenly in coverage for plotting / AURC.
    cov_grid = np.linspace(1.0 / n, 1.0, n_points)
    risk_grid = np.interp(cov_grid, cov_full, risk_full)
    aurc = float(np.trapz(risk_grid, cov_grid))  # integral_0^1 risk(c) dc
    return cov_grid, risk_grid, aurc

def oracle_aurc(err, n_points=200):
    """Oracle = sort ascending by true error."""
    return risk_coverage_curve(err, err, n_points)

def random_aurc(err, n_trials=20, n_points=200):
    rng = np.random.default_rng(0)
    aurcs = []
    curves = []
    for _ in range(n_trials):
        perm = rng.permutation(len(err)).astype(np.float64)
        _, risk, a = risk_coverage_curve(perm, err, n_points)
        aurcs.append(a); curves.append(risk)
    return np.linspace(1.0 / len(err), 1.0, n_points), np.mean(curves, axis=0), float(np.mean(aurcs))

# Per judge + pooled
report_rows = []
curve_rows = []
COV_GRID = np.linspace(0.05, 1.0, 20)

for judge in sorted(df['judge'].unique()) + ['POOLED']:
    sub = df if judge == 'POOLED' else df[df['judge'] == judge]
    err = sub['judge_err'].to_numpy(dtype=np.float64)
    selectors = {
        'cp_width':       sub['sel_width'].to_numpy(),
        'judge_margin':   sub['sel_margin'].to_numpy(),
        'judge_entropy':  sub['sel_entropy'].to_numpy(),
    }
    # Oracle and random reference
    cg_o, r_o, aurc_o = oracle_aurc(err)
    cg_r, r_r, aurc_r = random_aurc(err)
    # Mean error (baseline; equals risk at coverage=1)
    base_err = float(err.mean())
    print(f"\n=== {judge}  (n={len(sub)}, baseline MAE={base_err:.4f}) ===")
    print(f"  {'selector':<16s} {'AURC':>8s} {'E-AURC':>8s} "
          f"{'risk@10%':>10s} {'risk@25%':>10s} {'risk@50%':>10s} {'risk@75%':>10s}")
    print(f"  {'oracle':<16s} {aurc_o:>8.4f} {0.0:>8.4f}")
    print(f"  {'random':<16s} {aurc_r:>8.4f} {aurc_r - aurc_o:>8.4f}")
    for sname, sel in selectors.items():
        cg, rg, aurc = risk_coverage_curve(sel, err)
        # Interpolate risk at standard coverage points
        r_at = {c: float(np.interp(c, cg, rg)) for c in [0.10, 0.25, 0.50, 0.75]}
        print(f"  {sname:<16s} {aurc:>8.4f} {aurc - aurc_o:>8.4f} "
              f"{r_at[0.10]:>10.4f} {r_at[0.25]:>10.4f} {r_at[0.50]:>10.4f} {r_at[0.75]:>10.4f}")
        report_rows.append({
            'judge': judge, 'selector': sname,
            'aurc': aurc, 'aurc_oracle': aurc_o, 'aurc_random': aurc_r,
            'e_aurc': aurc - aurc_o,
            'improvement_over_random': aurc_r - aurc,
            'risk_at_10pct': r_at[0.10], 'risk_at_25pct': r_at[0.25],
            'risk_at_50pct': r_at[0.50], 'risk_at_75pct': r_at[0.75],
            'baseline_mae': base_err, 'n': len(sub),
        })
        # Save curve at standard grid
        for c in COV_GRID:
            curve_rows.append({
                'judge': judge, 'selector': sname,
                'coverage': float(c),
                'risk': float(np.interp(c, cg, rg)),
            })
    # Add reference curves to curve table
    for c in COV_GRID:
        curve_rows.append({'judge': judge, 'selector': 'oracle',
                           'coverage': float(c), 'risk': float(np.interp(c, cg_o, r_o))})
        curve_rows.append({'judge': judge, 'selector': 'random',
                           'coverage': float(c), 'risk': float(np.interp(c, cg_r, r_r))})

pd.DataFrame(report_rows).to_csv('results/rebuttal/exp7/aurc_summary.csv', index=False)
pd.DataFrame(curve_rows).to_csv('results/rebuttal/exp7/risk_coverage_curves.csv', index=False)
print(f"\nSaved: results/rebuttal/exp7/{{aurc_summary, risk_coverage_curves}}.csv")
