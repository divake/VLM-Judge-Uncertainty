#!/usr/bin/env python3
"""
Post-hoc analyses of per-instance CP data (saved by
rebuttal_exp4_per_instance_cp.py). Produces:
  - Exp 4: width <-> judge-error correlation + decile binning
  - Exp 5: selective-evaluation curve (vary tau)

All metrics computed both per-judge and pooled across judges.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = '/ssd_4TB/divake/VLM_Judge_Uncertainty'
os.chdir(ROOT)
SRC = 'results/rebuttal/exp4/per_instance.csv'
df = pd.read_csv(SRC)
print(f"Loaded {len(df)} per-instance records")
print(f"Judges: {sorted(df.judge.unique())}")
print(f"Seeds: {sorted(df.seed.unique())}")
print(f"Has judge_err: {df.judge_err.notna().sum()} / {len(df)}")
df = df[df.judge_err.notna()].copy()  # drop rows with no parsed_score
print(f"After dropping NaN judge_err: {len(df)}\n")

os.makedirs('results/rebuttal/exp4', exist_ok=True)

# ============================================================
# Exp 4: per-instance width vs judge-error correlation
# ============================================================
print('='*70)
print('Exp 4: Per-instance CP width vs. judge error |parsed - gt|')
print('='*70)

corr_rows = []
for judge in sorted(df.judge.unique()) + ['POOLED']:
    sub = df if judge == 'POOLED' else df[df.judge == judge]
    sp_raw, sp_raw_p = spearmanr(sub['width_raw'], sub['judge_err'])
    pe_raw, pe_raw_p = pearsonr (sub['width_raw'], sub['judge_err'])
    sp_adj, sp_adj_p = spearmanr(sub['width_adj'], sub['judge_err'])
    pe_adj, pe_adj_p = pearsonr (sub['width_adj'], sub['judge_err'])
    corr_rows.append({
        'judge': judge, 'n': len(sub),
        'spearman_raw': sp_raw,  'spearman_raw_p': sp_raw_p,
        'pearson_raw':  pe_raw,  'pearson_raw_p':  pe_raw_p,
        'spearman_adj': sp_adj,  'spearman_adj_p': sp_adj_p,
        'pearson_adj':  pe_adj,  'pearson_adj_p':  pe_adj_p,
    })
    print(f"  {judge:<14s} n={len(sub):>5d}  "
          f"Spearman(w_raw, err) = {sp_raw:+.4f} (p={sp_raw_p:.3g})  "
          f"Spearman(w_adj, err) = {sp_adj:+.4f} (p={sp_adj_p:.3g})")

pd.DataFrame(corr_rows).to_csv('results/rebuttal/exp4/correlations.csv', index=False)

# ============================================================
# Decile binning per judge (and pooled)
# ============================================================
print('\nDecile binning (judge error vs CP width quantile bins, raw width)\n')
bin_rows = []
for judge in sorted(df.judge.unique()) + ['POOLED']:
    sub = df if judge == 'POOLED' else df[df.judge == judge]
    qs = np.quantile(sub['width_raw'], np.linspace(0, 1, 11))
    sub = sub.copy()
    sub['width_bin'] = pd.cut(sub['width_raw'], bins=qs, include_lowest=True, duplicates='drop')
    g = sub.groupby('width_bin', observed=False).agg(
        n=('judge_err','size'),
        mean_width=('width_raw','mean'),
        mean_err=('judge_err','mean'),
        exact_acc=('judge_err', lambda v: float((v == 0).mean())),
        pm1_acc=('judge_err', lambda v: float((v <= 1).mean())),
        cp_cov_raw=('in_iv_raw','mean'),
    ).reset_index()
    print(f"\n  --- {judge} ---")
    print(f"    {'bin':<28s} {'n':>6s} {'mean_w':>8s} {'mean_err':>10s} {'exact_acc':>11s} {'+/-1_acc':>11s} {'cp_cov':>8s}")
    for _, r in g.iterrows():
        bin_str = str(r['width_bin'])
        print(f"    {bin_str:<28s} {int(r['n']):>6d} {r['mean_width']:>8.3f} {r['mean_err']:>10.4f} {r['exact_acc']:>11.4f} {r['pm1_acc']:>11.4f} {r['cp_cov_raw']:>8.4f}")
        bin_rows.append({'judge': judge, 'bin': bin_str,
                         'n': int(r['n']), 'mean_width': r['mean_width'],
                         'mean_err': r['mean_err'], 'exact_acc': r['exact_acc'],
                         'pm1_acc': r['pm1_acc'], 'cp_cov_raw': r['cp_cov_raw']})
pd.DataFrame(bin_rows).to_csv('results/rebuttal/exp4/width_deciles.csv', index=False)

# ============================================================
# Exp 5: selective evaluation via tau threshold (raw widths)
# ============================================================
print('\n' + '='*70)
print('Exp 5: Selective evaluation — accept calls with CP width <= tau')
print('='*70)
taus = np.arange(0.5, 4.01, 0.25)
sel_rows = []
for judge in sorted(df.judge.unique()) + ['POOLED']:
    sub = df if judge == 'POOLED' else df[df.judge == judge]
    print(f"\n  --- {judge} (n={len(sub)}) ---")
    print(f"    {'tau':>6s} {'accept%':>9s} {'exact%':>9s} {'+/-1%':>9s} {'mae':>7s} {'cp_cov%':>9s}")
    # Baseline: accept all
    base_exact = float((sub['judge_err'] == 0).mean())
    base_pm1   = float((sub['judge_err'] <= 1).mean())
    base_mae   = float(sub['judge_err'].mean())
    print(f"    {'all':>6s} {100.0:>8.1f}% {100*base_exact:>8.1f}% {100*base_pm1:>8.1f}% {base_mae:>7.4f} "
          f"{100*sub['in_iv_raw'].mean():>8.1f}%")
    sel_rows.append({'judge': judge, 'tau': 'all', 'accept_rate': 1.0,
                     'exact_acc': base_exact, 'pm1_acc': base_pm1,
                     'mae': base_mae, 'cp_cov_raw': float(sub['in_iv_raw'].mean())})
    for tau in taus:
        accepted = sub[sub['width_raw'] <= tau]
        if len(accepted) == 0:
            print(f"    {tau:>6.2f} {0.0:>8.1f}% {'-':>9s} {'-':>9s} {'-':>7s} {'-':>9s}")
            sel_rows.append({'judge': judge, 'tau': f"{tau:.2f}", 'accept_rate': 0.0,
                             'exact_acc': np.nan, 'pm1_acc': np.nan,
                             'mae': np.nan, 'cp_cov_raw': np.nan})
            continue
        accept_rate = len(accepted) / len(sub)
        exact = float((accepted['judge_err'] == 0).mean())
        pm1   = float((accepted['judge_err'] <= 1).mean())
        mae   = float(accepted['judge_err'].mean())
        cov   = float(accepted['in_iv_raw'].mean())
        print(f"    {tau:>6.2f} {100*accept_rate:>8.1f}% {100*exact:>8.1f}% {100*pm1:>8.1f}% {mae:>7.4f} {100*cov:>8.1f}%")
        sel_rows.append({'judge': judge, 'tau': f"{tau:.2f}", 'accept_rate': accept_rate,
                         'exact_acc': exact, 'pm1_acc': pm1, 'mae': mae, 'cp_cov_raw': cov})

pd.DataFrame(sel_rows).to_csv('results/rebuttal/exp4/selective_curve.csv', index=False)

# ============================================================
# Headline summary
# ============================================================
print('\n' + '='*70)
print('HEADLINE SUMMARY (for rebuttal)')
print('='*70)
df_p = df  # already filtered to judge_err not nan
total_wrong = int((df_p['judge_err'] > 0).sum())
total_correct = int((df_p['judge_err'] == 0).sum())

# "Saved by CP": of wrong judge calls, CP interval still covers
wrong = df_p[df_p['judge_err'] > 0]
saved_raw = int(wrong['in_iv_raw'].sum())
saved_adj = int(wrong['in_iv_adj'].sum())
print(f"\nPooled across all 3 judges, 10 seeds, full MLLM-Judge test sets ({len(df_p)} records, {df_p.sample_id.nunique()} unique items):")
print(f"  Judge exactly correct:      {total_correct:>6d} ({100*total_correct/len(df_p):.1f}%)")
print(f"  Judge wrong:                {total_wrong:>6d} ({100*total_wrong/len(df_p):.1f}%)")
print(f"  Of wrong judge calls:")
print(f"    Raw CP interval still contains truth:   {saved_raw}/{total_wrong}  ({100*saved_raw/total_wrong:.1f}%)")
print(f"    Adj CP interval still contains truth:   {saved_adj}/{total_wrong}  ({100*saved_adj/total_wrong:.1f}%)")
# Spearman width-vs-error pooled
sp_pool, sp_pool_p = spearmanr(df_p['width_raw'], df_p['judge_err'])
print(f"\nSpearman correlation (per-instance width_raw, judge error): {sp_pool:+.4f}  (p = {sp_pool_p:.3g})")

# Top vs bottom tertile error contrast
qs = np.quantile(df_p['width_raw'], [0, 1/3, 2/3, 1])
err_tight  = float(df_p[df_p['width_raw'] <= qs[1]]['judge_err'].mean())
err_mid    = float(df_p[(df_p['width_raw'] > qs[1]) & (df_p['width_raw'] <= qs[2])]['judge_err'].mean())
err_wide   = float(df_p[df_p['width_raw'] > qs[2]]['judge_err'].mean())
print(f"\nJudge MAE by interval-width tertile:")
print(f"  Tight tertile (width <= {qs[1]:.2f}):    MAE = {err_tight:.4f}")
print(f"  Middle tertile:                      MAE = {err_mid:.4f}")
print(f"  Wide tertile  (width > {qs[2]:.2f}):    MAE = {err_wide:.4f}")
print(f"\nRatio wide/tight MAE: {err_wide / err_tight:.2f}x")
print(f"\nFiles saved in results/rebuttal/exp4/{{correlations,width_deciles,selective_curve}}.csv")
