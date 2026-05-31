#!/usr/bin/env python3
"""
Per-task selective evaluation analysis (post-hoc on existing exp4 data).

Goal: address YU66 implicit concern about breadth + the explicit critique
that intervals don't carry instance-level information. Strategy:

  Show that selective evaluation works very differently by task type:
    - On tighter-interval tasks (AesBench, MM-Vet), modest tau yields
      high accept rate AND high accuracy on accepted.
    - On wide-interval tasks (ChartQA, MathVista, Infographics), even
      with tight tau accept rate is tiny — which is the diagnostic
      correctly saying "don't use Likert here, use pairwise."

This refutes the implicit complaint "the wide intervals on the other 92%
of items are garbage" by showing the wide intervals correctly identify
the *tasks* where Likert scoring should be replaced.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = '/ssd_4TB/divake/VLM_Judge_Uncertainty'
os.chdir(ROOT)

df = pd.read_csv('results/rebuttal/exp4/per_instance.csv')
df = df[df.judge_err.notna()].copy()
print(f"Loaded {len(df)} records (judge_err non-null)")

# Load MLLM-Judge metadata: sample_id -> original_dataset
META = {}
with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
    for i, line in enumerate(f):
        META[i] = json.loads(line).get('original_dataset', 'unknown')

df['orig_dataset'] = df['sample_id'].map(lambda s: META.get(int(s), 'unknown'))
print(f"Unique datasets in records: {df.orig_dataset.nunique()}")

# Paper's 4-way taxonomy (same as in scripts/run_r2ccp_per_dataset.py)
CATEGORIES = {
    'Vision-Heavy':  ['mathvista', 'ChartQA', 'infographicsVQA', 'ScienceQA', 'textVQA'],
    'General VQA':   ['coco', 'llava_bench', 'mm-vet', 'VisitBench'],
    'Knowledge/Web': ['WIT', 'Concept Caption', 'mind2web'],
    'Aesthetics/AI': ['AesBench', 'diffusiondb'],
}
ds_to_cat = {ds: cat for cat, lst in CATEGORIES.items() for ds in lst}
df['category'] = df['orig_dataset'].map(lambda d: ds_to_cat.get(d, 'Other'))
print(df.groupby('category').size())

os.makedirs('results/rebuttal/exp4', exist_ok=True)

# ============================================================
# Per-category: Spearman, baseline acc, selective at multiple tau
# ============================================================
print("\n" + "="*90)
print("Per-category selective evaluation (pooled across 3 judges, 10 seeds)")
print("="*90)
TAUS = [2.0, 2.25, 2.5, 3.0]
cat_rows = []
for cat in sorted(df['category'].unique()):
    sub = df[df.category == cat]
    if len(sub) < 100:
        continue
    sp, sp_p = spearmanr(sub['width_raw'], sub['judge_err'])
    base_exact = float((sub['judge_err'] == 0).mean())
    base_pm1   = float((sub['judge_err'] <= 1).mean())
    base_mae   = float(sub['judge_err'].mean())
    avg_w      = float(sub['width_raw'].mean())
    row = {
        'category': cat, 'n': len(sub),
        'avg_width_raw': avg_w, 'spearman': sp, 'spearman_p': sp_p,
        'base_exact': base_exact, 'base_pm1': base_pm1, 'base_mae': base_mae,
    }
    print(f"\n--- {cat} (n={len(sub)}) ---")
    print(f"  avg_width = {avg_w:.3f},  Spearman(w, err) = {sp:+.4f}  (p={sp_p:.3g})")
    print(f"  Baseline (accept all): exact={100*base_exact:.1f}%  +/-1={100*base_pm1:.1f}%  MAE={base_mae:.4f}")
    print(f"  {'tau':>6s} {'accept%':>10s} {'exact%':>9s} {'+/-1%':>9s} {'mae':>7s} {'cp_cov%':>9s}")
    for tau in TAUS:
        accepted = sub[sub['width_raw'] <= tau]
        if len(accepted) == 0:
            print(f"  {tau:>6.2f} {0.0:>9.1f}%  ----- ----- ----- -----")
            row[f'accept_{tau}'] = 0.0
            row[f'exact_{tau}']  = np.nan
            row[f'pm1_{tau}']    = np.nan
            row[f'mae_{tau}']    = np.nan
            continue
        ar  = len(accepted) / len(sub)
        ex  = float((accepted['judge_err'] == 0).mean())
        pm1 = float((accepted['judge_err'] <= 1).mean())
        mae = float(accepted['judge_err'].mean())
        cov = float(accepted['in_iv_raw'].mean())
        print(f"  {tau:>6.2f} {100*ar:>9.1f}% {100*ex:>8.1f}% {100*pm1:>8.1f}% {mae:>7.4f} {100*cov:>8.1f}%")
        row[f'accept_{tau}'] = ar
        row[f'exact_{tau}']  = ex
        row[f'pm1_{tau}']    = pm1
        row[f'mae_{tau}']    = mae
    cat_rows.append(row)

pd.DataFrame(cat_rows).to_csv('results/rebuttal/exp4/per_category_selective.csv', index=False)

# ============================================================
# Per-dataset: same analysis at one canonical tau = 2.5
# (showing the dataset-by-dataset story)
# ============================================================
print("\n" + "="*90)
print("Per-dataset selective evaluation at tau = 2.5 (and baseline)")
print("="*90)
ds_rows = []
for ds in sorted(df['orig_dataset'].unique()):
    sub = df[df.orig_dataset == ds]
    if len(sub) < 100:
        continue
    avg_w = float(sub['width_raw'].mean())
    base_exact = float((sub['judge_err'] == 0).mean())
    sp, sp_p = spearmanr(sub['width_raw'], sub['judge_err'])
    accepted = sub[sub['width_raw'] <= 2.5]
    if len(accepted) > 0:
        ar = len(accepted) / len(sub)
        ex = float((accepted['judge_err'] == 0).mean())
        pm1 = float((accepted['judge_err'] <= 1).mean())
        mae = float(accepted['judge_err'].mean())
    else:
        ar = ex = pm1 = mae = np.nan
    ds_rows.append({
        'dataset': ds, 'category': ds_to_cat.get(ds, 'Other'),
        'n': len(sub), 'avg_width_raw': avg_w,
        'spearman_w_err': sp, 'spearman_p': sp_p,
        'base_exact': base_exact,
        'accept_at_2_5': ar, 'exact_at_2_5': ex,
        'pm1_at_2_5': pm1, 'mae_at_2_5': mae,
    })
ds_df = pd.DataFrame(ds_rows).sort_values(['category', 'avg_width_raw'])
print(ds_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
ds_df.to_csv('results/rebuttal/exp4/per_dataset_selective.csv', index=False)

# ============================================================
# Headline: contrast easy vs hard categories
# ============================================================
print("\n" + "="*90)
print("HEADLINE CONTRAST")
print("="*90)
sub_aes = df[df.category == 'Aesthetics/AI']
sub_vh  = df[df.category == 'Vision-Heavy']
for name, s in [('Aesthetics/AI', sub_aes), ('Vision-Heavy', sub_vh)]:
    base = float((s['judge_err'] == 0).mean())
    print(f"\n{name} (n={len(s)}):")
    for tau in [2.5, 3.0]:
        acc = s[s['width_raw'] <= tau]
        if len(acc) == 0:
            print(f"  tau={tau:.2f}: accept 0% (no items)")
            continue
        ar = len(acc) / len(s)
        ex = float((acc['judge_err'] == 0).mean())
        print(f"  tau={tau:.2f}: accept {100*ar:.1f}%  exact_acc {100*ex:.1f}% (baseline {100*base:.1f}%)")

print("\nFiles saved in results/rebuttal/exp4/{per_category_selective, per_dataset_selective}.csv")
