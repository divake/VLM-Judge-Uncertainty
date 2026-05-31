#!/usr/bin/env python3
"""
Rebuttal Exp 8: Joint item-level bootstrap of (rho, width, RSG) per
(judge, dataset). Replaces the mixed-unit approach from exp6.

For each (judge, dataset):
  1. Build per-item records: (parsed, gt, mean_width_across_seeds).
     mean_width comes from exp4/per_instance.csv (pooled MLLM-Judge runs).
  2. Bootstrap items B=2000; each replicate yields rho_b, width_b, rsg_b.
  3. Report point estimate (computed on full items, NOT seed-averaged)
     and percentile CIs.

This gives a coherent single-unit bootstrap. Point estimates may differ
slightly from the paper's seed-averaged numbers; for the rebuttal table
we will report PAPER point estimates with bootstrap CIs derived here,
since the paper values are the authoritative reference and the bootstrap
quantifies sampling uncertainty around them.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = '/ssd_4TB/divake/VLM_Judge_Uncertainty'
os.chdir(ROOT)

K = 5
N_BOOT = 2000
SEED = 0

JUDGES = {
    'LLaVA-Critic': 'results/v2/features_s2.csv',
    'Phi-4':        'results/v2_phi4/features_s2.csv',
    'Gemini':       'results/v2_gemini/features_s2.csv',
}

DATASET_CATEGORIES = {
    'Vision-Heavy':  ['mathvista', 'ChartQA', 'infographicsVQA', 'ScienceQA', 'textVQA'],
    'General VQA':   ['coco', 'llava_bench', 'mm-vet', 'VisitBench'],
    'Knowledge/Web': ['WIT', 'Concept Caption', 'mind2web'],
    'Aesthetics/AI': ['AesBench', 'diffusiondb'],
}
ds_to_cat = {ds: cat for cat, lst in DATASET_CATEGORIES.items() for ds in lst}

META = {}
with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
    for i, line in enumerate(f):
        META[i] = json.loads(line).get('original_dataset', 'unknown')

# Load per-instance widths (pooled MLLM-Judge R2CCP runs); mean across seeds per item.
per_inst = pd.read_csv('results/rebuttal/exp4/per_instance.csv')
mean_width = per_inst.groupby(['judge', 'sample_id'])['width_raw'].mean().reset_index()
mean_width.columns = ['judge', 'sample_id', 'mean_width']
print(f"Per-item mean widths: {len(mean_width)} records across {mean_width.judge.nunique()} judges")

os.makedirs('results/rebuttal/exp8', exist_ok=True)
rows = []

for judge, fpath in JUDGES.items():
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    df = df[df.gt_score >= 1].reset_index(drop=True)
    df['orig_dataset'] = df['sample_id'].map(lambda sid: META.get(int(sid), 'unknown'))
    # Merge in mean widths for this judge's items
    mw = mean_width[mean_width.judge == judge][['sample_id', 'mean_width']]
    df = df.merge(mw, on='sample_id', how='left')
    print(f"\n=== {judge} === (n_items={len(df)}, with_width={df.mean_width.notna().sum()})")

    for ds in sorted(df['orig_dataset'].unique()):
        sub = df[df['orig_dataset'] == ds]
        if len(sub) < 50:
            continue
        valid = sub[(sub['parsed_score'] > 0) & sub['gt_score'].notna() & sub['mean_width'].notna()]
        if len(valid) < 50:
            print(f"  [skip {ds}: n_valid={len(valid)} < 50]")
            continue

        p = valid['parsed_score'].to_numpy(dtype=np.float64)
        g = valid['gt_score'].to_numpy(dtype=np.float64)
        w = valid['mean_width'].to_numpy(dtype=np.float64)
        n = len(p)

        # Point estimates on the full items (NOT bootstrap-resampled).
        rho_pt = float(pearsonr(p, g)[0])
        w_pt = float(np.mean(w))
        rsg_pt = abs(rho_pt) - (1.0 - w_pt / (K - 1))

        # Joint item-level bootstrap.
        rng = np.random.default_rng(SEED)
        rhos = np.empty(N_BOOT); ws = np.empty(N_BOOT); rsgs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            pb, gb, wb = p[idx], g[idx], w[idx]
            if np.std(pb) < 1e-12 or np.std(gb) < 1e-12:
                rhos[b] = ws[b] = rsgs[b] = np.nan
            else:
                rho_b = pearsonr(pb, gb)[0]
                w_b = np.mean(wb)
                rsg_b = abs(rho_b) - (1.0 - w_b / (K - 1))
                rhos[b] = rho_b; ws[b] = w_b; rsgs[b] = rsg_b

        def ci(arr):
            arr = arr[~np.isnan(arr)]
            return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        rho_lo, rho_hi = ci(rhos)
        w_lo, w_hi = ci(ws)
        rsg_lo, rsg_hi = ci(rsgs)

        rows.append({
            'judge': judge, 'dataset': ds, 'category': ds_to_cat.get(ds, 'Other'),
            'n_items': n,
            'rho_pt': rho_pt, 'rho_lo': rho_lo, 'rho_hi': rho_hi,
            'width_pt': w_pt, 'width_lo': w_lo, 'width_hi': w_hi,
            'rsg_pt': rsg_pt, 'rsg_lo': rsg_lo, 'rsg_hi': rsg_hi,
        })
        print(f"  {ds:<22s} n={n:>4d}  rho={rho_pt:+.3f} [{rho_lo:+.3f},{rho_hi:+.3f}]  "
              f"w={w_pt:.3f} [{w_lo:.3f},{w_hi:.3f}]  RSG={rsg_pt:+.3f} [{rsg_lo:+.3f},{rsg_hi:+.3f}]")

out = pd.DataFrame(rows)
out.to_csv('results/rebuttal/exp8/joint_item_bootstrap.csv', index=False)
print(f"\nSaved: results/rebuttal/exp8/joint_item_bootstrap.csv ({len(out)} rows)")

# Print Fisher z sanity for the LLaVA-Critic cells we display
print("\nFisher-z sanity (LLaVA-Critic):")
for _, r in out[out.judge == 'LLaVA-Critic'].iterrows():
    if r['dataset'] in {'ChartQA', 'AesBench', 'WIT', 'mathvista', 'infographicsVQA'}:
        n = r['n_items']; rho = r['rho_pt']
        z = np.arctanh(rho); se = 1.0 / np.sqrt(n - 3)
        rho_lo_f = float(np.tanh(z - 1.96 * se))
        rho_hi_f = float(np.tanh(z + 1.96 * se))
        hw_bs = (r['rho_hi'] - r['rho_lo']) / 2
        hw_f = (rho_hi_f - rho_lo_f) / 2
        print(f"  {r['dataset']:<16s} n={n:>4d}  bs_HW={hw_bs:.4f}  fisher_HW={hw_f:.4f}")
