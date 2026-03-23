#!/usr/bin/env python3
"""Run R2CCP per original_dataset category to see which task types
produce wider/narrower intervals. 10 seeds each."""

import sys, os
import numpy as np
import pandas as pd
import json
import random
import torch
import time
import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2CCP.main import R2CCP
from sklearn.model_selection import train_test_split

ALPHA = 0.10
N_SEEDS = 10

# Load features
df = pd.read_csv('results/v2/features_s2.csv')
df = df[df.gt_score >= 1].reset_index(drop=True)

# Load dataset metadata
meta = {}
with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        meta[i] = d.get('original_dataset', 'unknown')

df['orig_dataset'] = df['sample_id'].map(lambda sid: meta.get(int(sid), 'unknown'))

# Categorize datasets by type
DATASET_CATEGORIES = {
    'Vision-Heavy': ['mathvista', 'ChartQA', 'infographicsVQA', 'ScienceQA', 'textVQA'],
    'General VQA': ['coco', 'llava_bench', 'mm-vet', 'VisitBench'],
    'Knowledge/Web': ['WIT', 'Concept Caption', 'mind2web'],
    'Aesthetics/AI': ['AesBench', 'diffusiondb'],
}

def range_mod(lo, hi):
    return np.clip(lo, 1, 5), np.clip(hi, 1, 5)

def boundary_adjust(lo, hi):
    return np.clip(np.floor(lo), 1, 5), np.clip(np.ceil(hi), 1, 5)

def run_r2ccp_subset(X, y, n_seeds=N_SEEDS):
    """Run R2CCP on a subset, return (cov_raw, w_raw, cov_adj, w_adj) means."""
    if len(y) < 50:
        return None  # too small

    cov_raw, w_raw, cov_adj, w_adj = [], [], [], []
    for seed in range(1, n_seeds + 1):
        try:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            Xc, Xt, yc, yt = train_test_split(X, y, test_size=0.5, random_state=seed)

            mp = f'results/v2/tmp_r2ccp_ds.pth'
            if os.path.exists(mp): os.remove(mp)
            model = R2CCP({'model_path': mp, 'max_epochs': 100, 'alpha': ALPHA})
            model.fit(Xc, yc.flatten())
            intervals = model.get_intervals(Xt)

            merged = []
            for si in intervals:
                if not si: merged.append((1, 5))
                else: merged.append((min(l for l, h in si), max(h for l, h in si)))

            lo = np.array([iv[0] for iv in merged])
            hi = np.array([iv[1] for iv in merged])
            lo, hi = range_mod(lo, hi)

            in_iv = (yt >= lo) & (yt <= hi)
            cov_raw.append(in_iv.mean())
            w_raw.append((hi - lo).mean())

            la, ha = boundary_adjust(lo, hi)
            in_iv2 = (yt >= la) & (yt <= ha)
            cov_adj.append(in_iv2.mean())
            w_adj.append((ha - la).mean())

            del model; torch.cuda.empty_cache()
            if os.path.exists(mp): os.remove(mp)
        except Exception as e:
            print(f"    Seed {seed} failed: {e}")

    if not cov_raw:
        return None
    return (np.mean(cov_raw), np.std(cov_raw), np.mean(w_raw), np.std(w_raw),
            np.mean(cov_adj), np.std(cov_adj), np.mean(w_adj), np.std(w_adj))

# --- Run per dataset ---
print(f"R2CCP per dataset (α={ALPHA}, {N_SEEDS} seeds)")
print(f"Total: {len(df)} samples across {df['orig_dataset'].nunique()} datasets\n")

print(f"{'Dataset':<22s} {'N':>5s} │ {'Cov(raw)':>13s} {'Width(raw)':>13s} │ {'Cov(adj)':>13s} {'Width(adj)':>13s} │ {'Pearson':>8s}")
print("─" * 110)

results_rows = []
for ds in sorted(df['orig_dataset'].unique()):
    mask = df['orig_dataset'] == ds
    sub = df[mask]
    X = sub[['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']].to_numpy().astype(np.float32)
    y = sub['gt_score'].to_numpy().astype(np.float32)

    # Point prediction
    from scipy.stats import pearsonr
    valid = sub[sub.parsed_score > 0]
    if len(valid) > 10:
        p, _ = pearsonr(valid.parsed_score, valid.gt_score)
    else:
        p = 0.0

    t0 = time.time()
    result = run_r2ccp_subset(X, y)
    elapsed = time.time() - t0

    if result:
        cr_m, cr_s, wr_m, wr_s, ca_m, ca_s, wa_m, wa_s = result
        print(f"{ds:<22s} {len(sub):5d} │ {cr_m:.4f}±{cr_s:.4f} {wr_m:.4f}±{wr_s:.4f} │ {ca_m:.4f}±{ca_s:.4f} {wa_m:.4f}±{wa_s:.4f} │ {p:8.4f}  ({elapsed:.0f}s)")
        results_rows.append({'dataset': ds, 'n': len(sub), 'pearson': p,
                             'cov_raw': cr_m, 'width_raw': wr_m, 'cov_adj': ca_m, 'width_adj': wa_m})
    else:
        print(f"{ds:<22s} {len(sub):5d} │ SKIPPED (too small)")
    sys.stdout.flush()

# --- Category summary ---
print("\n\n" + "=" * 80)
print("CATEGORY SUMMARY")
print("=" * 80)
results_df = pd.DataFrame(results_rows)
for cat, datasets in DATASET_CATEGORIES.items():
    cat_df = results_df[results_df['dataset'].isin(datasets)]
    if len(cat_df) == 0: continue
    avg_w_raw = cat_df['width_raw'].mean()
    avg_w_adj = cat_df['width_adj'].mean()
    avg_p = cat_df['pearson'].mean()
    total_n = cat_df['n'].sum()
    print(f"\n{cat} (N={total_n}):")
    print(f"  Avg Width(raw)={avg_w_raw:.4f}, Avg Width(adj)={avg_w_adj:.4f}, Avg Pearson={avg_p:.4f}")
    for _, row in cat_df.iterrows():
        print(f"    {row['dataset']:20s}: Width(raw)={row['width_raw']:.4f}, Pearson={row['pearson']:.4f}")

print("\nDone!")
