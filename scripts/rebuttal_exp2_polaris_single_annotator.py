#!/usr/bin/env python3
"""
Rebuttal Experiment 2 (for Reviewer WoGp R4): Polaris single-annotator ablation.

Reviewer's concern: the 4.5x width reduction from MLLM-Judge (3.05) to Polaris
(0.68) confounds task type, score aggregation, # annotators, label continuity,
distribution shape, and benchmark construction. So we cannot causally attribute
the gap to "annotation quality" alone.

This experiment isolates ONE of those variables: annotation aggregation. Polaris
ships per-annotator scores (1-22 raters per (imgid, mt) pair). We:
  - sample ONE rater per item per seed (simulating MLLM-Judge's single-rater setup)
  - map that single score to 1-5 integer the same way the paper does
  - re-fit R2CCP on the same s2 features (judge logprobs)
  - measure width

If the width grows from 0.68 toward MLLM-Judge's 3.05 when we switch from
multi-annotator-mean to single-annotator, we have partial causal evidence
that annotation aggregation is one driver of the gap. Task type, label
continuity, etc. stay fixed (same Polaris images and captions).

10 seeds, alpha=0.10. Uses LLaVA-Critic-7B s2 features (the same as the
paper's Polaris row in Table 7).
"""
import sys, os, time, random, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2CCP.main import R2CCP
from sklearn.model_selection import train_test_split

ALPHA = 0.10
N_SEEDS = 10
SCORE_RANGE = (1, 5)
LP_COLS = ['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']

FEAT_PATH      = 'results/v2_polaris/features_s2.csv'
PROC_PATH      = 'data/polaris/polaris_test_processed.csv'   # multi-annotator mean already there
RAW_PATH       = 'data/polaris/polaris_test.csv'             # raw per-annotator scores

def range_mod(lo, hi):
    return np.clip(lo, *SCORE_RANGE), np.clip(hi, *SCORE_RANGE)

def boundary_adjust_expand(lo, hi):
    return np.clip(np.floor(lo), *SCORE_RANGE), np.clip(np.ceil(hi), *SCORE_RANGE)

def metrics(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))

def merge_intervals(intervals_list):
    lo, hi = [], []
    for sub in intervals_list:
        if not sub:
            lo.append(SCORE_RANGE[0]); hi.append(SCORE_RANGE[1])
        else:
            lo.append(min(l for l, h in sub)); hi.append(max(h for l, h in sub))
    return np.array(lo), np.array(hi)

# ---- Load features (paper's existing Polaris features for LLaVA-Critic) ----
feat = pd.read_csv(FEAT_PATH)
print(f"Polaris features: {len(feat)} rows")
print(f"feat columns: {feat.columns.tolist()}")

# ---- Load processed Polaris (has the aggregated multi-annotator gt_score) ----
proc = pd.read_csv(PROC_PATH)
print(f"Polaris processed: {len(proc)} rows")
# Sanity: features sample_id matches row index in processed
assert len(feat) == len(proc), f"feat ({len(feat)}) and proc ({len(proc)}) have different lengths"

# ---- Load raw per-annotator scores ----
raw = pd.read_csv(RAW_PATH)
print(f"Polaris raw: {len(raw)} rows (per-annotator)")

# Build a map (imgid, mt) -> list of single-annotator continuous scores [0,1]
key_to_scores = {}
for _, r in raw.iterrows():
    k = (r['imgid'], r['mt'])
    key_to_scores.setdefault(k, []).append(float(r['score']))

n_per = np.array([len(key_to_scores[(r['imgid'], r['mt'])]) for _, r in proc.iterrows()])
print(f"Annotators per item: mean={n_per.mean():.2f}, median={int(np.median(n_per))}, max={n_per.max()}")
print(f"Items with >=2 annotators: {(n_per >= 2).sum()} / {len(n_per)}")

# ---- For each row in processed: pre-extract its per-annotator score list ----
score_lists = []
for _, r in proc.iterrows():
    score_lists.append(key_to_scores[(r['imgid'], r['mt'])])

X = feat[LP_COLS].to_numpy().astype(np.float32)

# Multi-annotator (paper baseline): use the already-computed gt_score from features file
y_multi = feat['gt_score'].to_numpy().astype(np.float32)

os.makedirs('results/rebuttal/exp2', exist_ok=True)
out_rows = []

def run_r2ccp_seed(X, y, seed, tag):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X, y, test_size=0.5, random_state=seed)
    tmp = f'results/rebuttal/exp2/tmp_{tag}_s{seed}.pth'
    if os.path.exists(tmp): os.remove(tmp)
    model = R2CCP({'model_path': tmp, 'max_epochs': 100, 'alpha': ALPHA, 'seed': seed})
    model.fit(Xc, yc.flatten())
    intervals = model.get_intervals(Xt)
    lo, hi = merge_intervals(intervals)
    lo, hi = range_mod(lo, hi)
    cov_raw, w_raw = metrics(lo, hi, yt)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, yt)
    del model
    torch.cuda.empty_cache()
    if os.path.exists(tmp): os.remove(tmp)
    return cov_raw, w_raw, cov_adj, w_adj

# ---- Run both arms for each seed ----
print(f"\n{'='*75}")
print("Running Polaris: multi-annotator (paper baseline) vs single-annotator (ablation)")
print(f"{'='*75}")
for seed in range(1, N_SEEDS + 1):
    rng = np.random.RandomState(seed)
    # Build single-annotator y for this seed.
    y_single = np.zeros(len(proc), dtype=np.float32)
    for i, scores in enumerate(score_lists):
        pick = rng.choice(scores)             # one annotator's continuous score in [0,1]
        y_single[i] = int(round(pick * 4) + 1)  # same mapping as paper: 0->1, 0.25->2, 0.5->3, 0.75->4, 1->5

    # Sanity: only items with >=2 annotators see a different y in single-arm
    n_changed = int((y_single != y_multi).sum())
    print(f"\nseed={seed:2d}: single-annotator y differs from multi at {n_changed}/{len(y_single)} items")

    t0 = time.time()
    cov_m, w_m, ca_m, wa_m = run_r2ccp_seed(X, y_multi, seed, 'multi')
    dt = time.time() - t0
    out_rows.append({'seed': seed, 'arm': 'multi_annotator', 'cov_raw': cov_m, 'width_raw': w_m,
                     'cov_adj': ca_m, 'width_adj': wa_m, 'elapsed_sec': dt})
    print(f"  multi:  cov_raw={cov_m:.4f} w_raw={w_m:.4f}  cov_adj={ca_m:.4f} w_adj={wa_m:.4f}  ({dt:.1f}s)", flush=True)
    pd.DataFrame(out_rows).to_csv('results/rebuttal/exp2/per_seed.csv', index=False)

    t0 = time.time()
    cov_s, w_s, ca_s, wa_s = run_r2ccp_seed(X, y_single, seed, 'single')
    dt = time.time() - t0
    out_rows.append({'seed': seed, 'arm': 'single_annotator', 'cov_raw': cov_s, 'width_raw': w_s,
                     'cov_adj': ca_s, 'width_adj': wa_s, 'elapsed_sec': dt})
    print(f"  single: cov_raw={cov_s:.4f} w_raw={w_s:.4f}  cov_adj={ca_s:.4f} w_adj={wa_s:.4f}  ({dt:.1f}s)", flush=True)
    pd.DataFrame(out_rows).to_csv('results/rebuttal/exp2/per_seed.csv', index=False)

# ---- Summary ----
df_out = pd.DataFrame(out_rows)
df_out.to_csv('results/rebuttal/exp2/per_seed.csv', index=False)
summary = df_out.groupby('arm').agg(
    cov_raw_mean=('cov_raw','mean'), cov_raw_std=('cov_raw','std'),
    w_raw_mean=('width_raw','mean'), w_raw_std=('width_raw','std'),
    cov_adj_mean=('cov_adj','mean'), cov_adj_std=('cov_adj','std'),
    w_adj_mean=('width_adj','mean'), w_adj_std=('width_adj','std'),
).reset_index()
summary.to_csv('results/rebuttal/exp2/summary.csv', index=False)

print("\n" + "="*90)
print("FINAL SUMMARY (Polaris)")
print("="*90)
print(summary.to_string(index=False))
print("\nReference numbers from paper (Table 7):")
print("  MLLM-Judge R2CCP raw width:  3.05 (single annotator, integer GT)")
print("  Polaris   R2CCP raw width:  0.68 (multi annotator mean, continuous GT mapped to int)")
print("\nIf single-annotator Polaris width is between 0.68 and 3.05, annotation")
print("aggregation is shown to contribute partially to the gap (but other factors")
print("— task type, etc. — also matter). See results/rebuttal/exp2/summary.csv")
