#!/usr/bin/env python3
"""
Rebuttal Experiments 4 + 5 (for Reviewer YU66 R2): per-instance CP utility.

Reviewer YU66 R2 (the load-bearing rejection reasoning):
  "While coverage is achieved, the intervals provide little actionable
   information at the instance level. This limitation is not sufficiently
   examined."

This script saves per-instance (lo, hi) outputs from R2CCP at each test
sample, across 10 seeds, for all 3 judges. The same outputs feed two
post-hoc analyses (no extra training):

  Exp 4: Per-instance width vs. judge error correlation.
    For each sample, compute |parsed_score - gt_score| (judge point error)
    and the CP interval width. Compute Spearman/Pearson correlation; bin
    into width deciles; show mean error per bin. A monotone width->error
    relationship is direct instance-level utility.

  Exp 5: Selective evaluation via CP-width threshold.
    Vary tau in [0.5, 4.0]. For samples with width <= tau:
      - accept_rate(tau) = fraction of samples accepted
      - accept_accuracy(tau) = exact judge accuracy among accepted
      - accept_pm1(tau)     = +/-1 judge accuracy among accepted
      - accept_mae(tau)     = mean judge MAE among accepted
      - accept_cov(tau)     = CP coverage among accepted (should stay >=90%)
    This operationalizes the intervals as a decision rule.

Reuses the paper's exact protocol: 50/50 outer split, R2CCP internal split,
alpha=0.10, 10 seeds.
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

JUDGES = {
    'LLaVA-Critic': 'results/v2/features_s2.csv',
    'Phi-4':        'results/v2_phi4/features_s2.csv',
    'Gemini':       'results/v2_gemini/features_s2.csv',
}

def range_mod(lo, hi):
    return np.clip(lo, *SCORE_RANGE), np.clip(hi, *SCORE_RANGE)

def boundary_adjust_expand(lo, hi):
    return np.clip(np.floor(lo), *SCORE_RANGE), np.clip(np.ceil(hi), *SCORE_RANGE)

def merge_intervals(intervals_list):
    lo, hi = [], []
    for sub in intervals_list:
        if not sub:
            lo.append(SCORE_RANGE[0]); hi.append(SCORE_RANGE[1])
        else:
            lo.append(min(l for l, h in sub)); hi.append(max(h for l, h in sub))
    return np.array(lo), np.array(hi)

os.makedirs('results/rebuttal/exp4', exist_ok=True)
all_rows = []

for judge_name, feat_path in JUDGES.items():
    if not os.path.exists(feat_path):
        print(f"[SKIP] {judge_name}: {feat_path} not found", flush=True)
        continue
    df = pd.read_csv(feat_path)
    df = df[df.gt_score >= 1].reset_index(drop=True)
    X = df[LP_COLS].to_numpy().astype(np.float32)
    y = df['gt_score'].to_numpy().astype(np.float32)
    parsed = df['parsed_score'].to_numpy().astype(np.float32)
    sample_ids = df['sample_id'].to_numpy() if 'sample_id' in df.columns else np.arange(len(df))

    print(f"\n{'='*70}\n{judge_name}: N={len(df)}\n{'='*70}", flush=True)

    for seed in range(1, N_SEEDS + 1):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        # Replicate the paper's outer 50/50 split (same as run_all_conformal.py)
        idx = np.arange(len(df))
        idx_cal, idx_test = train_test_split(idx, test_size=0.5, random_state=seed)

        Xc, Xt = X[idx_cal], X[idx_test]
        yc, yt = y[idx_cal], y[idx_test]
        parsed_test = parsed[idx_test]
        sid_test = sample_ids[idx_test]

        tmp = f'results/rebuttal/exp4/tmp_{judge_name}_s{seed}.pth'
        if os.path.exists(tmp): os.remove(tmp)
        t0 = time.time()
        try:
            model = R2CCP({'model_path': tmp, 'max_epochs': 100, 'alpha': ALPHA, 'seed': seed})
            model.fit(Xc, yc.flatten())
            intervals = model.get_intervals(Xt)
            lo, hi = merge_intervals(intervals)
            lo_raw, hi_raw = range_mod(lo, hi)
            lo_adj, hi_adj = boundary_adjust_expand(lo_raw, hi_raw)

            for i in range(len(idx_test)):
                all_rows.append({
                    'judge': judge_name, 'seed': seed,
                    'sample_id': int(sid_test[i]),
                    'gt_score': float(yt[i]),
                    'parsed_score': float(parsed_test[i]),
                    'lo_raw': float(lo_raw[i]), 'hi_raw': float(hi_raw[i]),
                    'lo_adj': float(lo_adj[i]), 'hi_adj': float(hi_adj[i]),
                    'width_raw': float(hi_raw[i] - lo_raw[i]),
                    'width_adj': float(hi_adj[i] - lo_adj[i]),
                    'in_iv_raw': int((yt[i] >= lo_raw[i]) and (yt[i] <= hi_raw[i])),
                    'in_iv_adj': int((yt[i] >= lo_adj[i]) and (yt[i] <= hi_adj[i])),
                    'judge_err': float(abs(parsed_test[i] - yt[i])) if parsed_test[i] > 0 else np.nan,
                })

            dt = time.time() - t0
            n_now = len([r for r in all_rows if r['judge']==judge_name])
            print(f"  seed={seed:2d}: {len(idx_test)} test samples saved (cum {n_now} for {judge_name})  ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"  seed={seed:2d} FAILED: {e}", flush=True)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
            del model
            torch.cuda.empty_cache()
        # Save incrementally
        pd.DataFrame(all_rows).to_csv('results/rebuttal/exp4/per_instance.csv', index=False)

# Final write
df_out = pd.DataFrame(all_rows)
df_out.to_csv('results/rebuttal/exp4/per_instance.csv', index=False)
print(f"\nSaved: results/rebuttal/exp4/per_instance.csv  (n={len(df_out)} per-instance records)", flush=True)
print("Next: run scripts/rebuttal_exp4_5_analyze.py to generate Exp 4 and Exp 5 tables.", flush=True)
