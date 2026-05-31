#!/usr/bin/env python3
"""
Rebuttal Experiment 1 (for Reviewer WoGp R1): External 40/10/50 split for R2CCP.

The reviewer's concern: paper §3.2 says f(x) is "trained on the calibration set,"
which (literally read) breaks split-conformal validity.

What we actually do in code: split data 50/50 (cal/test), then R2CCP internally
splits its 50% input 80/20: 40% trains the network, 10% computes the conformal
quantile. Quantile-set is disjoint from both the network-training set and the
test set. This is valid split conformal.

This experiment removes any ambiguity by doing the split EXTERNALLY:
  - 40% trains the R2CCP regression network
  - 10% computes the conformal quantile (held out from network training)
  - 50% test
We compare coverage/width to the original protocol with R2CCP's internal split.
If the numbers agree within seed noise, the protocol-validity concern is settled.

10 seeds, alpha=0.10, target coverage = 0.90.
"""
import sys, os, time, random, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2CCP.main import R2CCP
from R2CCP.cp import get_cp_lists, find_intervals_above_value_with_interpolation
from sklearn.model_selection import train_test_split

ALPHA = 0.10
N_SEEDS = 10
SCORE_RANGE = (1, 5)

FEATURE_FILES = {
    'LLaVA-Critic': 'results/v2/features_s2.csv',
    'Phi-4':        'results/v2_phi4/features_s2.csv',
    'Gemini':       'results/v2_gemini/features_s2.csv',
}

LP_COLS = ['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']

def range_mod(lo, hi):
    return np.clip(lo, *SCORE_RANGE), np.clip(hi, *SCORE_RANGE)

def boundary_adjust_expand(lo, hi):
    return np.clip(np.floor(lo), *SCORE_RANGE), np.clip(np.ceil(hi), *SCORE_RANGE)

def metrics(lo, hi, y):
    cov = float(np.mean((y >= lo) & (y <= hi)))
    w   = float(np.mean(hi - lo))
    return cov, w

def merge_intervals(intervals_list):
    """R2CCP can return disjoint subintervals; merge to outer envelope."""
    merged_lo, merged_hi = [], []
    for sub in intervals_list:
        if not sub:
            merged_lo.append(SCORE_RANGE[0])
            merged_hi.append(SCORE_RANGE[1])
        else:
            merged_lo.append(min(l for l, h in sub))
            merged_hi.append(max(h for l, h in sub))
    return np.array(merged_lo), np.array(merged_hi)

# --- Protocol A: original 50/50, R2CCP internal split (40 train / 10 cal / 50 test) ---
def run_protocol_A(X, y, seed, tmp_path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X, y, test_size=0.5, random_state=seed)
    if os.path.exists(tmp_path): os.remove(tmp_path)
    model = R2CCP({'model_path': tmp_path, 'max_epochs': 100, 'alpha': ALPHA, 'seed': seed})
    model.fit(Xc, yc.flatten())
    intervals = model.get_intervals(Xt)
    lo, hi = merge_intervals(intervals)
    lo, hi = range_mod(lo, hi)
    cov_raw, w_raw = metrics(lo, hi, yt)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, yt)
    del model
    torch.cuda.empty_cache()
    if os.path.exists(tmp_path): os.remove(tmp_path)
    return cov_raw, w_raw, cov_adj, w_adj

# --- Protocol B: external 40/10/50 split, manual conformal quantile on the 10% ---
def run_protocol_B(X, y, seed, tmp_path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # First split off test (50%); then split the remaining 50% into 40%+10% (80/20 of remaining).
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    # 80/20 of 50% gives 40/10 of total.
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.20, random_state=seed)
    assert abs(len(X_train) / len(X) - 0.40) < 0.02, f"train fraction {len(X_train)/len(X):.3f} != 0.40"
    assert abs(len(X_cal)   / len(X) - 0.10) < 0.02, f"cal fraction {len(X_cal)/len(X):.3f} != 0.10"
    assert abs(len(X_test)  / len(X) - 0.50) < 0.02, f"test fraction {len(X_test)/len(X):.3f} != 0.50"

    if os.path.exists(tmp_path): os.remove(tmp_path)
    # Pass only the 40% training set. R2CCP will internally split this 40% 80/20
    # for early-stopping val loss, but neither chunk is our external cal set.
    model = R2CCP({'model_path': tmp_path, 'max_epochs': 100, 'alpha': ALPHA, 'seed': seed})
    model.fit(X_train, y_train.flatten())

    # Now manually compute the conformal quantile on the EXTERNAL 10% cal set,
    # then form intervals on the test set using that quantile.
    # Replicate R2CCP.get_intervals but feed our held-out X_cal/y_cal instead of
    # its internal inner-cal portion.
    args = model._args
    range_vals = model.range_vals
    X_cal_sc = model.scaler_X.transform(X_cal)
    y_cal_sc = model.scaler_y.transform(y_cal.reshape(-1, 1))
    X_test_sc = model.scaler_X.transform(X_test)
    # Convert to torch tensors as R2CCP expects.
    X_cal_t = torch.tensor(X_cal_sc, dtype=torch.float32)
    y_cal_t = torch.tensor(y_cal_sc, dtype=torch.float32)
    intervals = get_cp_lists(X_test_sc, args, range_vals, X_cal_t, y_cal_t, model.model)
    intervals_inv = model.invert_intervals(intervals)
    lo, hi = merge_intervals(intervals_inv)
    lo, hi = range_mod(lo, hi)
    cov_raw, w_raw = metrics(lo, hi, y_test)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, y_test)
    del model
    torch.cuda.empty_cache()
    if os.path.exists(tmp_path): os.remove(tmp_path)
    return cov_raw, w_raw, cov_adj, w_adj

# --- Main ---
os.makedirs('results/rebuttal/exp1', exist_ok=True)
out_rows = []

for judge_name, feat_path in FEATURE_FILES.items():
    if not os.path.exists(feat_path):
        print(f"[SKIP] {judge_name}: {feat_path} not found")
        continue
    df = pd.read_csv(feat_path)
    df = df[df.gt_score >= 1].reset_index(drop=True)
    X = df[LP_COLS].to_numpy().astype(np.float32)
    y = df['gt_score'].to_numpy().astype(np.float32)
    print(f"\n{'='*70}\n{judge_name}: N={len(df)} samples\n{'='*70}")

    for seed in range(1, N_SEEDS + 1):
        for proto_name, proto_fn in [('A_internal_split', run_protocol_A),
                                      ('B_external_split', run_protocol_B)]:
            t0 = time.time()
            tmp = f'results/rebuttal/exp1/tmp_{judge_name}_{proto_name}_s{seed}.pth'
            try:
                cov_raw, w_raw, cov_adj, w_adj = proto_fn(X, y, seed, tmp)
                dt = time.time() - t0
                out_rows.append({
                    'judge': judge_name, 'protocol': proto_name, 'seed': seed,
                    'cov_raw': cov_raw, 'width_raw': w_raw,
                    'cov_adj': cov_adj, 'width_adj': w_adj,
                    'elapsed_sec': dt,
                })
                print(f"  seed={seed:2d} {proto_name}: cov_raw={cov_raw:.4f} w_raw={w_raw:.4f}  cov_adj={cov_adj:.4f} w_adj={w_adj:.4f}  ({dt:.1f}s)", flush=True)
                # Save after every seed/protocol so we always have incremental progress on disk.
                pd.DataFrame(out_rows).to_csv('results/rebuttal/exp1/per_seed.csv', index=False)
            except Exception as e:
                print(f"  seed={seed:2d} {proto_name}: FAILED — {e}", flush=True)
            sys.stdout.flush()

# Final summary
df_out = pd.DataFrame(out_rows)
df_out.to_csv('results/rebuttal/exp1/per_seed.csv', index=False)
summary = df_out.groupby(['judge','protocol']).agg(
    cov_raw_mean=('cov_raw','mean'), cov_raw_std=('cov_raw','std'),
    w_raw_mean=('width_raw','mean'), w_raw_std=('width_raw','std'),
    cov_adj_mean=('cov_adj','mean'), cov_adj_std=('cov_adj','std'),
    w_adj_mean=('width_adj','mean'), w_adj_std=('width_adj','std'),
).reset_index()
summary.to_csv('results/rebuttal/exp1/summary.csv', index=False)
print("\n" + "="*90)
print("FINAL SUMMARY")
print("="*90)
print(summary.to_string(index=False))
print("\nSaved: results/rebuttal/exp1/{per_seed.csv, summary.csv}")
