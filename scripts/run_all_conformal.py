#!/usr/bin/env python3
"""Run all 9 conformal prediction methods on new CoT features.

Uses exact patterns from conformal_predictors/*.py (Sheng et al.'s code).
Reports coverage and width before and after boundary adjustment.
Prints results after each method for live monitoring.
"""

import sys, os
import numpy as np
import pandas as pd
import random
import torch
import time
import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('medium')

# Add dependency paths
LLM_REPO = '/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge'
sys.path.insert(0, os.path.join(LLM_REPO, 'chr'))
sys.path.insert(0, os.path.join(LLM_REPO, 'LvD'))
sys.path.insert(0, os.path.join(LLM_REPO, 'boosted-conformal'))
sys.path.insert(0, os.path.join(LLM_REPO, 'boosted-conformal', 'third_party'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from scipy.special import softmax

ALPHA = 0.10
N_SEEDS = 10
SCORE_RANGE = (1, 5)

# ============================================================
# Utility
# ============================================================

def range_mod(lows, highs):
    return np.clip(lows, *SCORE_RANGE), np.clip(highs, *SCORE_RANGE)

def boundary_adjust_expand(lows, highs):
    """Expand: floor lower, ceil upper."""
    return np.clip(np.floor(lows), *SCORE_RANGE), np.clip(np.ceil(highs), *SCORE_RANGE)

def metrics(lows, highs, y_test):
    return float(np.mean((y_test >= lows) & (y_test <= highs))), float(np.mean(highs - lows))

def run_seeds(method_fn, label):
    """Generic seed runner. method_fn(seed) -> (lows, highs, y_test)"""
    cr, wr, ca, wa = [], [], [], []
    t0 = time.time()
    for seed in range(1, N_SEEDS + 1):
        try:
            lows, highs, y_test = method_fn(seed)
            c, w = metrics(lows, highs, y_test)
            cr.append(c); wr.append(w)
            la, ha = boundary_adjust_expand(lows, highs)
            c2, w2 = metrics(la, ha, y_test)
            ca.append(c2); wa.append(w2)
        except Exception as e:
            print(f"  [WARN] {label} seed {seed} failed: {e}")
    elapsed = time.time() - t0
    if cr:
        print(f"\n{'='*65}")
        print(f"  {label}  ({elapsed:.1f}s, {elapsed/len(cr):.1f}s/seed)")
        print(f"  Raw:      Coverage={np.mean(cr):.4f}±{np.std(cr):.4f}  Width={np.mean(wr):.4f}±{np.std(wr):.4f}")
        print(f"  Adjusted: Coverage={np.mean(ca):.4f}±{np.std(ca):.4f}  Width={np.mean(wa):.4f}±{np.std(wa):.4f}")
        print(f"{'='*65}")
    else:
        print(f"\n  {label}: ALL SEEDS FAILED")
    sys.stdout.flush()
    return cr, wr, ca, wa

# ============================================================
# Load data
# ============================================================

df = pd.read_csv('results/v2/features_s2.csv')
df = df[df.gt_score >= 1].reset_index(drop=True)
X_logits = df[['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']].to_numpy().astype(np.float32)
y_all = df['gt_score'].to_numpy().astype(np.float32)
X_probs = softmax(X_logits, axis=1).astype(np.float32)

print(f"Data: {len(df)} samples, α={ALPHA}, {N_SEEDS} seeds")
all_results = {}

# ============================================================
# 1. CQR (mapie, symmetry=False)
# ============================================================
print("\n>>> 1/9 CQR...")
from sklearn.ensemble import GradientBoostingRegressor
from mapie.quantile_regression import MapieQuantileRegressor

def cqr_seed(seed):
    random.seed(seed); np.random.seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    gb = GradientBoostingRegressor(loss="quantile", random_state=seed)
    mqr = MapieQuantileRegressor(estimator=gb, alpha=ALPHA)
    mqr.fit(Xc, yc, random_state=seed)
    _, pis = mqr.predict(Xt, symmetry=False)
    lo, hi = range_mod(pis[:, 0].ravel(), pis[:, 1].ravel())
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(cqr_seed, "CQR")
all_results['CQR'] = (cr, wr, ca, wa)

# ============================================================
# 2. Asymmetric CQR (mapie, symmetry=True)
# ============================================================
print("\n>>> 2/9 Asymmetric CQR...")

def asymcqr_seed(seed):
    random.seed(seed); np.random.seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    gb = GradientBoostingRegressor(loss="quantile", random_state=seed)
    mqr = MapieQuantileRegressor(estimator=gb, alpha=ALPHA)
    mqr.fit(Xc, yc, random_state=seed)
    _, pis = mqr.predict(Xt, symmetry=True)
    lo, hi = range_mod(pis[:, 0].ravel(), pis[:, 1].ravel())
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(asymcqr_seed, "Asymmetric CQR")
all_results['AsymCQR'] = (cr, wr, ca, wa)

# ============================================================
# 3. CHR (exact pattern from CHR_random.py)
# ============================================================
print("\n>>> 3/9 CHR...")
from chr.black_boxes import QNet
from chr.histogram import Histogram
from chr.grey_boxes import HistogramAccumulator

def chr_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    grid_quantiles = np.arange(0.01, 1.0, 0.01)
    bbox = QNet(grid_quantiles, Xc.shape[1], no_crossing=True, batch_size=32,
                dropout=0.1, num_epochs=500, learning_rate=0.0005, num_hidden=256, calibrate=False)
    bbox.fit(Xc, yc)
    grid_histogram = np.arange(0, 6, 0.1)
    hist = Histogram(grid_quantiles, grid_histogram)
    Q_test = bbox.predict(Xt)
    histogram_test = hist.compute_histogram(Q_test, 0, 6, 0.001)
    accumulator = HistogramAccumulator(histogram_test, grid_histogram, alpha=ALPHA, delta_alpha=0.01)
    epsilon = np.random.uniform(low=0.0, high=1.0, size=Xt.shape[0])
    S, bands = accumulator.predict_intervals(ALPHA, epsilon=epsilon)
    S_int = [np.arange(S[i][0], S[i][1]+1) for i in range(len(S))]
    intervals = np.array([[grid_histogram[S_int[i]-1][0], grid_histogram[S_int[i]][-1]] for i in range(len(S_int))])
    lo, hi = range_mod(intervals[:, 0], intervals[:, 1])
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(chr_seed, "CHR")
all_results['CHR'] = (cr, wr, ca, wa)

# ============================================================
# 4. LVD (exact pattern from LVD_random.py)
# ============================================================
print("\n>>> 4/9 LVD...")
from data.preprocess_small_datasets import pretrain_general
from models.regmodel import KernelMLKR
from models.conformal import LocalConditional

def lvd_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    yc_r = yc.reshape(-1, 1); yt_r = yt.reshape(-1, 1)
    DNN_model, readout_layer = pretrain_general(Xc, yc_r, seed=0, quiet=True, model_setting=0)
    embed_cal = DNN_model.embed(Xc)
    embed_test = DNN_model.embed(Xt)
    kernel_model = KernelMLKR(d=10, seed=0, n_iters=500, norm=True, lr=1e-3)
    kernel_model.fit(embed_cal, yc.flatten())
    lvd = LocalConditional(K_obj=kernel_model)
    lvd.fit(embed_cal, yc.flatten(), m=readout_layer)
    results = lvd.eval(embed_test, yt.flatten(), lvd.PI, alpha=ALPHA, quiet=True)
    lo = results['lo'].to_numpy()
    hi = results['hi'].to_numpy()
    lo, hi = range_mod(lo, hi)
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(lvd_seed, "LVD")
all_results['LVD'] = (cr, wr, ca, wa)

# ============================================================
# 5. Boosted CQR (exact pattern from BoostedCP_random.py)
# ============================================================
print("\n>>> 5/9 Boosted CQR...")
from boostedCP.len_cqr_boost import len_cqr_boost

def boosted_cqr_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    results = len_cqr_boost(Xc, yc, Xc, yc, Xt, yt, ALPHA, seed,
                            n_rounds_cv=500, learning_rate=0.02, store=True, verbose=False)
    lo, hi = range_mod(results["boosted_lower"].flatten(), results["boosted_upper"].flatten())
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(boosted_cqr_seed, "Boosted CQR")
all_results['BoostedCQR'] = (cr, wr, ca, wa)

# ============================================================
# 6. Boosted LCP (exact pattern from BoostedCP_random.py)
# ============================================================
print("\n>>> 6/9 Boosted LCP...")
from boostedCP.len_local_boost import len_local_boost

def boosted_lcp_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    results = len_local_boost(Xc, yc, Xc, yc, Xt, yt, ALPHA, seed,
                              n_rounds_cv=500, learning_rate=0.02, store=True, verbose=False)
    lo, hi = range_mod(results["boosted_lower"].flatten(), results["boosted_upper"].flatten())
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(boosted_lcp_seed, "Boosted LCP")
all_results['BoostedLCP'] = (cr, wr, ca, wa)

# ============================================================
# 7. R2CCP
# ============================================================
print("\n>>> 7/9 R2CCP...")
from R2CCP.main import R2CCP as R2CCPModel

def r2ccp_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc, Xt, yc, yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    mp = 'results/v2/tmp_r2ccp.pth'
    if os.path.exists(mp): os.remove(mp)
    model = R2CCPModel({'model_path': mp, 'max_epochs': 100, 'alpha': ALPHA})
    model.fit(Xc, yc.flatten())
    intervals = model.get_intervals(Xt)
    merged = []
    for si in intervals:
        if not si: merged.append((1, 5))
        else: merged.append((min(l for l, h in si), max(h for l, h in si)))
    lo = np.array([iv[0] for iv in merged])
    hi = np.array([iv[1] for iv in merged])
    lo, hi = range_mod(lo, hi)
    del model; torch.cuda.empty_cache()
    if os.path.exists(mp): os.remove(mp)
    return lo, hi, yt

cr, wr, ca, wa = run_seeds(r2ccp_seed, "R2CCP")
all_results['R2CCP'] = (cr, wr, ca, wa)

# ============================================================
# 8. OrdinalAPS (5-class, extended grid)
# ============================================================
print("\n>>> 8/9 OrdinalAPS...")

def ordinal_aps_predict(val_scores, qhat):
    n_samples, n_classes = val_scores.shape
    P = val_scores == val_scores.max(axis=1)[:, None]
    idx_inc = (val_scores * P.astype(float)).sum(axis=1) <= qhat
    for _ in range(n_classes):
        if idx_inc.sum() == 0: break
        P_inc = P[idx_inc]; scores_inc = val_scores[idx_inc]
        cumsum = P_inc.cumsum(axis=1)
        lo = (P_inc > 0).argmax(axis=1); hi = cumsum.argmax(axis=1)
        left_ok = (lo - 1) >= 0; right_ok = (hi + 1) < n_classes
        expand_left = np.zeros(scores_inc.shape[0], dtype=bool)
        expand_left[~right_ok & left_ok] = True
        both = left_ok & right_ok
        if both.any():
            ls = scores_inc[np.arange(scores_inc.shape[0])[both], lo[both]-1]
            rs = scores_inc[np.arange(scores_inc.shape[0])[both], hi[both]+1]
            expand_left[both] = ls > rs
        vl = expand_left & left_ok; P_inc[vl, lo[vl]-1] = True
        vr = (~expand_left) & right_ok; P_inc[vr, hi[vr]+1] = True
        P[idx_inc] = P_inc
        idx_inc = (val_scores * P.astype(float)).sum(axis=1) <= qhat
    else:
        P[idx_inc] = True
    return P

def ordinal_aps_seed(seed):
    np.random.seed(seed)
    y_cls = (y_all - 1).astype(int)
    Xc, Xt, yc, yt = train_test_split(X_probs, y_cls, test_size=0.5, random_state=seed)
    n_cal = Xc.shape[0]
    target = np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal
    grid = np.linspace(1e-3, 1 - 1e-8, 50000)
    qhat = grid[0]
    for q in grid[::-1]:
        sets = ordinal_aps_predict(np.copy(Xc), q)
        if sets[np.arange(n_cal), yc].mean() <= target:
            qhat = min(q + 1/(len(grid)-1), 1.0-1e-10)
            break
    test_sets = ordinal_aps_predict(np.copy(Xt), qhat)
    lo_l, hi_l = [], []
    for ps in test_sets:
        idx = np.where(ps)[0]
        lo_l.append(idx.min() + 1.0); hi_l.append(idx.max() + 1.0)
    return np.array(lo_l), np.array(hi_l), yt.astype(float) + 1.0

cr, wr, ca, wa = run_seeds(ordinal_aps_seed, "OrdinalAPS (5-class)")
all_results['OrdinalAPS'] = (cr, wr, ca, wa)

# ============================================================
# 9. OrdinalRC (5-class)
# ============================================================
print("\n>>> 9/9 OrdinalRC...")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'conformal_predictors'))

# Import OrdinalRC inline to avoid top-level script execution
import importlib.util
spec = importlib.util.spec_from_file_location("ordrc",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'conformal_predictors', 'OrdinalRC_random.py'))
ordrc_mod = importlib.util.module_from_spec(spec)

# We need to extract the class manually since the file has top-level code
# Let's just re-implement the key class here
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
          'conformal_predictors', 'OrdinalRC_random.py')).read().split('\nimport os')[0])

def ordinal_rc_seed(seed):
    np.random.seed(seed); random.seed(seed)
    y_cls = (y_all - 1).astype(int)
    Xc, Xt, yc, yt = train_test_split(X_probs, y_cls, test_size=0.5, random_state=seed)

    n_classes = 5
    hy = np.arange(1, n_classes + 1, dtype=float)
    predictor = WeightedCRPredictor(hy)
    lambda_val = predictor.find_lambda(Xc, yc, ALPHA)
    pred_results = predictor.run_predictions(Xt, yt, lambda_val)

    lo = np.array([r[0] + 1.0 for r in pred_results])
    hi = np.array([r[1] + 1.0 for r in pred_results])
    return lo, hi, yt.astype(float) + 1.0

cr, wr, ca, wa = run_seeds(ordinal_rc_seed, "OrdinalRC (5-class)")
all_results['OrdinalRC'] = (cr, wr, ca, wa)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 80)
print(f"FINAL SUMMARY — All Methods (α={ALPHA}, target=0.90, {N_SEEDS} seeds)")
print("=" * 80)
print(f"\n{'Method':<22s} │ {'Cov(raw)':>13s} {'Width(raw)':>13s} │ {'Cov(adj)':>13s} {'Width(adj)':>13s}")
print("─" * 80)

for method in ['CQR', 'AsymCQR', 'CHR', 'LVD', 'BoostedCQR', 'BoostedLCP', 'R2CCP', 'OrdinalAPS', 'OrdinalRC']:
    if method in all_results:
        cr, wr, ca, wa = all_results[method]
        if cr:
            print(f"{method:<22s} │ {np.mean(cr):.4f}±{np.std(cr):.4f} {np.mean(wr):.4f}±{np.std(wr):.4f} │ {np.mean(ca):.4f}±{np.std(ca):.4f} {np.mean(wa):.4f}±{np.std(wa):.4f}")
        else:
            print(f"{method:<22s} │ FAILED")

print("\nDone!")
