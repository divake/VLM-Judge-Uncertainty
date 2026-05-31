#!/usr/bin/env python3
"""
Rebuttal Exp 1b: CQR and CHR under Protocol A (current 50/50) vs
Protocol B (external 40/10/50 split). Addresses YU66 R1 explicitly naming
R2CCP, CQR, CHR. R2CCP is covered by exp1.

Protocol A (paper): train fitted model on 50%, conformal step on same 50%
(CQR's internal split; CHR with raw alpha as in paper code).

Protocol B (clean split):
- 40% train fitted model
- 10% disjoint conformal calibration
- 50% test (never touched)

For CQR: MapieQuantileRegressor supports X_calib / y_calib explicitly.
For CHR: Train QNet on 40%; use HistogramAccumulator.calibrate_intervals on 10%
to obtain conformity scores; use empirical (1-alpha) quantile of those scores
to call predict_intervals on the test set.
"""
import sys, os, time, random, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# CHR package path
sys.path.insert(0, '/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge/chr')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.metrics import regression_coverage_score

from chr.black_boxes import QNet
from chr.histogram import Histogram
from chr.grey_boxes import HistogramAccumulator
from scipy.stats.mstats import mquantiles

ALPHA = 0.10
N_SEEDS = 10
SCORE_RANGE = (1, 5)

FEATURE_FILES = {
    'LLaVA-Critic': 'results/v2/features_s2.csv',
    'Phi-4':        'results/v2_phi4/features_s2.csv',
    'Gemini':       'results/v2_gemini/features_s2.csv',
}
LP_COLS = ['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']


def boundary_adjust_expand(lo, hi):
    return np.clip(np.floor(lo), *SCORE_RANGE), np.clip(np.ceil(hi), *SCORE_RANGE)


def metrics(lo, hi, y):
    cov = float(np.mean((y >= lo) & (y <= hi)))
    w = float(np.mean(hi - lo))
    return cov, w


# ---------------- CQR ----------------
def run_cqr_A(X, y, seed):
    """Protocol A: paper protocol — Mapie internal 50/50 cal split."""
    random.seed(seed); np.random.seed(seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    gb = GradientBoostingRegressor(loss='quantile', random_state=seed)
    mq = MapieQuantileRegressor(estimator=gb, alpha=ALPHA)
    mq.fit(X_cal, y_cal, random_state=seed)  # internal split via calib_size=0.3 default
    _, y_pis = mq.predict(X_test, symmetry=True)
    lo = np.clip(y_pis[:, 0, 0], *SCORE_RANGE)
    hi = np.clip(y_pis[:, 1, 0], *SCORE_RANGE)
    cov_raw, w_raw = metrics(lo, hi, y_test)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, y_test)
    return cov_raw, w_raw, cov_adj, w_adj


def run_cqr_B(X, y, seed):
    """Protocol B: explicit 40/10/50 — X_calib/y_calib passed externally."""
    random.seed(seed); np.random.seed(seed)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.20, random_state=seed)
    gb = GradientBoostingRegressor(loss='quantile', random_state=seed)
    mq = MapieQuantileRegressor(estimator=gb, alpha=ALPHA)
    mq.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib, random_state=seed)
    _, y_pis = mq.predict(X_test, symmetry=True)
    lo = np.clip(y_pis[:, 0, 0], *SCORE_RANGE)
    hi = np.clip(y_pis[:, 1, 0], *SCORE_RANGE)
    cov_raw, w_raw = metrics(lo, hi, y_test)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, y_test)
    return cov_raw, w_raw, cov_adj, w_adj


# ---------------- CHR ----------------
def _chr_qnet_and_grid(d_in, seed):
    grid_q = np.arange(0.01, 1.0, 0.01)
    bbox = QNet(
        grid_q, d_in,
        no_crossing=True, batch_size=32, dropout=0.1,
        num_epochs=1000, learning_rate=0.0005, num_hidden=256,
        calibrate=False,
    )
    return bbox, grid_q


def run_chr_A(X, y, seed):
    """Protocol A: paper code path. Train QNet on 50%, predict on test 50%
    with raw alpha (no explicit calibration step)."""
    random.seed(seed); np.random.seed(seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    bbox, grid_q = _chr_qnet_and_grid(X_cal.shape[1], seed)
    bbox.fit(X_cal, y_cal)
    grid_h = np.arange(0, 6, 0.1)
    hist = Histogram(grid_q, grid_h)
    Q_test = bbox.predict(X_test)
    hist_test = hist.compute_histogram(Q_test, 0, 6, 0.001)
    acc = HistogramAccumulator(hist_test, grid_h, alpha=ALPHA, delta_alpha=0.01)
    epsilon = np.random.uniform(0.0, 1.0, X_test.shape[0])
    S, bands = acc.predict_intervals(ALPHA, epsilon=epsilon)
    lo = bands[:, 0]; hi = bands[:, 1]
    lo = np.clip(lo, *SCORE_RANGE); hi = np.clip(hi, *SCORE_RANGE)
    cov_raw, w_raw = metrics(lo, hi, y_test)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, y_test)
    return cov_raw, w_raw, cov_adj, w_adj


def run_chr_B(X, y, seed):
    """Protocol B: 40/10/50 split. Train QNet on 40%; compute conformity
    scores on disjoint 10% via calibrate_intervals; finite-sample (1-alpha)
    quantile of scores gives the corrected alpha used for test intervals."""
    random.seed(seed); np.random.seed(seed)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.20, random_state=seed)
    bbox, grid_q = _chr_qnet_and_grid(X_train.shape[1], seed)
    bbox.fit(X_train, y_train)
    grid_h = np.arange(0, 6, 0.1)
    hist = Histogram(grid_q, grid_h)

    # Calibration step on disjoint 10% — follows chr.methods.CHR.calibrate exactly.
    Q_cal = bbox.predict(X_calib)
    hist_cal = hist.compute_histogram(Q_cal, 0, 6, 0.001)
    acc_cal = HistogramAccumulator(hist_cal, grid_h, alpha=ALPHA, delta_alpha=0.01)
    eps_cal = np.random.uniform(0.0, 1.0, X_calib.shape[0])
    scores = acc_cal.calibrate_intervals(y_calib, epsilon=eps_cal, verbose=False)
    n2 = len(scores)
    level_adj = (1.0 - ALPHA) * (1.0 + 1.0 / float(n2))
    level_adj = float(np.clip(level_adj, 0.0, 1.0))
    alpha_eff = float(np.round(1.0 - mquantiles(scores, prob=level_adj)[0], 4))
    alpha_eff = float(np.clip(alpha_eff, 1e-3, 0.5))

    # Test prediction with corrected alpha.
    Q_test = bbox.predict(X_test)
    hist_test = hist.compute_histogram(Q_test, 0, 6, 0.001)
    acc_test = HistogramAccumulator(hist_test, grid_h, alpha=alpha_eff, delta_alpha=0.01)
    eps_test = np.random.uniform(0.0, 1.0, X_test.shape[0])
    S, bands = acc_test.predict_intervals(alpha_eff, epsilon=eps_test)
    lo = bands[:, 0]; hi = bands[:, 1]
    lo = np.clip(lo, *SCORE_RANGE); hi = np.clip(hi, *SCORE_RANGE)
    cov_raw, w_raw = metrics(lo, hi, y_test)
    la, ha = boundary_adjust_expand(lo, hi)
    cov_adj, w_adj = metrics(la, ha, y_test)
    return cov_raw, w_raw, cov_adj, w_adj


# ---------------- Main ----------------
os.makedirs('results/rebuttal/exp1b', exist_ok=True)
rows = []

METHODS = [
    ('CQR', 'A_internal_split', run_cqr_A),
    ('CQR', 'B_external_split', run_cqr_B),
    ('CHR', 'A_paper_protocol', run_chr_A),
    ('CHR', 'B_external_split', run_chr_B),
]

for judge_name, feat_path in FEATURE_FILES.items():
    if not os.path.exists(feat_path):
        print(f"[SKIP] {judge_name}: {feat_path} not found")
        continue
    df = pd.read_csv(feat_path)
    df = df[df.gt_score >= 1].reset_index(drop=True)
    X = df[LP_COLS].to_numpy().astype(np.float32)
    y = df['gt_score'].to_numpy().astype(np.float32)
    print(f"\n{'='*70}\n{judge_name}: N={len(df)} samples\n{'='*70}", flush=True)

    for seed in range(1, N_SEEDS + 1):
        for method_name, proto_name, proto_fn in METHODS:
            t0 = time.time()
            try:
                cov_raw, w_raw, cov_adj, w_adj = proto_fn(X, y, seed)
                dt = time.time() - t0
                rows.append({
                    'judge': judge_name, 'method': method_name,
                    'protocol': proto_name, 'seed': seed,
                    'cov_raw': cov_raw, 'width_raw': w_raw,
                    'cov_adj': cov_adj, 'width_adj': w_adj,
                    'elapsed_sec': dt,
                })
                print(f"  {method_name:4s} seed={seed:2d} {proto_name}: "
                      f"cov_raw={cov_raw:.4f} w_raw={w_raw:.4f}  "
                      f"cov_adj={cov_adj:.4f} w_adj={w_adj:.4f}  ({dt:.1f}s)", flush=True)
                pd.DataFrame(rows).to_csv('results/rebuttal/exp1b/per_seed.csv', index=False)
            except Exception as e:
                print(f"  {method_name:4s} seed={seed:2d} {proto_name}: FAILED — {e}", flush=True)
            sys.stdout.flush()

df_out = pd.DataFrame(rows)
df_out.to_csv('results/rebuttal/exp1b/per_seed.csv', index=False)
summary = df_out.groupby(['judge', 'method', 'protocol']).agg(
    cov_raw_mean=('cov_raw', 'mean'), cov_raw_std=('cov_raw', 'std'),
    w_raw_mean=('width_raw', 'mean'), w_raw_std=('width_raw', 'std'),
    cov_adj_mean=('cov_adj', 'mean'), cov_adj_std=('cov_adj', 'std'),
    w_adj_mean=('width_adj', 'mean'), w_adj_std=('width_adj', 'std'),
).reset_index()
summary.to_csv('results/rebuttal/exp1b/summary.csv', index=False)
print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))
print("\nSaved: results/rebuttal/exp1b/{per_seed.csv, summary.csv}")
