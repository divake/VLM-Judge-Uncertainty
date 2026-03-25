#!/usr/bin/env python3
"""Full analysis for Phi-4 results — mirrors all LLaVA-Critic analyses."""

import sys, os
import numpy as np
import pandas as pd
import random
import torch
import time
import json
import math
import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LLM_REPO = '/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge'
sys.path.insert(0, os.path.join(LLM_REPO, 'chr'))
sys.path.insert(0, os.path.join(LLM_REPO, 'LvD'))
sys.path.insert(0, os.path.join(LLM_REPO, 'boosted-conformal'))
sys.path.insert(0, os.path.join(LLM_REPO, 'boosted-conformal', 'third_party'))

from sklearn.model_selection import train_test_split
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr, kendalltau

ALPHA = 0.10
N_SEEDS = 10
OUT_DIR = 'results/v2_gemini'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(f'{OUT_DIR}/features_s2.csv')
df = df[df.gt_score >= 1].reset_index(drop=True)
X_logits = df[['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']].to_numpy().astype(np.float32)
y_all = df['gt_score'].to_numpy().astype(np.float32)
parsed = df['parsed_score'].to_numpy().astype(np.float32)
X_probs = softmax(X_logits, axis=1).astype(np.float32)

# Load dataset metadata
meta = {}
with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        meta[i] = d.get('original_dataset', 'unknown')
df['orig_dataset'] = df['sample_id'].map(lambda sid: meta.get(int(sid), 'unknown'))

# Also load LLaVA-Critic for comparison
df_llava = pd.read_csv('results/v2/features_s2.csv')
df_llava = df_llava[df_llava.gt_score >= 1].reset_index(drop=True)

print(f"Gemini data: {len(df)} samples")
print(f"LLaVA-Critic data: {len(df_llava)} samples")

# ============================================================
# 1. POINT PREDICTION COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("1. POINT PREDICTION: Phi-4 vs LLaVA-Critic")
print("=" * 70)

results_rows = []
for label, d in [("LLaVA-Critic-7B", df_llava), ("Gemini-2.5-Flash", df)]:
    valid = d[d.parsed_score > 0]
    p, _ = pearsonr(valid.parsed_score, valid.gt_score)
    s, _ = spearmanr(valid.parsed_score, valid.gt_score)
    k, _ = kendalltau(valid.parsed_score, valid.gt_score)
    acc = (valid.parsed_score == valid.gt_score).mean()
    mae = (valid.parsed_score - valid.gt_score).abs().mean()
    within1 = ((valid.parsed_score - valid.gt_score).abs() <= 1).mean()
    bias = (valid.parsed_score - valid.gt_score).mean()

    X = d[['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']].to_numpy()
    X_sm = softmax(X, axis=1)
    max_p = X_sm.max(axis=1)

    print(f"\n{label} ({len(valid)} valid)")
    print(f"  Pearson={p:.4f}, Spearman={s:.4f}, Kendall={k:.4f}")
    print(f"  Accuracy={acc:.4f}, ±1 Accuracy={within1:.4f}, MAE={mae:.4f}")
    print(f"  Bias={bias:+.4f}")
    print(f"  Max softmax: mean={max_p.mean():.4f}, >0.99={100*(max_p>0.99).mean():.1f}%, >0.999={100*(max_p>0.999).mean():.1f}%")

    results_rows.append({
        'judge': label, 'n_samples': len(valid),
        'pearson': round(p,4), 'spearman': round(s,4), 'kendall': round(k,4),
        'accuracy': round(acc,4), 'within_1_accuracy': round(within1,4),
        'mae': round(mae,4), 'bias': round(bias,4),
        'max_softmax_mean': round(max_p.mean(),4),
        'overconfident_99': round((max_p>0.99).mean(),4),
        'overconfident_999': round((max_p>0.999).mean(),4),
    })

pd.DataFrame(results_rows).to_csv(f'{OUT_DIR}/point_prediction_comparison.csv', index=False)
print(f"\nSaved: {OUT_DIR}/point_prediction_comparison.csv")

# ============================================================
# 2. ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("2. ERROR ANALYSIS")
print("=" * 70)

valid = df[df.parsed_score > 0].copy()
valid['error'] = valid.parsed_score - valid.gt_score
valid['abs_error'] = valid.error.abs()

# Error distribution
print(f"\nError distribution:")
for e in range(5):
    n = (valid.abs_error == e).sum()
    print(f"  |Error|={e}: {n:5d} ({n/len(valid)*100:.1f}%)")

# Relaxed accuracy
print(f"\nRelaxed accuracy:")
for tol in [0, 1, 2, 3]:
    acc = (valid.abs_error <= tol).mean()
    print(f"  ±{tol}: {acc:.1%}")

# Confusion matrix
print(f"\nConfusion matrix:")
cm_rows = []
for gt in range(1, 6):
    mask = valid.gt_score == gt
    total = mask.sum()
    row = {'gt_score': gt, 'total': total}
    for j in range(1, 6):
        n = ((valid.gt_score == gt) & (valid.parsed_score == j)).sum()
        row[f'pred_{j}'] = n
        row[f'pred_{j}_pct'] = round(n/total*100, 2) if total > 0 else 0
    within1 = ((valid[mask].parsed_score - gt).abs() <= 1).sum()
    row['within_1_accuracy'] = round(within1/total, 4) if total > 0 else 0
    cm_rows.append(row)
    pcts = [f"{row[f'pred_{j}_pct']:5.1f}%" for j in range(1, 6)]
    print(f"  GT={gt}: {' '.join(pcts)}  ±1={row['within_1_accuracy']:.1%}")

pd.DataFrame(cm_rows).to_csv(f'{OUT_DIR}/confusion_matrix.csv', index=False)

# Per-GT bias
print(f"\nPer-GT bias:")
for gt in range(1, 6):
    mask = valid.gt_score == gt
    bias = (valid[mask].parsed_score - gt).mean()
    print(f"  GT={gt}: {bias:+.3f}")

# Per-dataset relaxed accuracy
print(f"\nPer-dataset ±1 accuracy:")
ds_rows = []
for ds in sorted(valid.orig_dataset.unique()):
    sub = valid[valid.orig_dataset == ds]
    if len(sub) < 10: continue
    ae = sub.abs_error
    row = {
        'dataset': ds, 'n': len(sub),
        'exact_accuracy': round((ae==0).mean(), 4),
        'within_1_accuracy': round((ae<=1).mean(), 4),
        'within_2_accuracy': round((ae<=2).mean(), 4),
        'mae': round(ae.mean(), 4),
        'judge_bias': round((sub.parsed_score - sub.gt_score).mean(), 4),
    }
    ds_rows.append(row)
    print(f"  {ds:22s}: ±1={row['within_1_accuracy']:.1%}, MAE={row['mae']:.3f}")

pd.DataFrame(ds_rows).to_csv(f'{OUT_DIR}/error_analysis_per_dataset.csv', index=False)
print(f"Saved: {OUT_DIR}/error_analysis_per_dataset.csv, confusion_matrix.csv")

# ============================================================
# 3. R2CCP (10 seeds)
# ============================================================
print("\n" + "=" * 70)
print("3. R2CCP (10 seeds)")
print("=" * 70)

from R2CCP.main import R2CCP

def run_r2ccp(X, y, n_seeds=N_SEEDS):
    coverages_raw, widths_raw, coverages_adj, widths_adj = [], [], [], []
    for seed in range(1, n_seeds+1):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        Xc, Xt, yc, yt = train_test_split(X, y, test_size=0.5, random_state=seed)
        mp = f'{OUT_DIR}/tmp_r2ccp.pth'
        if os.path.exists(mp): os.remove(mp)
        model = R2CCP({'model_path': mp, 'max_epochs': 100, 'alpha': ALPHA})
        model.fit(Xc, yc.flatten())
        intervals = model.get_intervals(Xt)
        merged = []
        for si in intervals:
            if not si: merged.append((1,5))
            else: merged.append((min(l for l,h in si), max(h for l,h in si)))
        lo = np.clip(np.array([iv[0] for iv in merged]), 1, 5)
        hi = np.clip(np.array([iv[1] for iv in merged]), 1, 5)
        coverages_raw.append(((yt>=lo)&(yt<=hi)).mean())
        widths_raw.append((hi-lo).mean())
        la, ha = np.clip(np.floor(lo),1,5), np.clip(np.ceil(hi),1,5)
        coverages_adj.append(((yt>=la)&(yt<=ha)).mean())
        widths_adj.append((ha-la).mean())
        del model; torch.cuda.empty_cache()
        if os.path.exists(mp): os.remove(mp)
    return (np.mean(coverages_raw), np.std(coverages_raw), np.mean(widths_raw), np.std(widths_raw),
            np.mean(coverages_adj), np.std(coverages_adj), np.mean(widths_adj), np.std(widths_adj))

cr_m, cr_s, wr_m, wr_s, ca_m, ca_s, wa_m, wa_s = run_r2ccp(X_logits, y_all)
print(f"R2CCP: Raw Coverage={cr_m:.4f}±{cr_s:.4f}, Width={wr_m:.4f}±{wr_s:.4f}")
print(f"        Adj Coverage={ca_m:.4f}±{ca_s:.4f}, Width={wa_m:.4f}±{wa_s:.4f}")

# ============================================================
# 4. PER-DATASET R2CCP
# ============================================================
print("\n" + "=" * 70)
print("4. PER-DATASET R2CCP")
print("=" * 70)

ds_r2ccp_rows = []
for ds in sorted(df.orig_dataset.unique()):
    mask = df.orig_dataset == ds
    sub = df[mask]
    if len(sub) < 50: continue
    Xi = sub[['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']].to_numpy().astype(np.float32)
    yi = sub['gt_score'].to_numpy().astype(np.float32)

    valid_sub = sub[sub.parsed_score > 0]
    pear = pearsonr(valid_sub.parsed_score, valid_sub.gt_score)[0] if len(valid_sub) > 10 else 0

    try:
        cr, cs, wr, ws, ca, cas, wa, was = run_r2ccp(Xi, yi, n_seeds=N_SEEDS)
        print(f"  {ds:22s} N={len(sub):4d}: Width(raw)={wr:.3f}, Width(adj)={wa:.3f}, Pearson={pear:.3f}")
        ds_r2ccp_rows.append({
            'dataset': ds, 'n': len(sub), 'pearson': round(pear,4),
            'cov_raw': round(cr,4), 'width_raw': round(wr,4),
            'cov_adj': round(ca,4), 'width_adj': round(wa,4),
        })
    except Exception as e:
        print(f"  {ds:22s}: FAILED ({e})")

pd.DataFrame(ds_r2ccp_rows).to_csv(f'{OUT_DIR}/r2ccp_per_dataset.csv', index=False)
print(f"Saved: {OUT_DIR}/r2ccp_per_dataset.csv")

# ============================================================
# 5. CP VALUE ANALYSIS (Error bins)
# ============================================================
print("\n" + "=" * 70)
print("5. CP VALUE ANALYSIS (Error bins, 10 seeds)")
print("=" * 70)

bin_stats = {e: {'cov_raw':[], 'cov_adj':[], 'w_raw':[], 'w_adj':[]} for e in range(5)}

for seed in range(1, N_SEEDS+1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    idx_all = np.arange(len(y_all))
    idx_cal, idx_test = train_test_split(idx_all, test_size=0.5, random_state=seed)
    Xc, Xt = X_logits[idx_cal], X_logits[idx_test]
    yc, yt = y_all[idx_cal], y_all[idx_test]
    par_t = parsed[idx_test]

    mp = f'{OUT_DIR}/tmp_r2ccp.pth'
    if os.path.exists(mp): os.remove(mp)
    model = R2CCP({'model_path': mp, 'max_epochs': 100, 'alpha': ALPHA})
    model.fit(Xc, yc.flatten())
    intervals = model.get_intervals(Xt)
    merged = []
    for si in intervals:
        if not si: merged.append((1,5))
        else: merged.append((min(l for l,h in si), max(h for l,h in si)))
    lo = np.clip(np.array([iv[0] for iv in merged]), 1, 5)
    hi = np.clip(np.array([iv[1] for iv in merged]), 1, 5)
    la, ha = np.clip(np.floor(lo),1,5), np.clip(np.ceil(hi),1,5)

    err_t = np.abs(par_t - yt).astype(int)
    cov_r = (yt >= lo) & (yt <= hi)
    cov_a = (yt >= la) & (yt <= ha)

    for e in range(5):
        m = err_t == e
        if m.sum() > 0:
            bin_stats[e]['cov_raw'].append(cov_r[m].mean())
            bin_stats[e]['cov_adj'].append(cov_a[m].mean())
            bin_stats[e]['w_raw'].append((hi-lo)[m].mean())
            bin_stats[e]['w_adj'].append((ha-la)[m].mean())

    del model; torch.cuda.empty_cache()
    if os.path.exists(mp): os.remove(mp)
    print(f"  Seed {seed} done")

bin_rows = []
for e in range(5):
    s = bin_stats[e]
    if not s['cov_raw']: continue
    label = {0:"Exact",1:"±1",2:"±2",3:"±3",4:"±4"}[e]
    row = {
        'error': e, 'label': label,
        'cov_raw_mean': round(np.mean(s['cov_raw']),4), 'cov_raw_std': round(np.std(s['cov_raw']),4),
        'width_raw_mean': round(np.mean(s['w_raw']),4), 'width_raw_std': round(np.std(s['w_raw']),4),
        'cov_adj_mean': round(np.mean(s['cov_adj']),4), 'cov_adj_std': round(np.std(s['cov_adj']),4),
        'width_adj_mean': round(np.mean(s['w_adj']),4), 'width_adj_std': round(np.std(s['w_adj']),4),
    }
    bin_rows.append(row)
    print(f"  {label:>7s}: Cov_raw={row['cov_raw_mean']:.4f}±{row['cov_raw_std']:.4f}, Cov_adj={row['cov_adj_mean']:.4f}±{row['cov_adj_std']:.4f}")

pd.DataFrame(bin_rows).to_csv(f'{OUT_DIR}/error_bins_cp_coverage.csv', index=False)
print(f"Saved: {OUT_DIR}/error_bins_cp_coverage.csv")

# ============================================================
# 6. ALL CP METHODS (same as run_all_conformal.py but for Phi-4)
# ============================================================
print("\n" + "=" * 70)
print("6. KEY CP METHODS: R2CCP, CHR, Boosted LCP, CQR")
print("=" * 70)

# CQR
from sklearn.ensemble import GradientBoostingRegressor
from mapie.quantile_regression import MapieQuantileRegressor

def run_method(method_fn, label):
    cr, wr, ca, wa = [], [], [], []
    for seed in range(1, N_SEEDS+1):
        try:
            lo, hi, yt = method_fn(seed)
            cr.append(((yt>=lo)&(yt<=hi)).mean()); wr.append((hi-lo).mean())
            la, ha = np.clip(np.floor(lo),1,5), np.clip(np.ceil(hi),1,5)
            ca.append(((yt>=la)&(yt<=ha)).mean()); wa.append((ha-la).mean())
        except Exception as e:
            print(f"  {label} seed {seed} failed: {e}")
    if cr:
        print(f"  {label:20s}: Raw={np.mean(cr):.4f}±{np.std(cr):.4f}/{np.mean(wr):.4f}±{np.std(wr):.4f}  Adj={np.mean(ca):.4f}±{np.std(ca):.4f}/{np.mean(wa):.4f}±{np.std(wa):.4f}")
    return cr, wr, ca, wa

def cqr_fn(seed):
    random.seed(seed); np.random.seed(seed)
    Xc,Xt,yc,yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    gb = GradientBoostingRegressor(loss="quantile", random_state=seed)
    mqr = MapieQuantileRegressor(estimator=gb, alpha=ALPHA)
    mqr.fit(Xc, yc, random_state=seed)
    _, pis = mqr.predict(Xt, symmetry=False)
    return np.clip(pis[:,0].ravel(),1,5), np.clip(pis[:,1].ravel(),1,5), yt

all_methods = {}
cr,wr,ca,wa = run_method(cqr_fn, "CQR")
all_methods['CQR'] = {'cov_raw':np.mean(cr),'width_raw':np.mean(wr),'cov_adj':np.mean(ca),'width_adj':np.mean(wa)}

# CHR
from chr.black_boxes import QNet
from chr.histogram import Histogram
from chr.grey_boxes import HistogramAccumulator

def chr_fn(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc,Xt,yc,yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    grid_q = np.arange(0.01,1.0,0.01)
    bbox = QNet(grid_q, Xc.shape[1], no_crossing=True, batch_size=32, dropout=0.1,
                num_epochs=500, learning_rate=0.0005, num_hidden=256, calibrate=False)
    bbox.fit(Xc, yc)
    grid_h = np.arange(0,6,0.1)
    hist = Histogram(grid_q, grid_h)
    Q_test = bbox.predict(Xt)
    hist_test = hist.compute_histogram(Q_test, 0, 6, 0.001)
    acc = HistogramAccumulator(hist_test, grid_h, alpha=ALPHA, delta_alpha=0.01)
    eps = np.random.uniform(0,1,Xt.shape[0])
    S, _ = acc.predict_intervals(ALPHA, epsilon=eps)
    S_int = [np.arange(S[i][0],S[i][1]+1) for i in range(len(S))]
    ivs = np.array([[grid_h[S_int[i]-1][0],grid_h[S_int[i]][-1]] for i in range(len(S_int))])
    return np.clip(ivs[:,0],1,5), np.clip(ivs[:,1],1,5), yt

cr,wr,ca,wa = run_method(chr_fn, "CHR")
all_methods['CHR'] = {'cov_raw':np.mean(cr),'width_raw':np.mean(wr),'cov_adj':np.mean(ca),'width_adj':np.mean(wa)}

# R2CCP already computed above
all_methods['R2CCP'] = {'cov_raw':cr_m,'width_raw':wr_m,'cov_adj':ca_m,'width_adj':wa_m}

# Boosted LCP
from boostedCP.len_local_boost import len_local_boost
def blcp_fn(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xc,Xt,yc,yt = train_test_split(X_logits, y_all, test_size=0.5, random_state=seed)
    results = len_local_boost(Xc,yc,Xc,yc,Xt,yt,ALPHA,seed,n_rounds_cv=500,learning_rate=0.02,store=True,verbose=False)
    return np.clip(results["boosted_lower"].flatten(),1,5), np.clip(results["boosted_upper"].flatten(),1,5), yt

cr,wr,ca,wa = run_method(blcp_fn, "Boosted LCP")
all_methods['BoostedLCP'] = {'cov_raw':np.mean(cr),'width_raw':np.mean(wr),'cov_adj':np.mean(ca),'width_adj':np.mean(wa)}

# Save all methods
pd.DataFrame([{'method':k,**v} for k,v in all_methods.items()]).to_csv(f'{OUT_DIR}/cp_methods_results.csv', index=False)
print(f"\nSaved: {OUT_DIR}/cp_methods_results.csv")

print("\n\nALL ANALYSES COMPLETE!")
