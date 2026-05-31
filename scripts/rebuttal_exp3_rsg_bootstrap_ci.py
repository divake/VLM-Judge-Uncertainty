#!/usr/bin/env python3
"""
Rebuttal Experiment 3 (for Reviewer WoGp R5c): Bootstrap 95% CIs on
Pearson rho, R2CCP width, and RSG, per dataset, per judge.

Reviewer's concern: Table 17 (RSG by dataset and judge) reports point
estimates with no statistical reliability. With only 10 seeds and modest
per-dataset N, the differences could be noise.

Plan:
  1) Re-run R2CCP per (judge, dataset) for 10 seeds, saving the PER-SEED
     coverage, width, and (where >=10 valid parsed scores) Pearson rho.
  2) Compute means and bootstrap 95% CIs (B=2000) over the seed values.
  3) Compute RSG per seed as |rho| - (1 - w/(K-1)), match the formula in
     §5.5 of the paper exactly. Bootstrap CIs likewise.
  4) Emit a fresh diagnostic CSV with means and 95% CI lower/upper bounds
     for each of (rho, width_raw, cp_informativeness, RSG), per dataset
     and judge.

This is the same pipeline as scripts/run_r2ccp_per_dataset.py but persists
the per-seed values so we can bootstrap.

K = 5 score levels, so K - 1 = 4 in the RSG formula (paper §5.5).
alpha = 0.10, 10 seeds.
"""
import sys, os, time, json, random, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2CCP.main import R2CCP
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

ALPHA = 0.10
N_SEEDS = 10
N_BOOT  = 2000
K = 5  # score levels: 1..5
LP_COLS = ['s2_lp1','s2_lp2','s2_lp3','s2_lp4','s2_lp5']
SCORE_RANGE = (1, 5)

JUDGES = {
    'LLaVA-Critic': 'results/v2/features_s2.csv',
    'Phi-4':        'results/v2_phi4/features_s2.csv',
    'Gemini':       'results/v2_gemini/features_s2.csv',
}

# Dataset metadata (matches scripts/run_r2ccp_per_dataset.py taxonomy)
DATASET_CATEGORIES = {
    'Vision-Heavy':  ['mathvista', 'ChartQA', 'infographicsVQA', 'ScienceQA', 'textVQA'],
    'General VQA':   ['coco', 'llava_bench', 'mm-vet', 'VisitBench'],
    'Knowledge/Web': ['WIT', 'Concept Caption', 'mind2web'],
    'Aesthetics/AI': ['AesBench', 'diffusiondb'],
}
ds_to_cat = {ds: cat for cat, lst in DATASET_CATEGORIES.items() for ds in lst}

def range_mod(lo, hi): return np.clip(lo, *SCORE_RANGE), np.clip(hi, *SCORE_RANGE)
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
def bootstrap_ci(values, n_boot=N_BOOT, ci=0.95):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = []
    rng = np.random.RandomState(0)
    for _ in range(n_boot):
        idx = rng.randint(0, len(values), len(values))
        boots.append(values[idx].mean())
    lo = float(np.percentile(boots, 100*(1-ci)/2))
    hi = float(np.percentile(boots, 100*(1+ci)/2))
    return float(np.mean(values)), lo, hi

# Load dataset metadata once
META = {}
with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
    for i, line in enumerate(f):
        META[i] = json.loads(line).get('original_dataset', 'unknown')

os.makedirs('results/rebuttal/exp3', exist_ok=True)

# ---- Run R2CCP per (judge, dataset, seed), keep per-seed ----
per_seed_rows = []

for judge, feat_path in JUDGES.items():
    if not os.path.exists(feat_path):
        print(f"[SKIP] {judge}: {feat_path} not found")
        continue
    df = pd.read_csv(feat_path)
    df = df[df.gt_score >= 1].reset_index(drop=True)
    df['orig_dataset'] = df['sample_id'].map(lambda sid: META.get(int(sid), 'unknown'))
    print(f"\n{'#'*70}\n# {judge}: N={len(df)} samples across {df['orig_dataset'].nunique()} datasets\n{'#'*70}")

    for ds in sorted(df['orig_dataset'].unique()):
        sub = df[df['orig_dataset'] == ds]
        if len(sub) < 50:
            print(f"  [skip {ds}: n={len(sub)} too small]")
            continue
        X = sub[LP_COLS].to_numpy().astype(np.float32)
        y = sub['gt_score'].to_numpy().astype(np.float32)
        parsed = sub['parsed_score'].to_numpy()
        gt = sub['gt_score'].to_numpy()

        t0 = time.time()
        for seed in range(1, N_SEEDS + 1):
            try:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                Xc, Xt, yc, yt = train_test_split(X, y, test_size=0.5, random_state=seed)
                tmp = f'results/rebuttal/exp3/tmp_{judge}_{ds}_s{seed}.pth'.replace(' ', '_')
                if os.path.exists(tmp): os.remove(tmp)
                model = R2CCP({'model_path': tmp, 'max_epochs': 100, 'alpha': ALPHA, 'seed': seed})
                model.fit(Xc, yc.flatten())
                intervals = model.get_intervals(Xt)
                lo, hi = merge_intervals(intervals)
                lo, hi = range_mod(lo, hi)
                cov_raw = float(np.mean((yt >= lo) & (yt <= hi)))
                w_raw = float(np.mean(hi - lo))
                la, ha = boundary_adjust_expand(lo, hi)
                cov_adj = float(np.mean((yt >= la) & (yt <= ha)))
                w_adj = float(np.mean(ha - la))

                # Per-seed Pearson on the TEST set's parsed_score vs gt
                test_idx = np.isin(np.arange(len(y)), np.where(np.isin(y, yt))[0])  # not exact; recompute properly
                # Recompute properly: re-derive the split indices
                idx_all = np.arange(len(y))
                _, idx_test = train_test_split(idx_all, test_size=0.5, random_state=seed)
                p_test = parsed[idx_test]
                g_test = gt[idx_test]
                valid = p_test > 0
                if valid.sum() >= 10:
                    rho, _ = pearsonr(p_test[valid], g_test[valid])
                else:
                    rho = np.nan

                cpi = 1.0 - w_raw / (K - 1)
                rsg = abs(rho) - cpi if not np.isnan(rho) else np.nan

                per_seed_rows.append({
                    'judge': judge, 'dataset': ds, 'category': ds_to_cat.get(ds, 'Other'),
                    'n': len(sub), 'seed': seed,
                    'cov_raw': cov_raw, 'width_raw': w_raw,
                    'cov_adj': cov_adj, 'width_adj': w_adj,
                    'pearson': rho, 'cp_informativeness': cpi, 'rsg': rsg,
                })

                del model
                torch.cuda.empty_cache()
                if os.path.exists(tmp): os.remove(tmp)
            except Exception as e:
                print(f"    {judge}/{ds}/seed{seed} FAILED: {e}")
        dt = time.time() - t0
        # Print one summary line per dataset
        sub_rows = [r for r in per_seed_rows if r['judge']==judge and r['dataset']==ds]
        if sub_rows:
            w = np.mean([r['width_raw'] for r in sub_rows])
            c = np.mean([r['cov_raw']   for r in sub_rows])
            rh = np.nanmean([r['pearson'] for r in sub_rows])
            rs = np.nanmean([r['rsg']     for r in sub_rows])
            print(f"  {ds:<22s}  cov={c:.4f}  w={w:.4f}  rho={rh:.4f}  RSG={rs:+.4f}  ({dt:.0f}s)", flush=True)
        sys.stdout.flush()
        pd.DataFrame(per_seed_rows).to_csv('results/rebuttal/exp3/per_seed.csv', index=False)

df_seed = pd.DataFrame(per_seed_rows)
df_seed.to_csv('results/rebuttal/exp3/per_seed.csv', index=False)
print(f"\nSaved per-seed: results/rebuttal/exp3/per_seed.csv (n={len(df_seed)} rows)")

# ---- Bootstrap CIs per (judge, dataset) ----
print("\nComputing bootstrap 95% CIs...")
ci_rows = []
for (judge, ds), grp in df_seed.groupby(['judge','dataset']):
    cat = grp['category'].iloc[0]
    n = int(grp['n'].iloc[0])
    rho_m, rho_lo, rho_hi = bootstrap_ci(grp['pearson'].values)
    w_m, w_lo, w_hi       = bootstrap_ci(grp['width_raw'].values)
    cpi_m, cpi_lo, cpi_hi = bootstrap_ci(grp['cp_informativeness'].values)
    rsg_m, rsg_lo, rsg_hi = bootstrap_ci(grp['rsg'].values)
    ci_rows.append({
        'judge': judge, 'dataset': ds, 'category': cat, 'n': n,
        'pearson_mean': rho_m, 'pearson_lo': rho_lo, 'pearson_hi': rho_hi,
        'width_mean': w_m, 'width_lo': w_lo, 'width_hi': w_hi,
        'cpi_mean': cpi_m, 'cpi_lo': cpi_lo, 'cpi_hi': cpi_hi,
        'rsg_mean': rsg_m, 'rsg_lo': rsg_lo, 'rsg_hi': rsg_hi,
    })
ci_df = pd.DataFrame(ci_rows).sort_values(['judge','category','dataset'])
ci_df.to_csv('results/rebuttal/exp3/rsg_with_95ci.csv', index=False)
print(f"Saved CIs: results/rebuttal/exp3/rsg_with_95ci.csv (n={len(ci_df)} dataset×judge rows)")

# ---- Print formatted table ----
print("\n" + "="*120)
print(f"{'Judge':<14s} {'Dataset':<20s} {'rho':>22s}   {'width':>22s}   {'RSG':>22s}")
print("="*120)
for _, r in ci_df.iterrows():
    rho_s = f"{r['pearson_mean']:+.3f} [{r['pearson_lo']:+.3f},{r['pearson_hi']:+.3f}]"
    w_s   = f"{r['width_mean']:.3f} [{r['width_lo']:.3f},{r['width_hi']:.3f}]"
    rsg_s = f"{r['rsg_mean']:+.3f} [{r['rsg_lo']:+.3f},{r['rsg_hi']:+.3f}]"
    print(f"{r['judge']:<14s} {r['dataset']:<20s} {rho_s:>22s}   {w_s:>22s}   {rsg_s:>22s}")
print("="*120)
print("Done.")
