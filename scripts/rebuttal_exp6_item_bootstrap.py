#!/usr/bin/env python3
"""
Rebuttal Exp 6: Item-level bootstrap CIs for per-dataset Pearson rho and
RSG. Replaces the seed-level bootstrap in exp3, which produced CIs ~4x too
tight relative to Fisher z (the seed CIs only quantify split-induced
variation, not sampling uncertainty in the underlying correlation).

For each (judge, dataset):
  - Pearson rho: bootstrap items B=2000.
  - Width: keep seed-level bootstrap (width is a stable summary; the
    sampling-unit objection is specifically about rho).
  - RSG = |rho_item_bs| - (1 - width_seed_mean / (K-1)); CI propagated
    from the rho item bootstrap and the width seed-level CI in quadrature
    via the simple identity Var(RSG) ≈ Var(|rho|) for fixed width, because
    width has much smaller relative SE. Explicitly: we report rho's
    item-level CI and width's seed CI; the combined RSG CI is the
    plug-in interval [|rho_lo| - (1 - w_hi/4), |rho_hi| - (1 - w_lo/4)].

Outputs:
  - results/rebuttal/exp6/rho_item_bootstrap.csv (judge, dataset, n_items,
    rho_pt, rho_lo, rho_hi, width_seed_mean, width_seed_lo, width_seed_hi,
    rsg_lo, rsg_hi).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

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

# Load seed-level widths from exp3
seed_df = pd.read_csv('results/rebuttal/exp3/per_seed.csv')

os.makedirs('results/rebuttal/exp6', exist_ok=True)

rows = []
for judge, fpath in JUDGES.items():
    if not os.path.exists(fpath):
        print(f"[SKIP] {judge}: {fpath} not found")
        continue
    df = pd.read_csv(fpath)
    df = df[df.gt_score >= 1].reset_index(drop=True)
    df['orig_dataset'] = df['sample_id'].map(lambda sid: META.get(int(sid), 'unknown'))
    print(f"\n=== {judge} ===")

    for ds in sorted(df['orig_dataset'].unique()):
        sub = df[df['orig_dataset'] == ds]
        if len(sub) < 50:
            continue
        valid = sub[(sub['parsed_score'] > 0) & sub['gt_score'].notna()]
        if len(valid) < 50:
            print(f"  [skip {ds}: n_valid={len(valid)} < 50]")
            continue
        p = valid['parsed_score'].to_numpy(dtype=np.float64)
        g = valid['gt_score'].to_numpy(dtype=np.float64)
        n = len(p)
        rho_pt = float(pearsonr(p, g)[0])

        rng = np.random.default_rng(SEED)
        rhos = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            pb, gb = p[idx], g[idx]
            if np.std(pb) < 1e-12 or np.std(gb) < 1e-12:
                rhos[b] = np.nan
            else:
                rhos[b] = pearsonr(pb, gb)[0]
        rhos = rhos[~np.isnan(rhos)]
        rho_lo = float(np.percentile(rhos, 2.5))
        rho_hi = float(np.percentile(rhos, 97.5))

        # Width: seed-level bootstrap from exp3 per-seed values.
        sd_rows = seed_df[(seed_df['judge'] == judge) & (seed_df['dataset'] == ds)]
        w_vals = sd_rows['width_raw'].to_numpy(dtype=np.float64)
        if len(w_vals) >= 2:
            rng2 = np.random.default_rng(SEED + 1)
            w_boots = np.array([w_vals[rng2.integers(0, len(w_vals), len(w_vals))].mean()
                                for _ in range(N_BOOT)])
            w_mean = float(np.mean(w_vals))
            w_lo = float(np.percentile(w_boots, 2.5))
            w_hi = float(np.percentile(w_boots, 97.5))
        else:
            w_mean = w_lo = w_hi = np.nan

        # Plug-in RSG CI: RSG = |rho| - (1 - w/(K-1)) = |rho| - 1 + w/(K-1).
        # Monotone increasing in BOTH |rho| AND w, so the worst-case-low bound
        # plugs in both lower CIs, and worst-case-high plugs in both upper CIs.
        if not np.isnan(w_lo):
            rsg_pt = abs(rho_pt) - (1.0 - w_mean / (K - 1))
            rsg_lo = abs(rho_lo) - (1.0 - w_lo / (K - 1))
            rsg_hi = abs(rho_hi) - (1.0 - w_hi / (K - 1))
        else:
            rsg_pt = rsg_lo = rsg_hi = np.nan

        rows.append({
            'judge': judge, 'dataset': ds, 'category': ds_to_cat.get(ds, 'Other'),
            'n_items': n,
            'rho_pt': rho_pt, 'rho_lo': rho_lo, 'rho_hi': rho_hi,
            'rho_halfwidth': (rho_hi - rho_lo) / 2.0,
            'width_seed_mean': w_mean, 'width_seed_lo': w_lo, 'width_seed_hi': w_hi,
            'rsg_pt': rsg_pt, 'rsg_lo': rsg_lo, 'rsg_hi': rsg_hi,
        })
        print(f"  {ds:<22s}  n={n:>4d}  rho={rho_pt:+.3f} [{rho_lo:+.3f},{rho_hi:+.3f}]  "
              f"w={w_mean:.3f} [{w_lo:.3f},{w_hi:.3f}]  "
              f"RSG={rsg_pt:+.3f} [{rsg_lo:+.3f},{rsg_hi:+.3f}]", flush=True)

out = pd.DataFrame(rows)
out.to_csv('results/rebuttal/exp6/rho_item_bootstrap.csv', index=False)
print(f"\nSaved: results/rebuttal/exp6/rho_item_bootstrap.csv ({len(out)} rows)")

# Quick sanity check: Fisher z reference half-width for a few cells
print("\nSanity vs Fisher z (expected analytic half-width):")
for _, r in out.iterrows():
    if r['judge'] == 'LLaVA-Critic' and r['dataset'] in {'ChartQA', 'AesBench', 'WIT', 'mathvista'}:
        n = r['n_items']; rho = r['rho_pt']
        z = np.arctanh(rho)
        se_z = 1.0 / np.sqrt(n - 3)
        rho_lo_fisher = float(np.tanh(z - 1.96 * se_z))
        rho_hi_fisher = float(np.tanh(z + 1.96 * se_z))
        hw_fisher = (rho_hi_fisher - rho_lo_fisher) / 2
        print(f"  {r['judge']:<14s} {r['dataset']:<14s} n={n:>4d}  "
              f"item-bs HW={r['rho_halfwidth']:.4f}  Fisher HW={hw_fisher:.4f}")
