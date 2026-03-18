#!/usr/bin/env python3
"""End-to-end S2-only signal extraction + conformal prediction on full dataset.

Usage:
    python scripts/run_full_analysis.py --config configs/experiments/pilot_mllm_judge.yaml
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml, load_json, load_pt, get_sample_paths


FALLBACK_LP = math.log(1e-5)
SCORE_TOKENS = ["1", "2", "3", "4", "5"]


def find_score_token(tokens):
    """Find score digit token index (3-strategy approach)."""
    stripped = [t.strip().lower() for t in tokens]

    # Strategy 1: "Score" + ":" + digit pattern (backward)
    for i in range(len(tokens) - 1, -1, -1):
        if stripped[i] in SCORE_TOKENS:
            for j in range(max(0, i - 3), i):
                if "score" in stripped[j]:
                    return i

    # Strategy 2: "score" keyword then look ahead
    for i in range(len(tokens) - 1, -1, -1):
        if "score" in stripped[i]:
            for j in range(i + 1, min(len(tokens), i + 5)):
                if stripped[j] in SCORE_TOKENS:
                    return j

    # Strategy 3: Plain backward scan (fallback)
    for i in range(len(tokens) - 1, -1, -1):
        if stripped[i] in SCORE_TOKENS:
            return i

    return None


def extract_s2(json_data, pt_data):
    """Extract Signal 2 (5-dim) from a single sample."""
    tokens = json_data["tokens"]
    top_logprobs = pt_data["top_logprobs"]

    score_idx = find_score_token(tokens)
    if score_idx is None:
        return None

    lp_dict = top_logprobs[score_idx]
    signal_2 = []
    for s in SCORE_TOKENS:
        lp = lp_dict.get(s, None)
        if lp is None:
            for variant in [f"▁{s}", f"Ġ{s}", f" {s}", f"\t{s}"]:
                if variant in lp_dict:
                    lp = lp_dict[variant]
                    break
        if lp is None:
            lp = FALLBACK_LP
        signal_2.append(lp)
    return signal_2


def extract_all_s2(judge_output_dir, num_samples):
    """Extract S2 from all samples, return DataFrame."""
    rows = []
    skipped = 0

    for sid in tqdm(range(num_samples), desc="Extracting S2"):
        json_path, pt_path = get_sample_paths(judge_output_dir, sid)
        if not os.path.exists(json_path) or not os.path.exists(pt_path):
            skipped += 1
            continue

        try:
            json_data = load_json(json_path)
            pt_data = load_pt(pt_path)
            s2 = extract_s2(json_data, pt_data)

            if s2 is None:
                skipped += 1
                continue

            gt = float(json_data.get("gt_score", 0))
            parsed = json_data.get("judge_output", "")

            # Also extract parsed score for point prediction metrics
            tokens = json_data["tokens"]
            score_idx = find_score_token(tokens)
            parsed_score = int(tokens[score_idx]) if score_idx is not None else None

            rows.append({
                "sample_id": sid,
                "s2_lp1": s2[0], "s2_lp2": s2[1], "s2_lp3": s2[2],
                "s2_lp4": s2[3], "s2_lp5": s2[4],
                "gt_score": gt,
                "parsed_score": parsed_score,
            })
        except Exception as e:
            print(f"\nSkipping sample {sid}: {e}")
            skipped += 1

    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} samples ({skipped} skipped)")
    return df


def compute_point_metrics(df):
    """Compute point prediction metrics (judge vs human)."""
    from scipy.stats import pearsonr, spearmanr, kendalltau

    valid = df.dropna(subset=["parsed_score"])
    y_true = valid["gt_score"].values
    y_pred = valid["parsed_score"].values

    pearson_r, pearson_p = pearsonr(y_pred, y_true)
    spearman_r, spearman_p = spearmanr(y_pred, y_true)
    kendall_t, kendall_p = kendalltau(y_pred, y_true)
    accuracy = float(np.mean(y_pred == y_true))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    metrics = {
        "n_samples": len(valid),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "kendall_tau": kendall_t, "kendall_p": kendall_p,
        "accuracy": accuracy, "mae": mae, "rmse": rmse,
    }

    print(f"\n{'='*60}")
    print(f"Point Prediction Metrics ({len(valid)} samples)")
    print(f"{'='*60}")
    print(f"  Pearson r:    {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r:   {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Kendall tau:  {kendall_t:.4f} (p={kendall_p:.2e})")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  RMSE:         {rmse:.4f}")

    return metrics


def run_conformal(df, method, alpha, num_seeds, test_size, output_dir, score_range=(1, 5)):
    """Run conformal prediction for a single method."""
    from src.conformal.runner import ConformalRunner

    feature_cols = ["s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5"]

    # Save temp CSV for ConformalRunner
    csv_path = os.path.join(output_dir, "features_s2.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)

    method_dir = os.path.join(output_dir, method, "S2_only")

    runner = ConformalRunner(
        feature_csv=csv_path,
        method=method,
        alpha=alpha,
        num_seeds=num_seeds,
        test_size=test_size,
        output_dir=method_dir,
        score_range=score_range,
    )

    if method == "OrdinalAPS":
        return runner.run(feature_cols=feature_cols)
    else:
        return runner.run(feature_cols=feature_cols)


def main():
    parser = argparse.ArgumentParser(description="Full S2 analysis pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=["R2CCP", "CQR", "OrdinalAPS"])
    args = parser.parse_args()

    exp = load_yaml(args.config)
    dataset_config = load_yaml(exp["dataset"]["config"])
    judge_config = load_yaml(exp["judge"]["config"])
    save_config = exp.get("save", {})
    conformal_config = exp.get("conformal", {})

    judge_name = judge_config["name"].replace("-", "_").replace(".", "")
    dataset_name = dataset_config["name"]

    # Resolve actor name for path
    actor_config_path = exp.get("actor", {}).get("config")
    if actor_config_path and os.path.exists(actor_config_path):
        actor_config = load_yaml(actor_config_path)
        actor_name = actor_config["name"].replace("-", "_").replace(".", "")
    else:
        actor_name = "dataset_answers"

    judge_output_dir = os.path.join(
        save_config.get("output_dir", "outputs"),
        "judge", judge_name, f"on_{actor_name}", dataset_name,
    )

    # Count samples
    n_json = len([f for f in os.listdir(judge_output_dir) if f.endswith(".json")])
    print(f"Found {n_json} judge output files in {judge_output_dir}")

    # Step 1: Extract S2
    print(f"\n{'='*60}")
    print("Step 1: Extracting Signal 2")
    print(f"{'='*60}")
    df = extract_all_s2(judge_output_dir, n_json)

    # Step 2: Point prediction metrics
    point_metrics = compute_point_metrics(df)

    # Step 3: Conformal prediction
    output_base = os.path.join("results", "full_dataset", f"{judge_name}_on_{actor_name}")
    os.makedirs(output_base, exist_ok=True)

    # Save features CSV
    csv_path = os.path.join(output_base, "features_s2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved features to {csv_path}")

    # Save point metrics
    with open(os.path.join(output_base, "point_metrics.json"), "w") as f:
        json.dump(point_metrics, f, indent=2)

    alpha = conformal_config.get("alpha", 0.10)
    num_seeds = conformal_config.get("num_seeds", 100)
    test_size = conformal_config.get("test_size", 0.5)

    all_results = {}
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Step 3: Conformal Prediction — {method}")
        print(f"{'='*60}")
        try:
            result = run_conformal(
                df, method, alpha, num_seeds, test_size,
                output_base, score_range=(1, 5)
            )
            all_results[method] = result
        except Exception as e:
            print(f"Method {method} failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — Full Dataset ({len(df)} samples)")
    print(f"{'='*60}")
    print(f"\nPoint Prediction:")
    print(f"  Pearson r: {point_metrics['pearson_r']:.4f}, Accuracy: {point_metrics['accuracy']:.4f}, MAE: {point_metrics['mae']:.4f}")
    print(f"\nConformal Prediction (alpha={alpha}, {num_seeds} seeds):")
    print(f"  {'Method':<15} {'Coverage':>15} {'Width':>15}")
    print(f"  {'-'*45}")
    for method, r in all_results.items():
        print(f"  {method:<15} {r['coverage_mean']:.4f} ± {r['coverage_std']:.4f}   {r['width_mean']:.4f} ± {r['width_std']:.4f}")

    # Save final summary
    summary = {
        "dataset": dataset_name,
        "judge": judge_name,
        "n_samples": len(df),
        "point_metrics": point_metrics,
        "conformal_results": {m: {k: v for k, v in r.items() if k not in ("coverage_all", "width_all")}
                              for m, r in all_results.items()},
    }
    with open(os.path.join(output_base, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_base}/final_summary.json")


if __name__ == "__main__":
    main()
