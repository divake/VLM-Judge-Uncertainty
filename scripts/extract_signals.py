#!/usr/bin/env python3
"""Extract 3-signal feature vectors from actor + judge outputs.

Usage:
    python scripts/extract_signals.py --config configs/experiments/pilot_mllm_judge.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from src.utils.io import load_yaml, load_json
from src.data.mllm_judge import MLLMJudgeDataset
from src.signals.extractor import SignalExtractor, SCORE_TOKENS
from src.evaluation.metrics import compute_point_prediction_metrics


def main():
    parser = argparse.ArgumentParser(description="Extract 3-signal features")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    # Load configs
    exp_config = load_yaml(args.config)
    actor_config = load_yaml(exp_config["actor"]["config"])
    judge_config = load_yaml(exp_config["judge"]["config"])
    dataset_config = load_yaml(exp_config["dataset"]["config"])
    save_config = exp_config.get("save", {})

    # Determine paths
    actor_name = actor_config["name"].replace("-", "_").replace(".", "")
    judge_name = judge_config["name"].replace("-", "_").replace(".", "")
    dataset_name = dataset_config["name"]
    output_base = save_config.get("output_dir", "outputs")

    actor_output_dir = os.path.join(output_base, "actor", actor_name, dataset_name)
    judge_output_dir = os.path.join(output_base, "judge", judge_name,
                                     f"on_{actor_name}", dataset_name)
    output_csv = os.path.join(output_base, "signals",
                               f"{actor_name}__{judge_name}__{dataset_name}.csv")

    print(f"[extract_signals] Actor outputs: {actor_output_dir}")
    print(f"[extract_signals] Judge outputs: {judge_output_dir}")
    print(f"[extract_signals] Output CSV: {output_csv}")

    # Load dataset for GT scores
    try:
        dataset = MLLMJudgeDataset(config=dataset_config)
    except FileNotFoundError:
        print("[extract_signals] Dataset not loaded, will use GT from saved JSONs")
        dataset = None

    # Extract signals
    extractor = SignalExtractor(
        actor_output_dir=actor_output_dir,
        judge_output_dir=judge_output_dir,
        dataset=dataset,
    )
    df = extractor.extract_all(output_csv=output_csv)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Feature matrix shape: {df.shape}")
    print(f"NaN count: {df.isna().sum().sum()}")
    print(f"\nFeature statistics:")
    print(df.describe().to_string())

    # --- Point prediction metrics: judge scores vs GT ---
    print(f"\n{'='*60}")
    print("POINT PREDICTION METRICS (Judge Score vs Human GT)")
    print(f"{'='*60}")

    # Extract predicted scores from judge JSON files
    judge_scores = []
    gt_scores = []
    for sid in df["sample_id"].astype(int):
        jpath = os.path.join(judge_output_dir, f"sample_{sid:04d}.json")
        if os.path.exists(jpath):
            jdata = load_json(jpath)
            tokens = jdata["tokens"]
            score_idx = SignalExtractor._find_score_token(tokens)
            if score_idx is not None:
                judge_scores.append(float(tokens[score_idx].strip()))
            else:
                judge_scores.append(np.nan)
            gt_scores.append(float(jdata["gt_score"]))

    judge_scores = np.array(judge_scores)
    gt_scores = np.array(gt_scores)

    metrics = compute_point_prediction_metrics(judge_scores, gt_scores)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")

    # Save metrics
    metrics_path = os.path.join(output_base, "signals",
                                f"{actor_name}__{judge_name}__{dataset_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {metrics_path}")


if __name__ == "__main__":
    main()
