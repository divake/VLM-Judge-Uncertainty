#!/usr/bin/env python3
"""Run conformal prediction on extracted signal features.

Usage:
    python scripts/run_conformal.py --config configs/experiments/pilot_mllm_judge.yaml
    python scripts/run_conformal.py --config configs/experiments/pilot_mllm_judge.yaml --num-seeds 10
    python scripts/run_conformal.py --config configs/experiments/pilot_mllm_judge.yaml --ablation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.conformal.runner import ConformalRunner


def main():
    parser = argparse.ArgumentParser(description="Run conformal prediction")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Override number of seeds from config")
    parser.add_argument("--ablation", action="store_true",
                        help="Run signal ablation study")
    parser.add_argument("--feature-csv", type=str, default=None,
                        help="Override feature CSV path")
    parser.add_argument("--method", type=str, default=None,
                        choices=["R2CCP", "CQR", "OrdinalAPS"],
                        help="Override conformal method")
    parser.add_argument("--all-methods", action="store_true",
                        help="Run all supported methods")
    args = parser.parse_args()

    # Load configs
    exp_config = load_yaml(args.config)
    actor_config = load_yaml(exp_config["actor"]["config"])
    judge_config = load_yaml(exp_config["judge"]["config"])
    dataset_config = load_yaml(exp_config["dataset"]["config"])
    save_config = exp_config.get("save", {})
    cp_config = exp_config.get("conformal", {})

    # Determine paths
    actor_name = actor_config["name"].replace("-", "_").replace(".", "")
    judge_name = judge_config["name"].replace("-", "_").replace(".", "")
    dataset_name = dataset_config["name"]
    output_base = save_config.get("output_dir", "outputs")

    feature_csv = args.feature_csv or os.path.join(
        output_base, "signals",
        f"{actor_name}__{judge_name}__{dataset_name}.csv"
    )
    conformal_output_dir = os.path.join(
        output_base, "conformal",
        f"{actor_name}__{judge_name}__{dataset_name}"
    )

    num_seeds = args.num_seeds or cp_config.get("num_seeds", 100)

    print(f"[run_conformal] Feature CSV: {feature_csv}")
    print(f"[run_conformal] Output: {conformal_output_dir}")
    print(f"[run_conformal] Method: {cp_config.get('method', 'R2CCP')}")
    print(f"[run_conformal] Alpha: {cp_config.get('alpha', 0.10)}")
    print(f"[run_conformal] Seeds: {num_seeds}")

    # Determine score range from dataset config
    target_scale = dataset_config.get("target_scale", [1, 5])
    score_range = tuple(target_scale)

    method = args.method or cp_config.get("method", "R2CCP")
    methods = ConformalRunner.SUPPORTED_METHODS if args.all_methods else [method]

    for method in methods:
        method_output = os.path.join(conformal_output_dir, method)
        runner = ConformalRunner(
            feature_csv=feature_csv,
            method=method,
            alpha=cp_config.get("alpha", 0.10),
            num_seeds=num_seeds,
            test_size=cp_config.get("test_size", 0.5),
            output_dir=method_output,
            score_range=score_range,
        )

        if args.ablation:
            print(f"\n[run_conformal] Running ablation for {method}...")
            # OrdinalAPS only works with S2 (5-dim), skip ablation for it
            if method == "OrdinalAPS":
                s2_cols = ["s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5"]
                results = {"S2_only": runner.run(feature_cols=s2_cols)}
            else:
                results = runner.run_ablation()
            print(f"\n{'='*60}")
            print(f"ABLATION SUMMARY ({method})")
            print(f"{'='*60}")
            for name, summary in results.items():
                print(f"  {name:15s}: Coverage={summary['coverage_mean']:.4f}+/-{summary['coverage_std']:.4f}  "
                      f"Width={summary['width_mean']:.4f}+/-{summary['width_std']:.4f}")
        else:
            # For OrdinalAPS, use S2 columns only
            if method == "OrdinalAPS":
                s2_cols = ["s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5"]
                summary = runner.run(feature_cols=s2_cols)
            else:
                summary = runner.run()

    print("\n[run_conformal] Complete!")


if __name__ == "__main__":
    main()
