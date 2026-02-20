#!/usr/bin/env python3
"""Run actor VLM inference on a dataset.

Usage:
    python scripts/run_actor.py --config configs/experiments/pilot_mllm_judge.yaml
    python scripts/run_actor.py --config configs/experiments/pilot_mllm_judge.yaml --limit 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.data.mllm_judge import MLLMJudgeDataset
from src.models.llava import LLaVAModel
from src.inference.actor_runner import ActorRunner


def main():
    parser = argparse.ArgumentParser(description="Run actor VLM inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of samples to process")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting sample index")
    args = parser.parse_args()

    # Load configs
    exp_config = load_yaml(args.config)
    actor_config = load_yaml(exp_config["actor"]["config"])
    dataset_config = load_yaml(exp_config["dataset"]["config"])
    save_config = exp_config.get("save", {})

    # Determine output directory
    actor_name = actor_config["name"].replace("-", "_").replace(".", "")
    dataset_name = dataset_config["name"]
    output_dir = os.path.join(
        save_config.get("output_dir", "outputs"),
        "actor",
        actor_name,
        dataset_name,
    )

    print(f"[run_actor] Experiment: {exp_config['experiment_name']}")
    print(f"[run_actor] Actor: {actor_config['name']}")
    print(f"[run_actor] Dataset: {dataset_name}")
    print(f"[run_actor] Output: {output_dir}")

    # Load dataset
    dataset = MLLMJudgeDataset(config=dataset_config)
    print(f"[run_actor] Loaded {len(dataset)} samples")

    # Load model
    model = LLaVAModel(config=actor_config)
    model.load_model(device=exp_config["actor"].get("gpu", "cuda:0"))

    # Run inference
    runner = ActorRunner(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        save_full_logits=save_config.get("save_full_logits", True),
    )
    runner.run(start_idx=args.start_idx, limit=args.limit)

    # Cleanup
    model.unload_model()
    print("[run_actor] Complete!")


if __name__ == "__main__":
    main()
