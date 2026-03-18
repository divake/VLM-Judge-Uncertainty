#!/usr/bin/env python3
"""Run judge VLM inference on dataset samples.

Usage:
    # Full dataset, single GPU:
    python scripts/run_judge.py --config configs/experiments/pilot_mllm_judge.yaml

    # Parallel on 2 GPUs (split dataset in half):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_judge.py --config configs/experiments/pilot_mllm_judge.yaml --start-idx 0 --end-idx 2860 --gpu cuda:0 &
    CUDA_VISIBLE_DEVICES=1 python scripts/run_judge.py --config configs/experiments/pilot_mllm_judge.yaml --start-idx 2860 --end-idx 5719 --gpu cuda:0 &
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.data.mllm_judge import MLLMJudgeDataset
from src.models.llava import LLaVAModel
from src.inference.judge_runner import JudgeRunner


def main():
    parser = argparse.ArgumentParser(description="Run judge VLM inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of samples to process")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting sample index")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Ending sample index (exclusive)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device (e.g., cuda:0). Overrides config.")
    parser.add_argument("--use-actor-answers", action="store_true",
                        help="Use answers from actor output files instead of dataset")
    args = parser.parse_args()

    # Load configs
    exp_config = load_yaml(args.config)
    judge_config = load_yaml(exp_config["judge"]["config"])
    dataset_config = load_yaml(exp_config["dataset"]["config"])
    prompt_config = load_yaml(exp_config["judge"]["prompt"])
    save_config = exp_config.get("save", {})

    # Determine paths
    judge_name = judge_config["name"].replace("-", "_").replace(".", "")
    dataset_name = dataset_config["name"]

    # For dataset-answer mode, we still use the actor name in the path for organization
    actor_config_path = exp_config.get("actor", {}).get("config")
    if actor_config_path and os.path.exists(actor_config_path):
        actor_config = load_yaml(actor_config_path)
        actor_name = actor_config["name"].replace("-", "_").replace(".", "")
    else:
        actor_name = "dataset_answers"

    actor_output_dir = os.path.join(
        save_config.get("output_dir", "outputs"),
        "actor", actor_name, dataset_name,
    )
    judge_output_dir = os.path.join(
        save_config.get("output_dir", "outputs"),
        "judge", judge_name,
        f"on_{actor_name}", dataset_name,
    )

    # GPU selection
    gpu = args.gpu or exp_config["judge"].get("gpu", "cuda:0")

    print(f"[run_judge] Experiment: {exp_config['experiment_name']}")
    print(f"[run_judge] Judge: {judge_config['name']}")
    print(f"[run_judge] GPU: {gpu}")
    print(f"[run_judge] Output: {judge_output_dir}")
    print(f"[run_judge] save_full_logits: {save_config.get('save_full_logits', False)}")

    # Load dataset
    dataset = MLLMJudgeDataset(config=dataset_config)
    print(f"[run_judge] Loaded {len(dataset)} samples")

    # Apply save_full_scores to model config
    judge_config["save_full_scores"] = save_config.get("save_full_logits", False)

    # Load model
    model = LLaVAModel(config=judge_config)
    model.load_model(device=gpu)

    # Get prompt template
    prompt_template = prompt_config["template"]
    print(f"[run_judge] Prompt template:\n{prompt_template[:200]}...")

    # Determine answer source
    use_dataset_answers = not args.use_actor_answers
    print(f"[run_judge] Answer source: {'dataset' if use_dataset_answers else 'actor outputs'}")

    # Run inference
    runner = JudgeRunner(
        model=model,
        dataset=dataset,
        actor_output_dir=actor_output_dir,
        output_dir=judge_output_dir,
        prompt_template=prompt_template,
        save_full_logits=save_config.get("save_full_logits", False),
        use_dataset_answers=use_dataset_answers,
    )
    runner.run(start_idx=args.start_idx, end_idx=args.end_idx, limit=args.limit)

    # Cleanup
    model.unload_model()
    print("[run_judge] Complete!")


if __name__ == "__main__":
    main()
