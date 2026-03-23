#!/usr/bin/env python3
"""Lean judge VLM inference: generates scores and extracts S2 logprobs in one pass.

Outputs a single CSV with columns:
    sample_id, s2_lp1, s2_lp2, s2_lp3, s2_lp4, s2_lp5, gt_score, parsed_score

No intermediate JSON/PT files. Much faster than the full pipeline.

Usage:
    # Single GPU (full dataset):
    python scripts/run_judge_lean.py --output results/v2/features_s2.csv

    # Two GPUs in parallel:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_judge_lean.py --start-idx 0 --end-idx 2860 --output results/v2/features_s2_gpu0.csv &
    CUDA_VISIBLE_DEVICES=1 python scripts/run_judge_lean.py --start-idx 2860 --end-idx 5719 --output results/v2/features_s2_gpu1.csv &
    # Then merge: head -1 gpu0.csv > features_s2.csv && tail -n+2 gpu0.csv >> features_s2.csv && tail -n+2 gpu1.csv >> features_s2.csv
"""

import argparse
import copy
import csv
import gc
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.data.mllm_judge import MLLMJudgeDataset

SCORE_TOKENS = ["1", "2", "3", "4", "5"]
FALLBACK_LP = math.log(1e-5)  # ~-11.51


def load_model(model_config, device="cuda:0"):
    """Load LLaVA-Critic model. Returns (model, tokenizer, image_processor)."""
    from transformers import AutoTokenizer, AutoConfig, Qwen2Config
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

    model_path = model_config["hf_id"]
    # Try local path first
    for local in [f"models/{model_config['name']}", f"models/{model_path.split('/')[-1]}"]:
        if os.path.exists(local):
            model_path = local
            break

    print(f"[load_model] Loading from: {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, 'text_config') and isinstance(config.text_config, dict):
        config.text_config = Qwen2Config(**config.text_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = LlavaQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        device_map="auto",
        config=config,
    )
    model.eval()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    return model, tokenizer, image_processor


def run_inference(model, tokenizer, image_processor, image_path, prompt,
                  max_new_tokens=1024):
    """Run VLM inference and return (generated_text, scores, sequences, input_length)."""
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_img.to(dtype=torch.float16, device=model.device)
                    for _img in image_tensor]

    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.no_grad():
        generation = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            repetition_penalty=1.1,
        )

    sequences = generation.sequences
    scores = generation.scores
    input_length = sequences.shape[1] - len(scores)

    generated_ids = sequences[0, input_length:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, scores, sequences, input_length


def extract_s2(scores, sequences, input_length, tokenizer):
    """Extract S2 logprobs (5-dim) and parsed score from generation output.

    Returns: (s2_logprobs, parsed_score) where s2_logprobs is [lp1, lp2, lp3, lp4, lp5]
    """
    # Decode tokens and compute log_softmax only at each step to find score token
    n_steps = len(scores)
    tokens = []
    for step_idx in range(n_steps):
        token_id = sequences[0, input_length + step_idx].item()
        token_str = tokenizer.decode([token_id]).strip()
        tokens.append(token_str)

    # Find score token index
    score_idx = _find_score_token(tokens)

    if score_idx is None:
        return [FALLBACK_LP] * 5, -1

    # Extract logprobs ONLY at the score token position
    step_scores = scores[score_idx]  # shape: [1, vocab_size]
    log_probs = F.log_softmax(step_scores.float(), dim=-1)

    # Get logprobs for score tokens "1" through "5"
    s2 = []
    for s in SCORE_TOKENS:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        if len(token_ids) == 1:
            lp = log_probs[0, token_ids[0]].item()
            if not (lp == lp):  # NaN check
                lp = FALLBACK_LP
            s2.append(lp)
        else:
            s2.append(FALLBACK_LP)

    # Parse the actual score
    parsed_score = int(tokens[score_idx]) if tokens[score_idx] in SCORE_TOKENS else -1

    return s2, parsed_score


def _find_score_token(tokens):
    """Find the score digit token index.

    Priority:
        1. "Score" + ":" + digit pattern (backward scan)
        2. "score" keyword then look ahead for digit
        3. Plain backward scan for last digit in {1,2,3,4,5}
    """
    stripped = [t.strip().lower() for t in tokens]

    # Strategy 1: Find "score" ... digit pattern (backward)
    for i in range(len(tokens) - 1, -1, -1):
        if stripped[i] in SCORE_TOKENS:
            for j in range(max(0, i - 3), i):
                if "score" in stripped[j]:
                    return i

    # Strategy 2: Find "score" keyword then look ahead
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


def main():
    parser = argparse.ArgumentParser(description="Lean judge inference → CSV")
    parser.add_argument("--config", type=str,
                        default="configs/experiments/pilot_mllm_judge.yaml")
    parser.add_argument("--prompt", type=str,
                        default="configs/prompts/mllm_judge_cot.yaml",
                        help="Prompt config YAML")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    # Load configs
    exp_config = load_yaml(args.config)
    dataset_config = load_yaml(exp_config["dataset"]["config"])
    model_config = load_yaml(exp_config["judge"]["config"])
    prompt_config = load_yaml(args.prompt)
    prompt_template = prompt_config["template"]

    # Load dataset
    dataset = MLLMJudgeDataset(config=dataset_config)
    print(f"[main] Dataset: {len(dataset)} samples")

    # Filter out invalid GT scores (score=0)
    valid_indices = [i for i in range(len(dataset)) if dataset.get_gt_score(i) >= 1]
    print(f"[main] Valid samples (GT >= 1): {len(valid_indices)}")

    # Apply index range
    end_idx = args.end_idx or len(valid_indices)
    indices = valid_indices[args.start_idx:end_idx]
    print(f"[main] Processing indices {args.start_idx} to {end_idx} ({len(indices)} samples)")

    # Load model
    model, tokenizer, image_processor = load_model(model_config)
    print(f"[main] Model loaded. Prompt:\n{prompt_template[:200]}...")

    # Prepare output CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    csv_file = open(args.output, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["sample_id", "s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5",
                      "gt_score", "parsed_score"])

    # Run inference
    errors = 0
    pbar = tqdm(indices, desc="Judge inference (lean)")

    for idx in pbar:
        try:
            image_path = dataset.get_image_path(idx)
            question = dataset.get_question(idx)
            actor_answer = dataset.get_answer(idx)
            gt_score = dataset.get_gt_score(idx)

            prompt = prompt_template.format(
                question=question,
                actor_answer=actor_answer,
            )

            generated_text, scores, sequences, input_length = run_inference(
                model, tokenizer, image_processor, image_path, prompt,
                max_new_tokens=args.max_new_tokens,
            )

            s2, parsed_score = extract_s2(scores, sequences, input_length, tokenizer)

            writer.writerow([idx, s2[0], s2[1], s2[2], s2[3], s2[4],
                             gt_score, parsed_score])
            csv_file.flush()  # flush after each row for crash safety

            # Clean up GPU memory
            del scores, sequences
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] Sample {idx}: {e}")
            errors += 1
            # Write fallback row so we don't lose the sample
            writer.writerow([idx, FALLBACK_LP, FALLBACK_LP, FALLBACK_LP,
                             FALLBACK_LP, FALLBACK_LP, gt_score, -1])
            csv_file.flush()

        pbar.set_postfix(err=errors)

    csv_file.close()
    print(f"\n[main] Done! {len(indices) - errors} succeeded, {errors} errors")
    print(f"[main] Output: {args.output}")

    # Cleanup
    del model, tokenizer, image_processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
