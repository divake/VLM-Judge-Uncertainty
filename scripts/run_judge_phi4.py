#!/usr/bin/env python3
"""Lean judge inference for Phi-4-reasoning-vision-15B.

Outputs same CSV format as run_judge_lean.py:
    sample_id, s2_lp1, s2_lp2, s2_lp3, s2_lp4, s2_lp5, gt_score, parsed_score

Usage:
    # Single GPU:
    CUDA_VISIBLE_DEVICES=0 conda run -n phi4_env python scripts/run_judge_phi4.py --output results/v2_phi4/features_s2.csv

    # Two GPUs:
    CUDA_VISIBLE_DEVICES=0 conda run -n phi4_env python scripts/run_judge_phi4.py --start-idx 0 --end-idx 2859 --output results/v2_phi4/features_s2_p0.csv &
    CUDA_VISIBLE_DEVICES=1 conda run -n phi4_env python scripts/run_judge_phi4.py --start-idx 2859 --end-idx 5717 --output results/v2_phi4/features_s2_p1.csv &
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SCORE_TOKENS = ["1", "2", "3", "4", "5"]
FALLBACK_LP = math.log(1e-5)

# Same CoT prompt as LLaVA-Critic
PROMPT_TEMPLATE = """Please serve as an unbiased judge in assessing the quality of the response from an AI assistant regarding the user's instruction and the provided image.

Evaluation Steps:
Please examine the provided image attentively. Begin by conducting a comprehensive analysis of the figure provided. Then, utilize the insights from your analysis to critically evaluate the response. Finally, based on your figure analysis and response evaluation, form a well-reasoned judgement.

Scoring Rubric:
Poor (1): The response significantly deviates from the user's instruction and fails to address the query effectively. It shows a lack of relevance, accuracy, and comprehensiveness. Creativity and granularity are absent or poorly executed.
Fair (2): The response addresses the user's instruction partially, with evident shortcomings in relevance, accuracy, or comprehensiveness. It lacks depth in creativity and granularity, indicating a superficial understanding of the user's inquiry.
Average (3): The response adequately addresses the user's instruction, showing a fair level of relevance, accuracy, and comprehensiveness. It reflects a basic level of creativity and granularity but may lack sophistication or depth in fully capturing the user's inquiry.
Good (4): The response is well-aligned with the user's instruction, demonstrating a high degree of relevance, accuracy, and comprehensiveness. It shows creativity and a nuanced understanding of the topic, with detailed granularity that enhances the response quality.
Excellent (5): The response perfectly adheres to the user's instruction, excelling in relevance, accuracy, comprehensiveness, creativity, and granularity. It provides an insightful, detailed, and thorough answer, indicating a deep and nuanced understanding of the user's inquiry.

Notice:
In your evaluation, weigh factors such as relevance, accuracy, comprehensiveness, creativity, and the granularity of the response. Do not allow the length of the response to influence your evaluation. Be as objective as possible.

Here is the input:
[The Start of User Instruction]
{question}
[The End of User Instruction]
[The Start of Assistant's Answer]
{actor_answer}
[The End of Assistant's Answer]

First, provide your analysis of the image and the response. Then, at the very end of your response, provide your final score in exactly this format on its own line: "Score: X" where X is a single number from 1 to 5."""


def load_model(model_path, device="cuda:0"):
    """Load Phi-4-reasoning-vision-15B."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[load_model] Loading Phi-4-reasoning-vision from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[load_model] Loaded successfully")
    return model, processor


def run_inference(model, processor, image_path, prompt, max_new_tokens=1024):
    """Run Phi-4 inference with image."""
    image = Image.open(image_path).convert("RGB")

    # Phi-4 uses <|image|> tag inline in text content
    messages = [
        {"role": "user", "content": f"<|image|>\n{prompt}"},
    ]

    # Process inputs
    inputs = processor(
        text=processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        images=image,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    sequences = generation.sequences
    scores = generation.scores
    actual_input_length = sequences.shape[1] - len(scores)

    generated_ids = sequences[0, actual_input_length:].tolist()
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, scores, sequences, actual_input_length


def extract_s2(scores, sequences, input_length, tokenizer):
    """Extract S2 logprobs and parsed score."""
    n_steps = len(scores)
    tokens = []
    for step_idx in range(n_steps):
        token_id = sequences[0, input_length + step_idx].item()
        token_str = tokenizer.decode([token_id]).strip()
        tokens.append(token_str)

    score_idx = _find_score_token(tokens)
    if score_idx is None:
        return [FALLBACK_LP] * 5, -1

    step_scores = scores[score_idx]
    log_probs = F.log_softmax(step_scores.float(), dim=-1)

    s2 = []
    for s in SCORE_TOKENS:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        if len(token_ids) >= 1:
            lp = log_probs[0, token_ids[0]].item()
            if not (lp == lp):
                lp = FALLBACK_LP
            s2.append(lp)
        else:
            s2.append(FALLBACK_LP)

    parsed_score = int(tokens[score_idx]) if tokens[score_idx] in SCORE_TOKENS else -1
    return s2, parsed_score


def _find_score_token(tokens):
    """Find score digit token index."""
    stripped = [t.strip().lower() for t in tokens]

    for i in range(len(tokens) - 1, -1, -1):
        if stripped[i] in SCORE_TOKENS:
            for j in range(max(0, i - 3), i):
                if "score" in stripped[j]:
                    return i

    for i in range(len(tokens) - 1, -1, -1):
        if "score" in stripped[i]:
            for j in range(i + 1, min(len(tokens), i + 5)):
                if stripped[j] in SCORE_TOKENS:
                    return j

    for i in range(len(tokens) - 1, -1, -1):
        if stripped[i] in SCORE_TOKENS:
            return i

    return None


def load_dataset():
    """Load MLLM-Judge dataset."""
    samples = []
    with open('data/mllm_judge/raw/Dataset/Benchmark/score.jsonl') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            gt = float(d['human'])
            if gt < 1:
                continue
            samples.append({
                'idx': i,
                'image_path': f"data/mllm_judge/raw/Dataset/image/{d['image_path']}",
                'question': d['instruction'],
                'answer': d['answer'],
                'gt_score': gt,
            })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Phi-4 judge inference → CSV")
    parser.add_argument("--model-path", type=str,
                        default="models/Phi-4-reasoning-vision-15B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    # Load dataset
    samples = load_dataset()
    print(f"[main] Dataset: {len(samples)} valid samples")

    end_idx = args.end_idx or len(samples)
    subset = samples[args.start_idx:end_idx]
    print(f"[main] Processing {len(subset)} samples (idx {args.start_idx} to {end_idx})")

    # Load model
    model, processor = load_model(args.model_path)

    # Prepare output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    csv_file = open(args.output, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["sample_id", "s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5",
                      "gt_score", "parsed_score"])

    errors = 0
    pbar = tqdm(subset, desc="Phi-4 judge inference")

    for sample in pbar:
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample['question'],
                actor_answer=sample['answer'],
            )

            generated_text, scores, sequences, input_length = run_inference(
                model, processor, sample['image_path'], prompt,
                max_new_tokens=args.max_new_tokens,
            )

            s2, parsed_score = extract_s2(scores, sequences, input_length,
                                          processor.tokenizer)

            writer.writerow([sample['idx'], s2[0], s2[1], s2[2], s2[3], s2[4],
                             sample['gt_score'], parsed_score])
            csv_file.flush()

            del scores, sequences
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] Sample {sample['idx']}: {e}")
            errors += 1
            writer.writerow([sample['idx'], FALLBACK_LP, FALLBACK_LP, FALLBACK_LP,
                             FALLBACK_LP, FALLBACK_LP, sample['gt_score'], -1])
            csv_file.flush()

        pbar.set_postfix(err=errors)

    csv_file.close()
    print(f"\n[main] Done! {len(subset) - errors} succeeded, {errors} errors")
    print(f"[main] Output: {args.output}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
