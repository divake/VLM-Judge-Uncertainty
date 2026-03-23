#!/usr/bin/env python3
"""Quick test: run lean judge on 3 samples, print everything for inspection."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.data.mllm_judge import MLLMJudgeDataset

# Import the lean runner functions
from scripts.run_judge_lean import load_model, run_inference, extract_s2

import torch

def main():
    # Load configs
    exp_config = load_yaml("configs/experiments/pilot_mllm_judge.yaml")
    dataset_config = load_yaml(exp_config["dataset"]["config"])
    model_config = load_yaml(exp_config["judge"]["config"])
    prompt_config = load_yaml("configs/prompts/mllm_judge_cot.yaml")
    prompt_template = prompt_config["template"]

    # Load dataset
    dataset = MLLMJudgeDataset(config=dataset_config)

    # Load model (timed)
    print("=" * 70)
    print("LOADING MODEL...")
    print("=" * 70)
    t0 = time.time()
    model, tokenizer, image_processor = load_model(model_config)
    load_time = time.time() - t0
    print(f"\nModel load time: {load_time:.1f}s")

    # Test on 3 samples
    test_indices = [0, 100, 500]

    for idx in test_indices:
        gt_score = dataset.get_gt_score(idx)
        if gt_score < 1:
            print(f"\nSkipping sample {idx}: GT={gt_score} (invalid)")
            continue

        image_path = dataset.get_image_path(idx)
        question = dataset.get_question(idx)
        actor_answer = dataset.get_answer(idx)

        prompt = prompt_template.format(
            question=question,
            actor_answer=actor_answer,
        )

        print("\n" + "=" * 70)
        print(f"SAMPLE {idx}")
        print("=" * 70)
        print(f"Image: {image_path}")
        print(f"Question: {question[:100]}...")
        print(f"Actor answer: {actor_answer[:100]}...")
        print(f"GT score: {gt_score}")

        # Run inference (timed)
        t1 = time.time()
        generated_text, scores, sequences, input_length = run_inference(
            model, tokenizer, image_processor, image_path, prompt,
            max_new_tokens=1024,
        )
        inference_time = time.time() - t1

        # Extract S2 (timed)
        t2 = time.time()
        s2, parsed_score = extract_s2(scores, sequences, input_length, tokenizer)
        extract_time = time.time() - t2

        n_tokens = len(scores)
        tokens_per_sec = n_tokens / inference_time if inference_time > 0 else 0

        print(f"\n--- Generated text ({n_tokens} tokens) ---")
        print(generated_text)
        print(f"\n--- S2 logprobs ---")
        print(f"  lp('1') = {s2[0]:.4f}")
        print(f"  lp('2') = {s2[1]:.4f}")
        print(f"  lp('3') = {s2[2]:.4f}")
        print(f"  lp('4') = {s2[3]:.4f}")
        print(f"  lp('5') = {s2[4]:.4f}")
        print(f"  Parsed score: {parsed_score}")
        print(f"  GT score: {gt_score}")
        print(f"  Match: {'YES' if parsed_score == gt_score else 'NO'}")

        # Softmax for readability
        import numpy as np
        probs = np.exp(s2)
        probs = probs / probs.sum()
        print(f"\n--- Score probabilities (softmax) ---")
        for i, p in enumerate(probs, 1):
            bar = "#" * int(p * 50)
            print(f"  P(score={i}) = {p:.4f}  {bar}")

        print(f"\n--- Timing ---")
        print(f"  Inference: {inference_time:.2f}s ({n_tokens} tokens, {tokens_per_sec:.1f} tok/s)")
        print(f"  S2 extraction: {extract_time*1000:.1f}ms")
        print(f"  Total per sample: {inference_time + extract_time:.2f}s")

        # Cleanup
        del scores, sequences
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("CSV ROW FORMAT (what the lean runner outputs):")
    print("=" * 70)
    print("sample_id, s2_lp1, s2_lp2, s2_lp3, s2_lp4, s2_lp5, gt_score, parsed_score")

    # Cleanup
    del model, tokenizer, image_processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
