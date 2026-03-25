#!/usr/bin/env python3
"""Lean judge inference for Gemini via Vertex AI + LiteLLM.

Requires:
    - gcloud auth application-default login (done once)
    - export VERTEXAI_PROJECT="gen-lang-client-0896152357"
    - export VERTEXAI_LOCATION="us-central1"

Outputs same CSV format:
    sample_id, s2_lp1, s2_lp2, s2_lp3, s2_lp4, s2_lp5, gt_score, parsed_score

Usage:
    conda run -n env_py311 python scripts/run_judge_gemini.py \
        --model vertex_ai/gemini-2.5-flash \
        --output results/v2_gemini/features_s2.csv
"""

import argparse
import base64
import csv
import io
import json
import math
import os
import time

from PIL import Image
from tqdm import tqdm

SCORE_TOKENS = ["1", "2", "3", "4", "5"]
FALLBACK_LP = math.log(1e-5)

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


def encode_image(image_path):
    """Encode image to base64 for API."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(model, image_b64, prompt, max_tokens=2048, max_retries=3):
    """Call Gemini API with retries."""
    from litellm import completion

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                temperature=0,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=20,
                num_retries=3,
            )
            return response
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)
                print(f"\n  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def extract_s2(response):
    """Extract S2 logprobs and parsed score from Gemini response."""
    lp = response.choices[0].logprobs

    if not lp or not hasattr(lp, 'content') or not lp.content:
        return [FALLBACK_LP] * 5, -1

    tokens = [t.token.strip() for t in lp.content]
    stripped = [t.lower() for t in tokens]

    # Find score token
    score_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        if stripped[i] in SCORE_TOKENS:
            for j in range(max(0, i - 3), i):
                if "score" in stripped[j]:
                    score_idx = i
                    break
            if score_idx:
                break
    if score_idx is None:
        for i in range(len(tokens) - 1, -1, -1):
            if stripped[i] in SCORE_TOKENS:
                score_idx = i
                break

    if score_idx is None:
        return [FALLBACK_LP] * 5, -1

    # Extract logprobs for score tokens from top_logprobs
    tok_data = lp.content[score_idx]
    lp_dict = {}
    if hasattr(tok_data, 'top_logprobs') and tok_data.top_logprobs:
        for alt in tok_data.top_logprobs:
            if alt.token.strip() in SCORE_TOKENS:
                lp_dict[alt.token.strip()] = alt.logprob

    s2 = [lp_dict.get(s, FALLBACK_LP) for s in SCORE_TOKENS]
    parsed_score = int(tokens[score_idx]) if tokens[score_idx] in SCORE_TOKENS else -1

    return s2, parsed_score


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
    parser = argparse.ArgumentParser(description="Gemini judge inference via Vertex AI")
    parser.add_argument("--model", type=str, default="vertex_ai/gemini-2.5-flash")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between requests (rate limit)")
    args = parser.parse_args()

    samples = load_dataset()
    print(f"[main] Dataset: {len(samples)} valid samples")

    end_idx = args.end_idx or len(samples)
    subset = samples[args.start_idx:end_idx]
    print(f"[main] Processing {len(subset)} samples (idx {args.start_idx} to {end_idx})")
    print(f"[main] Model: {args.model}")
    print(f"[main] Delay between requests: {args.delay}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    csv_file = open(args.output, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["sample_id", "s2_lp1", "s2_lp2", "s2_lp3", "s2_lp4", "s2_lp5",
                      "gt_score", "parsed_score"])

    # Second CSV for full responses
    resp_path = args.output.replace(".csv", "_responses.csv")
    resp_file = open(resp_path, "w", newline="", encoding="utf-8")
    resp_writer = csv.writer(resp_file)
    resp_writer.writerow(["sample_id", "gt_score", "parsed_score", "question", "actor_answer", "judge_response"])

    errors = 0
    pbar = tqdm(subset, desc=f"Gemini judge ({args.model.split('/')[-1]})")

    for sample in pbar:
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=sample['question'],
                actor_answer=sample['answer'],
            )
            image_b64 = encode_image(sample['image_path'])

            response = run_inference(args.model, image_b64, prompt,
                                     max_tokens=args.max_new_tokens)

            s2, parsed_score = extract_s2(response)
            judge_text = response.choices[0].message.content or ""

            writer.writerow([sample['idx'], s2[0], s2[1], s2[2], s2[3], s2[4],
                             sample['gt_score'], parsed_score])
            csv_file.flush()

            resp_writer.writerow([sample['idx'], sample['gt_score'], parsed_score,
                                  sample['question'][:500], sample['answer'][:500],
                                  judge_text])
            resp_file.flush()

            time.sleep(args.delay)

        except Exception as e:
            print(f"\n[ERROR] Sample {sample['idx']}: {e}")
            errors += 1
            writer.writerow([sample['idx'], FALLBACK_LP, FALLBACK_LP, FALLBACK_LP,
                             FALLBACK_LP, FALLBACK_LP, sample['gt_score'], -1])
            csv_file.flush()
            time.sleep(5)  # Extra wait on error

        pbar.set_postfix(err=errors)

    csv_file.close()
    resp_file.close()
    print(f"\n[main] Done! {len(subset) - errors} succeeded, {errors} errors")
    print(f"[main] Output: {args.output}")
    print(f"[main] Responses: {resp_path}")


if __name__ == "__main__":
    main()
