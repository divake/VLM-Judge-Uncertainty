#!/usr/bin/env python3
"""SMOKE TEST: LLaVA-Critic-7B on Multimodal RewardBench (pairwise scoring).

Scores each of the two candidate responses (Output1, Output2) on the 1-5 rubric,
extracts the 5-dim score-token logprobs (Signal 2), derives the judge's preferred
response (higher parsed score), and compares to the human 'Better' label.

Purpose: verify (a) the model loads on this env, (b) score parsing + logprob
extraction work on RewardBench, (c) predicted-vs-human preference is sane, and
(d) measure per-call latency for sharding math. NOT the full run.

Usage:
    conda run -n env_py311 python scripts/smoke_rewardbench_llava.py --n 4 --gpu cuda:0
"""
import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llava import LLaVAModel

DATA_ROOT = "data/multimodal_rewardbench/raw/data"
ALL_DATA = os.path.join(DATA_ROOT, "all_data.json")
SCORE_TOKENS = ["1", "2", "3", "4", "5"]
FALLBACK_LP = math.log(1e-5)

# Same rubric prompt used across the project (MLLM-Judge CoT scoring).
PROMPT = """Please serve as an unbiased judge in assessing the quality of the response from an AI assistant regarding the user's instruction and the provided image.

Evaluation Steps:
Please examine the provided image attentively. Begin by conducting a comprehensive analysis of the figure provided. Then, utilize the insights from your analysis to critically evaluate the response. Finally, based on your figure analysis and response evaluation, form a well-reasoned judgement.

Scoring Rubric:
Poor (1): The response significantly deviates from the user's instruction and fails to address the query effectively.
Fair (2): The response addresses the user's instruction partially, with evident shortcomings in relevance, accuracy, or comprehensiveness.
Average (3): The response adequately addresses the user's instruction, showing a fair level of relevance, accuracy, and comprehensiveness.
Good (4): The response is well-aligned with the user's instruction, demonstrating a high degree of relevance, accuracy, and comprehensiveness.
Excellent (5): The response perfectly adheres to the user's instruction, excelling in relevance, accuracy, comprehensiveness, creativity, and granularity.

[The Start of User Instruction] {question} [The End of User Instruction]
[The Start of Assistant's Answer] {answer} [The End of Assistant's Answer]

First, provide your analysis. Then, at the very end of your response, provide your final score in exactly this format on its own line: "Score: X" where X is a single integer from 1 to 5."""


def extract_score_and_logprobs(gen):
    """Find the score token (Score: anchor, else backward scan) and pull the
    5-dim score-token logprobs at that position. Mirrors the project pipeline."""
    tokens = gen.tokens
    # Strategy 1: 'Score:' anchor -> first digit after it
    score_idx = None
    for i, tok in enumerate(tokens):
        if "score" in tok.lower():
            for j in range(i, min(i + 6, len(tokens))):
                if tokens[j].strip() in SCORE_TOKENS:
                    score_idx = j
                    break
        if score_idx is not None:
            break
    # Strategy 3: backward scan for last digit 1-5
    if score_idx is None:
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].strip() in SCORE_TOKENS:
                score_idx = i
                break
    if score_idx is None:
        return None, [FALLBACK_LP] * 5
    parsed = int(tokens[score_idx].strip())
    lp_dict = gen.top_logprobs[score_idx]
    s2 = [lp_dict.get(s, FALLBACK_LP) for s in SCORE_TOKENS]
    return parsed, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of triplets to test")
    ap.add_argument("--gpu", type=str, default="cuda:0")
    args = ap.parse_args()

    data = json.load(open(ALL_DATA))
    # pick n triplets spread across categories
    seen, picks = set(), []
    for ex in data:
        c = ex.get("Category", "?")
        if c not in seen:
            seen.add(c); picks.append(ex)
        if len(picks) >= args.n:
            break
    if len(picks) < args.n:
        picks += data[:args.n - len(picks)]

    print(f"Loading LLaVA-Critic-7B on {args.gpu} ...")
    model = LLaVAModel(config_path="configs/models/llava_critic_7b.yaml")
    t0 = time.time()
    model.load_model(device=args.gpu)
    print(f"  model loaded in {time.time()-t0:.1f}s\n")

    correct, n_calls, call_times = 0, 0, []
    for k, ex in enumerate(picks):
        img = os.path.join(DATA_ROOT, ex["Image"])
        q = ex.get("Text") or "Describe / respond to the image."
        results = {}
        for out_key in ("Output1", "Output2"):
            prompt = PROMPT.format(question=q, answer=ex[out_key])
            tc = time.time()
            gen = model.generate(img, prompt)
            dt = time.time() - tc
            call_times.append(dt); n_calls += 1
            score, s2 = extract_score_and_logprobs(gen)
            results[out_key] = (score, s2, dt)
        s1 = results["Output1"][0]; s2_ = results["Output2"][0]
        if s1 is None or s2_ is None:
            pred = "PARSE_FAIL"
        elif s1 == s2_:
            pred = "TIE"
        else:
            pred = "Output1" if s1 > s2_ else "Output2"
        gt = ex["Better"]
        ok = (pred == gt)
        correct += int(ok)
        print(f"[{k+1}] cat={ex.get('Category','?'):32s} img_ok={os.path.exists(img)}")
        print(f"     Output1 score={results['Output1'][0]}  s2_lp={[round(x,2) for x in results['Output1'][1]]}  ({results['Output1'][2]:.1f}s)")
        print(f"     Output2 score={results['Output2'][0]}  s2_lp={[round(x,2) for x in results['Output2'][1]]}  ({results['Output2'][2]:.1f}s)")
        print(f"     predicted better={pred}  |  human Better={gt}  |  {'MATCH' if ok else 'miss'}\n")

    avg = sum(call_times) / len(call_times)
    print("="*60)
    print(f"SMOKE SUMMARY: {n_calls} judge calls | avg {avg:.1f}s/call")
    print(f"preference match: {correct}/{len(picks)} (sanity only, tiny N)")
    total_calls = len(data) * 2
    print(f"Full run = {len(data)} triplets x2 = {total_calls} calls")
    print(f"  1 GPU: ~{total_calls*avg/3600:.1f} h | 4-way: ~{total_calls*avg/3600/4:.1f} h | 6-way: ~{total_calls*avg/3600/6:.1f} h")


if __name__ == "__main__":
    main()
