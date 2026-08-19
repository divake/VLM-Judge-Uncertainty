#!/usr/bin/env python3
"""Run a VLM judge on Multimodal RewardBench (pairwise scoring), sharded + resumable.

Each triplet -> score Output1 and Output2 on the 1-5 rubric, extract parsed score
+ 5-dim Signal-2 logprobs for each. One CSV row per triplet. Deterministic shard
split (idx % num_shards == shard). Resumable: skips IDs already in the output CSV.
Flushes every row so progress is visible and crashes lose <=1 item.

Usage (one process per GPU shard):
    conda run -n vlmjudge python scripts/run_rewardbench_judge.py \
        --shard 0 --num-shards 6 --gpu cuda:0 \
        --out results/rewardbench/llava_critic/shard0.csv
"""
import argparse, csv, json, math, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llava import LLaVAModel

DATA_ROOT = "data/multimodal_rewardbench/raw/data"
ALL_DATA = os.path.join(DATA_ROOT, "all_data.json")
SCORE_TOKENS = ["1", "2", "3", "4", "5"]
FALLBACK_LP = math.log(1e-5)

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

FIELDS = ["ID", "Category", "Better", "img_ok",
          "score1", "o1_lp1", "o1_lp2", "o1_lp3", "o1_lp4", "o1_lp5",
          "score2", "o2_lp1", "o2_lp2", "o2_lp3", "o2_lp4", "o2_lp5"]


def extract(gen):
    tokens = gen.tokens
    score_idx = None
    for i, tok in enumerate(tokens):
        if "score" in tok.lower():
            for j in range(i, min(i + 6, len(tokens))):
                if tokens[j].strip() in SCORE_TOKENS:
                    score_idx = j; break
        if score_idx is not None:
            break
    if score_idx is None:
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].strip() in SCORE_TOKENS:
                score_idx = i; break
    if score_idx is None:
        return None, [FALLBACK_LP] * 5
    parsed = int(tokens[score_idx].strip())
    lp = gen.top_logprobs[score_idx]
    return parsed, [lp.get(s, FALLBACK_LP) for s in SCORE_TOKENS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--gpu", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--id-list", dest="id_list", type=str, default=None,
                    help="Optional file of triplet IDs (one per line); process only these (overrides shard split). For rebalancing.")
    args = ap.parse_args()

    data = json.load(open(ALL_DATA))
    if getattr(args, "id_list", None):
        idset = set(l.strip() for l in open(args.id_list) if l.strip())
        mine = [ex for ex in data if ex["ID"] in idset]
    else:
        mine = [ex for i, ex in enumerate(data) if i % args.num_shards == args.shard]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for row in csv.DictReader(f):
                done.add(row["ID"])
    todo = [ex for ex in mine if ex["ID"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} assigned, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        print(f"[shard {args.shard}] nothing to do — COMPLETE", flush=True); return

    model = LLaVAModel(config_path="configs/models/llava_critic_7b.yaml")
    t0 = time.time(); model.load_model(device=args.gpu)
    print(f"[shard {args.shard}] model loaded in {time.time()-t0:.1f}s", flush=True)

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if write_header:
        w.writeheader(); fout.flush()

    n, t_start, n_fail = 0, time.time(), 0
    for ex in todo:
        img = os.path.join(DATA_ROOT, ex["Image"])
        q = ex.get("Text") or "Describe / respond to the image."
        row = {"ID": ex["ID"], "Category": ex.get("Category", ""), "Better": ex["Better"],
               "img_ok": int(os.path.exists(img))}
        try:
            for tag, okey in (("1", "Output1"), ("2", "Output2")):
                gen = model.generate(img, PROMPT.format(question=q, answer=ex[okey]))
                sc, s2 = extract(gen)
                row[f"score{tag}"] = sc if sc is not None else ""
                for k in range(5):
                    row[f"o{tag}_lp{k+1}"] = round(s2[k], 5)
                if sc is None:
                    n_fail += 1
        except Exception:
            n_fail += 1
            traceback.print_exc()
            for tag in ("1", "2"):
                row.setdefault(f"score{tag}", "")
                for k in range(5):
                    row.setdefault(f"o{tag}_lp{k+1}", FALLBACK_LP)
        w.writerow(row); fout.flush()
        n += 1
        if n % 25 == 0:
            rate = n / (time.time() - t_start)
            eta = (len(todo) - n) / rate / 60
            print(f"[shard {args.shard}] {n}/{len(todo)} | {rate*3600:.0f}/h | ETA {eta:.0f} min | fails {n_fail}", flush=True)
    fout.close()
    print(f"[shard {args.shard}] DONE {n} triplets, {n_fail} parse/err fails", flush=True)


if __name__ == "__main__":
    main()
