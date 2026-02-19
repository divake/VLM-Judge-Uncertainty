#!/usr/bin/env python3
"""
Extract score-token logprobs from qwen_eval.py output JSONs
→ model_logits/dsr1/Summeval_{dimension}_logits.csv

CSV format: columns [1, 2, 3, 4, 5, {dimension}]
  - Columns 1-5: log-probabilities for each score token at the score position
  - Last column: human ground-truth label (averaged across raters)
"""

import json
import math
import os
import pandas as pd

FALLBACK_LP = math.log(1e-5)   # used when a score token isn't in top-10

DIM_MAP = {
    "con": "consistency",
    "coh": "coherence",
    "flu": "fluency",
    "rel": "relevance",
}

RESULTS_DIR = "results/dsr1"
OUT_DIR     = "model_logits/dsr1"
os.makedirs(OUT_DIR, exist_ok=True)

for short, dimension in DIM_MAP.items():
    json_path = os.path.join(RESULTS_DIR, f"dsr1_{short}_summeval.json")
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} not found")
        continue

    print(f"Processing {dimension} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    skipped = 0
    for item in data:
        tokens   = item["logprobs"]["tokens"]
        top_lps  = item["logprobs"]["top_logprobs"]

        # Find last token that is a pure score digit (1-5)
        score_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in ["1", "2", "3", "4", "5"]:
                score_idx = i
                break

        if score_idx is None:
            skipped += 1
            continue

        lp_dict = top_lps[score_idx]   # top-10 dict at the score position
        row = {
            "1": lp_dict.get("1", FALLBACK_LP),
            "2": lp_dict.get("2", FALLBACK_LP),
            "3": lp_dict.get("3", FALLBACK_LP),
            "4": lp_dict.get("4", FALLBACK_LP),
            "5": lp_dict.get("5", FALLBACK_LP),
            dimension: item["scores"][dimension],
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=["1", "2", "3", "4", "5", dimension])
    out_path = os.path.join(OUT_DIR, f"Summeval_{dimension}_logits.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows ({skipped} skipped) → {out_path}")

print("\nDone.")
