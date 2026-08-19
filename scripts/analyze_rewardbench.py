#!/usr/bin/env python3
"""Analyze LLaVA-Critic on Multimodal RewardBench: pairwise accuracy + error detection.

Merges shard CSVs, derives preference from logprob-weighted expected scores,
reports accuracy (overall + per category), then evaluates which confidence signal
best detects judge errors (AUROC + AURC / risk-coverage).

Usage: conda run -n vlmjudge python scripts/analyze_rewardbench.py
"""
import csv, glob, json, math, os
import numpy as np

ROOT = "/ssd_4TB/divake/VLM_Judge_Uncertainty"
os.chdir(ROOT)
DATA = "data/multimodal_rewardbench/raw/data/all_data.json"
SHARD_GLOB = "results/rewardbench/llava_critic/shard*.csv"
OUT = "plan/RESULTS_rewardbench_llava.md"

def softmax(lps):
    m = max(lps); e = [math.exp(x - m) for x in lps]; s = sum(e)
    return [x / s for x in e]

def expected_score(lps):
    p = softmax(lps); return sum((k + 1) * p[k] for k in range(5))

def entropy(lps):
    p = softmax(lps); return -sum(x * math.log(x + 1e-12) for x in p)

def auroc(scores, labels):
    # labels: 1=error(positive). scores: higher => more likely error.
    order = np.argsort(-np.asarray(scores)); y = np.asarray(labels)[order]
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0: return float("nan")
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / P; fpr = fp / N
    return float(np.trapz(tpr, fpr))

def aurc(conf, errors):
    # rank by confidence descending (most confident first); risk = cum error rate.
    order = np.argsort(-np.asarray(conf)); e = np.asarray(errors)[order]
    cum = np.cumsum(e) / (np.arange(len(e)) + 1)
    return float(cum.mean())

def main():
    # Merge all shard rows. Each row is a self-contained scored triplet (its own
    # Better/Category stored at scoring time). Benchmark has 4711 entries but only
    # 4482 unique IDs (229 share an ID with a different response pair), so we do
    # NOT dedup by ID — we dedup only exact-duplicate rows (from resume reprocessing).
    seen, allrows = set(), []
    n_empty = 0
    for f in sorted(glob.glob(SHARD_GLOB)):
        for r in csv.DictReader(open(f)):
            if r["score1"] == "" or r["score2"] == "":
                n_empty += 1; continue
            key = (r["ID"], r["Better"], r["score1"], r["score2"],
                   r.get("o1_lp4", ""), r.get("o2_lp4", ""))
            if key in seen: continue
            seen.add(key); allrows.append(r)
    n = len(allrows)
    print(f"merged {n} scored triplets ({n_empty} empty-score rows dropped) from {len(glob.glob(SHARD_GLOB))} shards")

    recs = []
    for r in allrows:
        lp1 = [float(r[f"o1_lp{k}"]) for k in range(1, 6)]
        lp2 = [float(r[f"o2_lp{k}"]) for k in range(1, 6)]
        e1, e2 = expected_score(lp1), expected_score(lp2)
        pred = "Output1" if e1 > e2 else "Output2"
        better = r["Better"]                                        # from CSV row (matches scored pair)
        err = int(pred != better)
        conf_margin = abs(e1 - e2)                                  # logprob/conformal-style
        conf_maxprob = max(max(softmax(lp1)), max(softmax(lp2)))    # token-prob baseline
        conf_negent = -0.5 * (entropy(lp1) + entropy(lp2))          # entropy baseline
        recs.append(dict(cat=r.get("Category", "?"), err=err,
                         margin=conf_margin, maxprob=conf_maxprob, negent=conf_negent))

    N = len(recs); errs = np.array([x["err"] for x in recs])
    acc = 1 - errs.mean()
    print(f"\nPairwise accuracy (expected-score preference): {acc:.4f}  (n={N})")

    # per category
    cats = sorted(set(x["cat"] for x in recs))
    cat_lines = []
    for c in cats:
        sub = [x["err"] for x in recs if x["cat"] == c]
        cat_lines.append(f"  {c:40s} acc={1-np.mean(sub):.3f}  n={len(sub)}")
        print(cat_lines[-1])

    # error detection: signal that best flags errors
    sig = {}
    for name, key in [("expected_margin(ours)", "margin"),
                      ("token_maxprob(baseline)", "maxprob"),
                      ("neg_entropy(baseline)", "negent")]:
        conf = np.array([x[key] for x in recs])
        # AUROC of (-conf) predicting error (low conf => error)
        a = auroc(-conf, errs)
        rc = aurc(conf, errs)
        sig[name] = (a, rc)
        print(f"error-detect {name:26s} AUROC={a:.3f}  AURC={rc:.3f}")

    best = max(sig, key=lambda k: sig[k][0])
    ours = sig["expected_margin(ours)"][0]
    base = sig["token_maxprob(baseline)"][0]
    go = ours > base

    with open(OUT, "w") as fo:
        fo.write("# RESULTS — LLaVA-Critic-7B on Multimodal RewardBench\n\n")
        fo.write(f"Merged **{n}** scored triplets ({n_empty} empty-score rows dropped, {N} usable). "
                 f"Benchmark = 4711 entries / 4482 unique IDs.\n\n")
        fo.write("## Pairwise preference accuracy (logprob expected-score)\n")
        fo.write(f"- **Overall: {acc:.4f}** (n={N}). RewardBench leaderboard ref: Llava-1.5-13B=0.489, GPT-4o=0.715.\n\n")
        fo.write("### Per category\n```\n" + "\n".join(cat_lines) + "\n```\n\n")
        fo.write("## Error detection — which confidence signal flags wrong judgments best?\n")
        fo.write("(AUROC higher=better; AURC lower=better. Signals rank triplets by judge confidence.)\n\n")
        fo.write("| Signal | AUROC | AURC |\n|---|---|---|\n")
        for name, (a, rc) in sig.items():
            fo.write(f"| {name} | {a:.3f} | {rc:.3f} |\n")
        fo.write(f"\n**Best detector: {best}**\n\n")
        fo.write("## Go / No-Go\n")
        fo.write(f"- Our logprob signal (expected_margin) AUROC={ours:.3f} vs token-maxprob baseline AUROC={base:.3f}.\n")
        fo.write(f"- **{'GO' if go else 'NO-GO'}**: the logprob/conformal-style signal "
                 f"{'BEATS' if go else 'does NOT beat'} the token-probability baseline at detecting judge errors "
                 f"on this clean pairwise benchmark.\n")
        fo.write("- Note: expected_margin is the raw logprob signal; the full R2CCP conformal signal "
                 "(nonconformity/interval) is the next step and expected to match or exceed this.\n")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
