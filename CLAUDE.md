# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

Research implementation for **"Uncertainty-Aware VLM-as-a-Judge: Interval Evaluations via 3-Signal Conformal Prediction"**.

**Target venue:** ECCV 2026

**Core idea:** Extend conformal prediction-based uncertainty quantification from text-only LLM judges (the sibling LLM repo) to Vision-Language Model (VLM) judges, using a novel 3-signal MLP that fuses:
1. **Signal 1** — Actor VLM's internal generation confidence (logprob entropy over visual tokens)
2. **Signal 2** — Judge VLM's confidence about the actor (score-token log-probabilities)
3. **Signal 3** — Judge VLM's self-uncertainty (entropy/margin of score distribution)

See `RESEARCH_INFO.md` for complete research context, paper references, datasets, models, and the full plan.

---

## Sibling Repository (Reference This Constantly)

**LLM-as-Judge repo:** `/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge/`

**What to borrow from it:**
- `evaluations and reprompt on server/qwen_eval.py` → pattern for running LLM inference with `output_scores=True` and logprob extraction. Adapt this for VLMs (add image input).
- `extract_logits.py` → exact pattern for scanning tokens backwards to find score token, extracting top-k logprobs for "1"-"5". Adapt for VLM output.
- `conformal predictors/R2CCP_rancom.py` → R2CCP conformal prediction code. Already copied to `conformal_predictors/` here.
- `analysis/score_performance.py` → correlation + coverage metrics calculation.
- `Example_GenAI-Bench/interval_processing.py` → `range_modification()` and `boundary_adjustment()`. Already copied to `conformal_predictors/` here.
- `Example_GenAI-Bench/evaluation_metrics.py` → Pearson/Spearman/Kendall/MSE/MAE. Already copied to `conformal_predictors/` here.

**Do NOT re-derive these patterns from scratch.** The LLM repo has working, validated code. Copy and adapt.

---

## Repository Structure

```
VLM_Judge_Uncertainty/
├── CLAUDE.md                         ← this file (generic guidance)
├── RESEARCH_INFO.md                  ← ALL research details, papers, ideas
├── actors/                           ← Actor VLM inference scripts
│   └── run_actor_vlm.py              ← runs actor VLM, saves answer + logprobs
├── judges/                           ← Judge VLM inference scripts
│   └── run_judge_vlm.py              ← runs judge VLM on (image, question, actor_answer)
├── mlp_fusion/                       ← 3-signal MLP training
│   └── train_mlp.py                  ← MLP (15→64→32→2) + conformal calibration
├── conformal_predictors/             ← Copied from LLM repo (working code)
│   ├── R2CCP_rancom.py               ← primary conformal method
│   ├── CQR_random.py
│   ├── BoostedCP_random.py
│   ├── CHR_random..py
│   ├── LVD_random.py
│   ├── OrdinalRC_random.py
│   ├── OrdinalAPS_random.py
│   ├── interval_processing.py        ← range_modification, boundary_adjustment
│   └── evaluation_metrics.py        ← Pearson/Spearman/Kendall/MSE/MAE/RMSE
├── data/
│   ├── mllm_judge/                   ← MLLM-as-a-Judge (4,414 samples, human GT)
│   ├── llava_bench/                  ← LLaVA-Bench In-the-Wild (300 QA)
│   ├── mmvet/                        ← MMVet (218 QA, 6 capability dimensions)
│   └── multimodal_rewardbench/       ← Multimodal RewardBench (5,211 triplets)
├── analysis/                         ← Jupyter notebooks for results
├── extract_logits_REFERENCE.py       ← LLM repo extract_logits.py for reference
├── models -> /ssd_4TB/divake/...     ← symlink to shared models (no duplication)
├── results/                          ← gitignored (judge output JSONs)
├── model_logits/                     ← gitignored (feature CSVs)
├── model_paths/                      ← gitignored (R2CCP checkpoints)
└── R2CCP-0.0.8-py3-none-any.whl     ← install with --no-deps
```

---

## Key Architecture Patterns

### The 3-Signal Feature Extraction Pipeline

```python
# Step 1: Run ACTOR VLM on (image, question)
#   → save: actor_answer (text), actor_logprobs (per-token entropy stats)
#   → script: actors/run_actor_vlm.py

# Step 2: Run JUDGE VLM on (image, question, actor_answer)
#   → save: judge_score (text), judge_score_logprobs (P("1")...P("5"))
#   → script: judges/run_judge_vlm.py

# Step 3: Extract 3-signal feature vector per sample
#   Signal 1 (5 dims):  [mean_entropy, max_entropy, min_confidence, std_entropy, n_tokens]
#   Signal 2 (5 dims):  [log_P("1"), log_P("2"), log_P("3"), log_P("4"), log_P("5")]
#   Signal 3 (3 dims):  [entropy_of_score_dist, top1_margin, score_probs_std]
#   Total: 13-dim feature vector per sample
#   → script: extract_features.py  (adapt from extract_logits_REFERENCE.py)

# Step 4: Conformal prediction
#   X = feature matrix (N × 13), y = human scores (N,)
#   50/50 calibration/test split → R2CCP or MLP → intervals
#   → scripts: conformal_predictors/R2CCP_rancom.py (baseline)
#              mlp_fusion/train_mlp.py (full 3-signal model)
```

### Score Token Extraction (from LLM repo, same logic applies to VLMs)

```python
# Scan backwards through generated tokens to find last score digit
score_idx = None
for i in range(len(tokens) - 1, -1, -1):
    if tokens[i] in ["1", "2", "3", "4", "5"]:
        score_idx = i
        break

# Extract logprobs for all 5 score tokens at that position
lp_dict = top_logprobs[score_idx]
signal_2 = [lp_dict.get(str(s), math.log(1e-5)) for s in range(1, 6)]
```

### Conformal Prediction Common Pattern

```python
from sklearn.model_selection import train_test_split
import numpy as np

alpha = 0.10  # targeting 90% coverage

X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

# Baseline: Signal 2 only (5 dims) → R2CCP
# Full model: all 3 signals (13 dims) → MLP nonconformity score → R2CCP

in_interval = [(low <= yt <= high) for (low, high), yt in zip(intervals, y_test)]
coverage = np.mean(in_interval)
width    = np.mean([high - low for low, high in intervals])
```

### Adding Image Input to Inference (VLM-specific)

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

processor = AutoProcessor.from_pretrained(model_name)
image = Image.open(image_path).convert("RGB")

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to("cuda:0")

generation = model.generate(
    **inputs,
    max_new_tokens=512,
    return_dict_in_generate=True,
    output_scores=True,   # ← critical for logprob extraction
    top_k=10,
    do_sample=False,
)
# Logprob extraction is IDENTICAL to LLM repo after this point
```

---

## Environment Setup

**Conda Environment:** `env_py311` (same as LLM repo — shared environment)

```bash
conda activate env_py311
pip install R2CCP-0.0.8-py3-none-any.whl --no-deps
pip install qwen_vl_utils pillow  # additional VLM dependencies
```

**Test R2CCP:**
```bash
python -c "from R2CCP.main import R2CCP; print('R2CCP OK')"
```

**MAPIE version is critical:** Must use `mapie==0.8.6` (NOT v1.x).
`MapieQuantileRegressor` was renamed to `ConformalizedQuantileRegressor` in v1.x.

---

## Hardware

- **GPUs:** 2× RTX 6000 Ada (47.4GB each = 94.8GB total VRAM)
- **Models in fp16:** Up to 47B fits comfortably on both GPUs with `device_map="auto"`
- **Models at int4:** Up to ~95B fits (e.g., Qwen2.5-VL-72B in NF4 ≈ 40GB)
- **Model location:** `./models/` (symlink to `/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge/models/`)

---

## Important Notes

- **Always run from project root:** Scripts use relative paths
- **Alpha = 0.10:** Standard significance level → 90% coverage target
- **Rating scale:** 1–5 (same as LLM paper) or 1–10 for LLaVA-Bench
- **Calibration/test split:** 50/50, varied across 100 seeds for stable statistics
- **Score token search:** Always scan BACKWARDS through token list — reasoning chains contain digits
- **MAPIE must be 0.8.6:** Check before running CQR methods
- **VLM image input:** Use `AutoProcessor` not `AutoTokenizer` for vision models
