# Current Project Status

**Last updated:** 2026-03-20
**Version:** 3.0
**Target venue:** CoLM (Conference on Language Modeling)
**Timeline:** Wrap up quickly (days, not weeks)

---

## One-Line Summary

Extending Sheng et al.'s LLM-as-Judge conformal prediction framework to VLM judges on multimodal tasks. CoT prompt implemented, all 9 methods running, per-dataset analysis in progress.

---

## What Works

- LLaVA-Critic-7B judge with **CoT prompt** on MLLM-Judge dataset (5,717 samples)
- Lean inference pipeline: 5s/sample, direct CSV output, 2-GPU parallel
- S2 feature extraction (5-dim score token logprobs) — less overconfident with CoT
- R2CCP: **0.900 coverage, 3.049 width** (hits 90% target exactly)
- CHR: **0.880 raw → 0.965 after boundary adjustment**
- LVD: **0.894 raw → 0.988 after boundary adjustment**
- All 9 conformal methods running
- Per-dataset R2CCP analysis running (14 dataset categories)

## What Doesn't Work / Was Abandoned

- **S1 (actor confidence):** No GT for actor-generated answers
- **S3 (judge self-uncertainty):** Deterministic function of S2
- **3-signal MLP fusion:** Dead end
- **CQR/AsymCQR:** Over-covers (99.6%) — features too weak for quantile regression
- **OrdinalAPS:** Over-covers (99.9%) with fixed grid — 5 discrete classes too coarse
- **CLIPScore as visual signal:** Pearson=-0.017 with judge error (no correlation)
- **Old generic prompt:** Overconfident (51.4% samples >0.99 softmax)

## What's Done (v3.0)

- [x] New CoT prompt + lean runner
- [x] Full dataset re-run (5,717 samples)
- [x] All 9 conformal methods + boundary adjustment
- [x] Per-dataset R2CCP analysis (all 14 categories)
- [x] Naive Split CP baseline (per-dataset comparison)
- [x] Error analysis (confusion matrix, relaxed accuracy, judge bias, all judges comparison)
- [x] CLIPScore pilot (negative result — doesn't correlate with judge error)

## What's Not Done Yet

- [ ] Novel contribution direction (TBD)
- [ ] More VLM judges (Qwen2.5-VL-7B, InternVL2-8B)
- [ ] Midpoint analysis
- [ ] Paper writing

---

## Key Files

| Purpose | File |
|---------|------|
| VLM inference | `src/models/llava.py` |
| Judge runner | `scripts/run_judge.py` |
| Full S2 analysis | `scripts/run_full_analysis.py` |
| Conformal runner | `src/conformal/runner.py` |
| Signal extraction | `src/signals/extractor.py` |
| Evaluation metrics | `src/evaluation/metrics.py` |
| Results | `results/full_dataset/llava_critic_7b_on_llava_15_7b/` |
| Features CSV | `results/full_dataset/llava_critic_7b_on_llava_15_7b/features_s2.csv` |
| Reference paper | `Analyzing_Uncertainty_of_LLM-as-a-Judge_*.pdf` |
| Sibling LLM repo | `/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge/` |

---

## Comparison: Our Results vs Sheng et al.

| Aspect | Sheng et al. (LLM) | Ours (VLM) |
|--------|-------------------|------------|
| R2CCP coverage | ~89-92% | 90.0% |
| R2CCP width | ~0.6-2.5 (varies by dataset) | 3.018 |
| CQR coverage | ~90-95% | 99.3% (over-covers) |
| Pearson (judge vs GT) | varies | 0.380 |
| Scale | 1-5 Likert | 1-5 Likert |

Our intervals are wider, likely because VLM judge logits are less informative for predicting human scores on multimodal tasks.
