# Current Project Status

**Last updated:** 2026-03-25
**Version:** 3.1
**Target venue:** CoLM 2026 (Conference on Language Modeling)
**Paper title:** "How Reliable Are VLM Judges? Conformal Prediction Reveals Task-Dependent Uncertainty in Multimodal Evaluation"
**Timeline:** ~10 days to submission

---

## One-Line Summary

First systematic study of conformal prediction for VLM-as-a-Judge across 3 judges (LLaVA-Critic-7B, Phi-4-15B, Gemini 2.5 Flash), 8 CP methods, 14 visual task categories, 5,717 samples. Key finding: uncertainty is strongly task-dependent and CP exposes a ranking-scoring decoupling invisible to standard metrics.

---

## VLM Judges

| Judge | Type | Size | Status | Env | Key Feature |
|-------|------|------|--------|-----|-------------|
| LLaVA-Critic-7B | Open-source | 7B | COMPLETE | env_py311 | Best calibrated logprob distributions |
| Phi-4-reasoning-vision-15B | Open-source | 15B | RUNNING (GPUs) | phi4_env | Reasoning model with `<think>` mode |
| Gemini 2.5 Flash | Closed-source | Unknown | RUNNING (API) | env_py311 + Vertex AI | First closed-source VLM with logprobs |

## What Works (v3.0 — LLaVA-Critic complete)

- CoT prompt (adapted from MLLM-Judge dataset) on all 3 judges
- All 8 conformal methods + boundary adjustment (R2CCP best: 0.900 coverage, 3.049 width)
- Per-dataset R2CCP analysis (14 categories: aesthetics→2.08 width, infographics→3.50 width)
- Error analysis: 75.1% within ±1, 3-class accuracy 58%, judge bias +0.38
- CP value analysis: recovers 97.8% of judge errors, coverage holds across all 14 datasets
- Per-dataset Naive CP baseline (R2CCP beats on all 14 datasets)
- Confusion matrix, all-judges comparison, relaxed accuracy per dataset

## What Doesn't Work / Was Abandoned

- S1/S3 signals (no improvement over S2 alone)
- 3-signal MLP fusion (dead end)
- CQR/AsymCQR/OrdinalAPS (over-cover to ~100% with near-full-range intervals)
- CLIPScore as visual grounding signal (Pearson=-0.017 with judge error)
- Old generic prompt (51.4% overconfident vs 25.6% with CoT)
- Gemini via Google AI Studio (logprobs not supported — must use Vertex AI)

## What's In Progress

- [ ] Phi-4-reasoning-vision-15B full run (running on 2 GPUs, ~8-9 hours remaining)
- [ ] Gemini 2.5 Flash full run (running on Vertex AI, 4 parallel, ~5-6 hours remaining)
- [ ] Novel contribution direction (discussing with professor)

## What's Not Done Yet

- [ ] Analysis on Phi-4 and Gemini results (run all CP methods, error analysis, per-dataset)
- [ ] Cross-judge comparison table
- [ ] Novel method contribution (multi-judge disagreement? adaptive α? re-prompting?)
- [ ] Midpoint analysis
- [ ] Paper writing

---

## Key Files

### Scripts
| Purpose | File | Env |
|---------|------|-----|
| LLaVA-Critic inference | `scripts/run_judge_lean.py` | env_py311 |
| Phi-4 inference | `scripts/run_judge_phi4.py` | phi4_env |
| Gemini inference | `scripts/run_judge_gemini.py` | env_py311 + Vertex AI |
| All 9 CP methods | `scripts/run_all_conformal.py` | env_py311 |
| Per-dataset R2CCP | `scripts/run_r2ccp_per_dataset.py` | env_py311 |
| CLIPScore pilot | `scripts/clip_pilot.py` | env_py311 |
| Test runner | `scripts/test_lean_runner.py` | env_py311 |

### Results
| Purpose | File |
|---------|------|
| LLaVA-Critic features | `results/v2/features_s2.csv` |
| Phi-4 features | `results/v2_phi4/features_s2.csv` (in progress) |
| Gemini features | `results/v2_gemini/features_s2.csv` (in progress) |
| All CP methods | `results/v2/all_methods_results.csv` |
| Per-dataset R2CCP | `results/v2/r2ccp_per_dataset.csv` |
| Error analysis | `results/v2/error_analysis*.csv` |
| Complete results doc | `results/v2/RESULTS_v3.0.md` |
| Paper abstract | `paper/abstract_draft.md` |

### Configs
| Purpose | File |
|---------|------|
| CoT prompt | `configs/prompts/mllm_judge_cot.yaml` |
| Experiment config | `configs/experiments/pilot_mllm_judge.yaml` |
| LLaVA-Critic model | `configs/models/llava_critic_7b.yaml` |
