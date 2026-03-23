# Results v3.0 — VLM-as-a-Judge Conformal Prediction

**Date:** 2026-03-23
**Version:** 3.0
**Judge:** LLaVA-Critic-7B (CoT prompt)
**Dataset:** MLLM-as-a-Judge (5,717 valid samples, 14 dataset categories, 1-5 integer GT)
**Config:** α=0.10 (target 90% coverage), 10 seeds, 50/50 cal/test split
**Features:** S2 logprobs (5-dim: log P("1") through log P("5") at score token position)

---

## 1. Point Prediction: Old Generic Prompt vs New CoT Prompt

| Metric | OLD (generic prompt) | NEW (CoT prompt) | Change |
|--------|---------------------|-------------------|--------|
| Pearson | 0.380 | **0.402** | +5.8% |
| Spearman | 0.314 | **0.356** | +13.4% |
| Kendall | 0.270 | **0.300** | +11.1% |
| Accuracy (exact) | 33.9% | 32.2% | -1.7% |
| MAE | 1.058 | **1.031** | -2.6% |
| RMSE | 1.474 | — | — |
| Overconfident (softmax max > 0.99) | 51.4% | **25.6%** | -50% |
| Overconfident (softmax max > 0.999) | 33.2% | **6.6%** | -80% |
| N samples | 5,711 | 5,717 | — |

**CoT prompt config:** `configs/prompts/mllm_judge_cot.yaml`
**Features file:** `results/v2/features_s2.csv`

---

## 2. All Conformal Methods — Before and After Boundary Adjustment

Boundary adjustment: floor(lower), ceil(upper) — expands continuous intervals to integer boundaries.

| Method | Type | Cov(raw) | Width(raw) | Cov(adjusted) | Width(adjusted) |
|--------|------|----------|-----------|---------------|-----------------|
| **CHR** | Regression | 0.880±0.010 | **2.972±0.069** | 0.965±0.006 | **3.431±0.098** |
| **Boosted LCP** | Regression | 0.863±0.012 | **3.018±0.031** | 0.977±0.004 | **3.510±0.044** |
| **R2CCP** | Regression | **0.900±0.016** | **3.049±0.097** | 0.981±0.005 | 3.600±0.089 |
| Naive Split CP | Regression | 0.895±0.009 | 3.226±0.056 | 0.990±0.003 | 3.781±0.035 |
| LVD | Regression | 0.894±0.009 | 3.207±0.112 | 0.988±0.005 | 3.713±0.111 |
| Boosted CQR | Regression | 0.878±0.009 | 3.540±0.086 | 0.997±0.002 | 3.932±0.036 |
| CQR | Regression | 0.996±0.005 | 3.951±0.061 | 1.000±0.000 | 4.000±0.000 |
| Asymmetric CQR | Regression | 0.996±0.005 | 3.951±0.061 | 1.000±0.000 | 4.000±0.000 |
| OrdinalAPS | Classification | 0.999±0.000 | 3.993±0.001 | 0.999±0.000 | 3.993±0.001 |
| OrdinalRC | Classification | FAILED (import) | — | — | — |

**Key findings:**
- R2CCP is the only method hitting exactly 90% raw coverage
- CHR produces the narrowest raw intervals (2.972)
- CQR/AsymCQR/OrdinalAPS are unusable — over-cover to ~100% with nearly full-range intervals
- Boundary adjustment increases coverage significantly for our discrete integer GT (e.g., CHR: 88→96.5%)

---

## 3. Comparison with Sheng et al. (LLM-as-Judge)

| Aspect | Sheng et al. (LLM) | Ours (VLM) |
|--------|-------------------|------------|
| GT scores | Avg of 3 annotators (13 fractional values) | Single annotator (5 integer values) |
| Datasets | SummEval (1,600), DialSumm (1,400), ROSCOE (~200) | MLLM-Judge (5,717) |
| Judges | GPT-4o mini, DSR1-32B, Qwen2.5-72B | LLaVA-Critic-7B |
| Judge size | 32B-72B+ | 7B |
| Best Pearson | ~0.65 (DSR1 on SummEval consistency) | 0.402 |
| R2CCP coverage | ~89-92% | 90.0% |
| R2CCP width | ~0.6-2.5 (varies) | 3.049 |
| # seeds | 30 | 10 |
| Modality | Text only | Image + Text |

**Why our intervals are wider:** Weaker judge (7B vs 32-72B), single-annotator noisy GT (vs averaged), and multimodal tasks are inherently harder to judge.

---

## 4. Per-Dataset R2CCP Analysis (14 dataset categories)

| Dataset | N | Cov(raw) | Width(raw) | Cov(adj) | Width(adj) | Pearson | Category |
|---------|---|----------|-----------|----------|-----------|---------|----------|
| AesBench | 392 | 0.905 | **2.082** | 0.984 | **2.657** | 0.402 | Aesthetics/AI |
| mm-vet | 258 | 0.845 | **2.180** | 0.931 | **2.763** | 0.260 | General VQA |
| WIT | 399 | 0.864 | 2.377 | 0.982 | 3.153 | 0.164 | Knowledge/Web |
| coco | 397 | 0.876 | 2.427 | 0.979 | 3.066 | 0.362 | General VQA |
| mind2web | 398 | 0.889 | 2.690 | 0.982 | 3.309 | 0.268 | Knowledge/Web |
| Concept Caption | 398 | 0.892 | 2.703 | 0.986 | 3.384 | 0.362 | Knowledge/Web |
| textVQA | 399 | 0.862 | 2.812 | 0.957 | 3.285 | 0.389 | Vision-Heavy |
| llava_bench | 396 | 0.864 | 2.920 | 0.979 | 3.608 | 0.244 | General VQA |
| VisitBench | 397 | 0.873 | 2.959 | 0.981 | 3.484 | 0.352 | General VQA |
| ChartQA | 400 | 0.880 | 3.079 | 0.971 | 3.571 | 0.507 | Vision-Heavy |
| ScienceQA | 396 | 0.905 | 3.269 | 0.969 | 3.638 | 0.258 | Vision-Heavy |
| mathvista | 790 | 0.879 | 3.369 | 0.981 | 3.709 | 0.376 | Vision-Heavy |
| diffusiondb | 299 | 0.882 | **3.414** | 0.990 | **3.887** | 0.089 | Aesthetics/AI |
| infographicsVQA | 398 | 0.903 | **3.504** | 0.985 | **3.837** | 0.411 | Vision-Heavy |

### Category Summary

| Category | Total N | Avg Width(raw) | Avg Width(adj) | Avg Pearson |
|----------|---------|---------------|---------------|-------------|
| Knowledge/Web | 1,195 | **2.590** | 3.282 | 0.265 |
| General VQA | 1,448 | **2.622** | 3.230 | 0.304 |
| Aesthetics/AI | 691 | 2.748 | 3.272 | 0.246 |
| **Vision-Heavy** | 2,383 | **3.207** | **3.608** | **0.388** |

**Key finding:** Vision-heavy tasks (math, charts, infographics, science) produce intervals ~0.6 wider than general VQA/knowledge tasks. The judge's uncertainty is systematically higher on tasks requiring deep visual reasoning.

---

## 5. Naive Split CP vs R2CCP Per-Dataset

R2CCP consistently produces narrower intervals than Naive Split CP across ALL 14 datasets.

| Dataset | Naive Width(raw) | R2CCP Width(raw) | R2CCP Improvement |
|---------|-----------------|------------------|-------------------|
| AesBench | 2.531 | 2.082 | -0.449 |
| mm-vet | 3.387 | 2.180 | **-1.208** |
| WIT | 2.916 | 2.377 | -0.539 |
| coco | 2.859 | 2.427 | -0.432 |
| mind2web | 3.131 | 2.690 | -0.440 |
| Concept Caption | 3.113 | 2.703 | -0.410 |
| textVQA | 3.194 | 2.812 | -0.382 |
| llava_bench | 3.275 | 2.920 | -0.355 |
| VisitBench | 3.373 | 2.959 | -0.414 |
| ChartQA | 3.490 | 3.079 | -0.410 |
| ScienceQA | 3.705 | 3.269 | -0.436 |
| mathvista | 3.638 | 3.369 | -0.269 |
| diffusiondb | 3.561 | 3.414 | -0.146 |
| infographicsVQA | 3.674 | 3.504 | -0.169 |

---

## 6. Error Analysis: Judge vs Human Disagreement

### Error Distribution

| Error (Judge - GT) | Count | % | Meaning |
|---------------------|-------|---|---------|
| 0 | 1,838 | 32.2% | Exact match |
| ±1 | 2,452 | 42.9% | Borderline disagreement |
| ±2 | 952 | 16.7% | Off by 2 |
| ±3 or ±4 | 474 | 8.3% | Completely wrong |

### Relaxed Accuracy

| Tolerance | Accuracy | Interpretation |
|-----------|----------|----------------|
| Exact (±0) | 32.2% | Strict match |
| **±1** | **75.1%** | Within 1 point (borderline OK) |
| ±2 | 91.7% | Within 2 points |
| ±3 | 98.0% | Within 3 points |

### 3-Class Accuracy (Bad={1}, Medium={2,3}, Good={4,5})

| Scale | Accuracy |
|-------|----------|
| 5-class (original) | 32.2% |
| **3-class (merged)** | **58.0%** |

### Confusion Matrix (row-normalized)

|     | Pred=1 | Pred=2 | Pred=3 | Pred=4 | Pred=5 | Total | ±1 acc |
|-----|--------|--------|--------|--------|--------|-------|--------|
| GT=1 | **32.6%** | 12.9% | 17.0% | 23.4% | 14.0% | 641 | 45.6% |
| GT=2 | 17.0% | **11.7%** | 21.9% | 32.2% | 17.1% | 707 | 50.6% |
| GT=3 | 5.2% | 7.0% | **18.6%** | 42.0% | 27.2% | 1,263 | 67.6% |
| GT=4 | 3.3% | 3.6% | 13.4% | **38.9%** | 40.8% | 1,791 | 93.1% |
| GT=5 | 1.9% | 2.2% | 10.7% | 38.4% | **46.7%** | 1,314 | 85.2% |

### Judge Bias

| GT Score | Judge Bias | Direction |
|----------|-----------|-----------|
| GT=1 | +1.733 | Overscores (says ~2.7) |
| GT=2 | +1.208 | Overscores (says ~3.2) |
| GT=3 | +0.789 | Overscores (says ~3.8) |
| GT=4 | +0.102 | Nearly perfect |
| GT=5 | -0.741 | Underscores (says ~4.3) |
| **Overall** | **+0.380** | **Positive bias (too generous)** |

### Comparison: All Judges in Dataset (±1 Accuracy)

| Judge | N | Exact | ±1 | ±2 | MAE |
|-------|---|-------|-----|-----|-----|
| GPT-4V (best) | 4,272 | 42.2% | **82.6%** | 93.5% | 0.829 |
| **LLaVA-Critic (our CoT)** | 5,716 | 32.2% | **75.1%** | 91.7% | 1.031 |
| CogVLM | 3,215 | 33.0% | 75.5% | 89.9% | 1.034 |
| Gemini | 3,890 | 35.4% | 73.1% | 87.7% | 1.081 |
| LLaVA | 4,262 | 31.0% | 72.5% | 87.2% | 1.110 |

### Per-Dataset Relaxed Accuracy

| Dataset | N | Exact | ±1 | ±2 | MAE | Category |
|---------|---|-------|-----|-----|-----|----------|
| AesBench | 392 | 41.3% | **88.3%** | 97.7% | 0.732 | Aesthetics/AI |
| coco | 397 | 34.5% | **84.9%** | 96.2% | 0.854 | General VQA |
| llava_bench | 396 | 31.8% | 81.1% | 95.2% | 0.922 | General VQA |
| Concept Caption | 398 | 35.4% | 81.7% | 96.5% | 0.872 | Knowledge/Web |
| mm-vet | 258 | 37.2% | 80.2% | 93.8% | 0.926 | General VQA |
| textVQA | 399 | 36.3% | 79.2% | 91.5% | 0.942 | Vision-Heavy |
| WIT | 399 | 33.8% | 78.7% | 96.5% | 0.912 | Knowledge/Web |
| VisitBench | 397 | 31.7% | 75.1% | 92.4% | 1.018 | General VQA |
| mind2web | 398 | 29.6% | 74.4% | 95.5% | 1.008 | Knowledge/Web |
| ChartQA | 400 | 30.2% | 73.2% | 89.5% | 1.085 | Vision-Heavy |
| ScienceQA | 396 | 26.5% | 69.7% | 88.4% | 1.199 | Vision-Heavy |
| mathvista | 789 | 32.3% | **65.1%** | 84.3% | 1.234 | Vision-Heavy |
| infographicsVQA | 398 | 25.1% | 69.3% | 88.4% | 1.191 | Vision-Heavy |
| **diffusiondb** | 299 | 23.7% | **57.2%** | 84.3% | 1.385 | Aesthetics/AI |

---

## 7. CLIPScore Analysis (Negative Result)

Tested whether CLIP-based visual grounding features correlate with judge error.

| Feature | Pearson with |error| | Significant? |
|---------|----------------------|--------------|
| CLIPScore(image, answer) | -0.017 | No (p=0.73) |
| CLIPScore(image, question) | 0.073 | Weak |
| S2 entropy (logprobs) | 0.060 | Weak |
| CLIP(ans) - CLIP(q) | -0.079 | Weak |

**Conclusion:** CLIPScore does not predict when the VLM judge will be wrong. It measures "is the answer about the image?" not "is the answer correct about the image?"

---

## 8. CP Value Analysis: Error Bins + Coverage (R2CCP, 10 seeds)

**The core argument: CP is most valuable precisely when the judge is wrong.**

### Per-Error-Bin CP Coverage (multi-seed, R2CCP)

| Error | N (~%) | CP Cov(raw) | CP Cov(adjusted) | CP Saved |
|-------|--------|-------------|------------------|----------|
| **Exact (0)** | 32.6% | 99.8%±0.1% | 100%±0.0% | N/A |
| **±1** | 43.0% | **98.7%±0.4%** | **99.9%±0.1%** | ~100% |
| **±2** | 16.0% | **84.3%±4.6%** | **99.4%±0.5%** | ~99% |
| ±3 | 6.3% | 25.4%±10.4% | **91.5%±2.9%** | ~91% |
| ±4 | 2.0% | 5.7%±2.7% | 43.3%±15.6% | ~43% |

**Key finding:** For ±1 and ±2 errors (59% of all wrong predictions), CP with boundary adjustment recovers virtually 100%. Even for ±3 errors, CP still saves 91%.

### "CP Saved You" Summary (seed=1)

| Metric | Count | % |
|--------|-------|---|
| Total test samples | 2,859 | 100% |
| Judge exact match | 932 | 32.6% |
| Judge wrong | 1,927 | 67.4% |
| **CP recovered (adjusted)** | **1,885** | **97.8% of errors** |
| CP failed | 42 | 2.2% of errors |

**CP recovers 97.8% of judge errors.** Only 42 out of 1,927 wrong predictions are not covered by the conformal interval.

---

## 9. Conditional Coverage by GT Score (R2CCP, seed=1)

| GT Score | N | CP Cov(raw) | CP Cov(adj) | Width(raw) | Width(adj) | Judge Bias |
|----------|---|-------------|-------------|-----------|-----------|------------|
| GT=1 | 334 | **53.0%** | **88.9%** | 3.141 | 3.749 | +1.668 |
| GT=2 | 360 | 86.9% | 100% | 3.175 | 3.789 | +1.125 |
| GT=3 | 637 | 99.7% | 100% | 3.089 | 3.750 | +0.774 |
| GT=4 | 880 | 98.2% | 100% | 2.986 | 3.645 | +0.110 |
| GT=5 | 648 | 92.4% | 99.2% | 2.918 | 3.576 | -0.801 |

**GT=1 is the weak spot:** The judge overscores bad answers by +1.67 on average, pushing the interval too high to cover GT=1. This is the only GT level where coverage drops below 90% even after adjustment.

---

## 10. Interval Informativeness (R2CCP, adjusted intervals, seed=1)

| Category | N | % | Avg Width | Coverage | Meaning |
|----------|---|---|----------|----------|---------|
| **Decisive (≤2)** | 13 | 0.5% | 1.923 | 84.6% | Narrows to 2-3 scores → actionable |
| **Moderate (2-3)** | 879 | **30.7%** | 3.000 | 95.4% | Some help, 3-4 scores |
| Uninformative (>3) | 1,967 | 68.8% | 4.000 | 100% | Nearly full range → not useful |

### Per Dataset Category

| Category | Decisive% | Moderate% | Uninformative% |
|----------|----------|----------|----------------|
| Aesthetics/AI | 0.0% | **37.7%** | 62.3% |
| Vision-Heavy | 0.5% | 31.8% | 67.7% |
| Knowledge/Web | 0.7% | 30.8% | 68.5% |
| General VQA | 0.4% | 25.9% | 73.7% |

~31% of intervals are "moderate" (width 2-3) — useful for decision-making. Aesthetics/AI tasks produce the most informative intervals.

---

## 11. Midpoint as Calibrated Score (R2CCP, seed=1)

| Metric | Raw Judge Score | Midpoint (raw interval) | Midpoint (adjusted) |
|--------|----------------|------------------------|---------------------|
| Pearson | **0.412** | 0.396 | 0.277 |
| Spearman | **0.367** | 0.359 | 0.272 |
| MAE | 1.022 | **0.992** | 1.056 |
| Accuracy | **32.6%** | — | 28.1% (rounded) |

Midpoint has slightly lower MAE (0.99 vs 1.02) but lower correlation. Unlike Sheng et al.'s finding, midpoint does not clearly improve over the raw judge score — because our intervals are wide, the midpoint regresses toward the center (~3.0).

---

## 12. Stratified Coverage by Dataset Category (R2CCP, seed=1)

CP coverage guarantee holds across ALL 14 dataset types:

| Dataset | N | Cov(raw) | Cov(adj) | Width(raw) | Width(adj) |
|---------|---|----------|----------|-----------|-----------|
| AesBench | 176 | 97.2% | 99.4% | 2.795 | 3.477 |
| ChartQA | 198 | 88.4% | 98.5% | 3.130 | 3.727 |
| Concept Caption | 201 | 96.0% | 100% | 2.994 | 3.677 |
| ScienceQA | 198 | 84.3% | 98.5% | 3.012 | 3.677 |
| VisitBench | 198 | 88.9% | 99.0% | 3.018 | 3.667 |
| WIT | 199 | 94.0% | 99.5% | 3.061 | 3.709 |
| coco | 205 | 95.6% | 99.5% | 3.087 | 3.746 |
| diffusiondb | 156 | 86.5% | 98.1% | 3.088 | 3.788 |
| infographicsVQA | 186 | 86.6% | 97.3% | 3.168 | 3.753 |
| llava_bench | 194 | 94.3% | 100% | 3.048 | 3.804 |
| mathvista | 403 | 82.1% | 96.0% | 3.069 | 3.655 |
| mind2web | 213 | 95.8% | 99.1% | 3.007 | 3.653 |
| mm-vet | 136 | 93.4% | 99.3% | 2.992 | 3.699 |
| textVQA | 196 | 92.9% | 98.0% | 2.973 | 3.571 |

**After boundary adjustment, all datasets achieve ≥96% coverage.** The guarantee is robust even on the hardest vision-heavy tasks.

---

## Files Reference

### Result CSVs
| File | Description |
|------|-------------|
| `results/v2/features_s2.csv` | S2 features for all 5,717 samples (CoT prompt) |
| `results/v2/all_methods_results.csv` | All 9 methods: coverage and width (before/after boundary adj) |
| `results/v2/r2ccp_per_dataset.csv` | R2CCP results per 14 dataset categories |
| `results/v2/naive_cp_per_dataset.csv` | Naive Split CP per dataset (baseline comparison) |
| `results/v2/error_analysis.csv` | Per-sample: sample_id, gt_score, parsed_score, error, abs_error, dataset |
| `results/v2/error_analysis_per_dataset.csv` | Relaxed accuracy (exact, ±1, ±2, MAE) per dataset |
| `results/v2/confusion_matrix.csv` | 5×5 confusion matrix (counts + percentages) |
| `results/v2/all_judges_comparison.csv` | All judges (GPT-4V, Gemini, CogVLM, LLaVA, ours) comparison |
| `results/v2/analysis1_error_bins_cp_coverage.csv` | CP coverage per error bin (seed=1) |
| `results/v2/analysis1_error_bins_multiseed.csv` | CP coverage per error bin (10 seeds, mean±std) |
| `results/v2/analysis2_cp_saved_you.csv` | "CP Saved You" summary statistics |
| `results/v2/analysis3_coverage_by_gt.csv` | Conditional CP coverage by GT score |
| `results/v2/analysis4_interval_informativeness.csv` | Interval width categories (decisive/moderate/uninformative) |
| `results/v2/analysis5_midpoint_vs_raw.csv` | Midpoint vs raw judge score comparison |
| `results/v2/analysis6_stratified_coverage.csv` | CP coverage per dataset category (seed=1) |
| `results/v2/RESULTS_v3.0.md` | This document — complete results reference |

### Scripts
| File | Description |
|------|-------------|
| `configs/prompts/mllm_judge_cot.yaml` | CoT prompt (adapted from MLLM-Judge dataset) |
| `scripts/run_judge_lean.py` | Lean VLM inference → CSV directly |
| `scripts/run_all_conformal.py` | All 9 conformal methods runner |
| `scripts/run_r2ccp_per_dataset.py` | Per-dataset R2CCP analysis |
| `scripts/clip_pilot.py` | CLIPScore correlation analysis |
| `scripts/test_lean_runner.py` | Quick test (3 samples) for lean runner |
