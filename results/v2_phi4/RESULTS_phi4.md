# Results — Phi-4-reasoning-vision-15B

**Date:** 2026-03-25
**Judge:** Phi-4-reasoning-vision-15B (Microsoft, 15B params, `<think>` reasoning mode)
**Dataset:** MLLM-as-a-Judge (5,717 valid samples, 14 dataset categories, 1-5 integer GT)
**Config:** α=0.10, 10 seeds, 50/50 cal/test split, same CoT prompt as LLaVA-Critic
**Env:** phi4_env (transformers==4.57.1)

---

## 1. Point Prediction: Phi-4 vs LLaVA-Critic

| Metric | LLaVA-Critic-7B | Phi-4-reasoning-15B | Better |
|--------|----------------|---------------------|--------|
| Pearson | **0.402** | 0.303 | LLaVA |
| Spearman | **0.356** | 0.293 | LLaVA |
| Kendall | **0.300** | 0.251 | LLaVA |
| Accuracy (exact) | 32.2% | **34.2%** | Phi-4 |
| ±1 Accuracy | 75.1% | **76.1%** | Phi-4 |
| MAE | 1.031 | **1.005** | Phi-4 |
| Bias | +0.380 | **+0.084** | Phi-4 (much less biased) |
| Max softmax mean | **0.875** | 0.906 | LLaVA (less overconfident) |
| Overconfident (>0.99) | **25.6%** | 43.0% | LLaVA |

**Key finding:** Phi-4 has higher accuracy and much lower bias but lower correlation and more overconfident distributions. The reasoning mode helps it score more accurately but makes logprobs more peaked.

---

## 2. Error Distribution

| Error | Count | % |
|-------|-------|---|
| Exact (0) | 1,933 | 34.2% |
| ±1 | 2,369 | 41.9% |
| ±2 | 856 | 15.1% |
| ±3 | 380 | 6.7% |
| ±4 | 115 | 2.0% |
| **Within ±1** | **4,302** | **76.1%** |

### Relaxed Accuracy

| Tolerance | Phi-4 | LLaVA-Critic |
|-----------|-------|-------------|
| Exact (±0) | **34.2%** | 32.2% |
| ±1 | **76.1%** | 75.1% |
| ±2 | 91.2% | **91.7%** |
| ±3 | **98.0%** | 98.0% |

---

## 3. Confusion Matrix (Phi-4)

|     | Pred=1 | Pred=2 | Pred=3 | Pred=4 | Pred=5 | ±1 acc |
|-----|--------|--------|--------|--------|--------|--------|
| GT=1 | 21.6% | 18.1% | 13.8% | 33.6% | 13.0% | 39.6% |
| GT=2 | 12.8% | 18.4% | 21.5% | 37.9% | 9.4% | 52.7% |
| GT=3 | 6.4% | 13.3% | 28.1% | 44.5% | 7.7% | 85.9% |
| GT=4 | 2.4% | 10.2% | 19.5% | 56.2% | 11.8% | 87.4% |
| GT=5 | 2.5% | 4.6% | 11.4% | 56.9% | 24.6% | 81.4% |

### Judge Bias by GT Score

| GT | Phi-4 Bias | LLaVA-Critic Bias |
|----|-----------|-------------------|
| GT=1 | +1.984 | +1.733 |
| GT=2 | +1.125 | +1.208 |
| GT=3 | **+0.338** | +0.789 |
| GT=4 | **-0.353** | +0.102 |
| GT=5 | **-1.037** | -0.741 |
| Overall | **+0.084** | +0.380 |

**Key finding:** Phi-4 has a strong "gravity toward 4" — 56.2% of GT=4 and 56.9% of GT=5 get scored 4. It underscore good answers (GT=5 bias=-1.04) more than LLaVA-Critic. Overall bias is much lower (+0.08 vs +0.38).

---

## 4. R2CCP Results

| Metric | Phi-4 | LLaVA-Critic |
|--------|-------|-------------|
| Coverage (raw) | 0.891±0.013 | **0.900±0.016** |
| Width (raw) | 3.133±0.088 | **3.049±0.097** |
| Coverage (adj) | 0.981±0.007 | 0.981±0.005 |
| Width (adj) | 3.702±0.069 | **3.600±0.089** |

Phi-4 produces slightly wider intervals (3.13 vs 3.05) and slightly under-covers (89.1% vs 90.0%). The more overconfident distributions give R2CCP less signal to work with.

### All CP Methods (Phi-4)

| Method | Cov(raw) | Width(raw) | Cov(adj) | Width(adj) |
|--------|----------|-----------|----------|-----------|
| R2CCP | **0.891±0.013** | **3.133±0.088** | 0.981±0.007 | **3.702±0.069** |
| CHR | 0.896±0.013 | 3.164±0.072 | 0.976±0.006 | 3.619±0.082 |
| Boosted LCP | 0.875±0.009 | 3.235±0.029 | 0.987±0.002 | 3.754±0.025 |
| CQR | 0.997±0.003 | 3.981±0.029 | 1.000±0.000 | 4.000±0.000 |

### CP Methods Comparison: Phi-4 vs LLaVA-Critic (R2CCP)

| Metric | Phi-4 | LLaVA-Critic |
|--------|-------|-------------|
| Coverage (raw) | 0.891 | **0.900** |
| Width (raw) | 3.133 | **3.049** |
| Coverage (adj) | 0.981 | 0.981 |
| Width (adj) | 3.702 | **3.600** |

LLaVA-Critic produces better-calibrated conformal intervals despite being a smaller model (7B vs 15B). This is because LLaVA-Critic's CoT prompt produces less overconfident logprob distributions, giving R2CCP more signal to work with.

---

## 5. Per-Dataset R2CCP

| Dataset | N | Width(raw) | Width(adj) | Pearson | Category |
|---------|---|-----------|-----------|---------|----------|
| AesBench | 392 | **1.975** | **2.633** | 0.353 | Aesthetics/AI |
| mm-vet | 258 | **2.385** | **2.910** | 0.296 | General VQA |
| WIT | 399 | 2.484 | 3.233 | 0.400 | Knowledge/Web |
| coco | 397 | 2.512 | 3.079 | 0.274 | General VQA |
| Concept Caption | 398 | 2.698 | 3.511 | 0.273 | Knowledge/Web |
| mind2web | 398 | 2.744 | 3.299 | 0.224 | Knowledge/Web |
| VisitBench | 397 | 2.900 | 3.443 | 0.339 | General VQA |
| llava_bench | 396 | 3.068 | 3.730 | 0.138 | General VQA |
| textVQA | 399 | 3.100 | 3.516 | 0.213 | Vision-Heavy |
| ScienceQA | 396 | 3.245 | 3.611 | 0.303 | Vision-Heavy |
| diffusiondb | 299 | 3.215 | 3.807 | 0.285 | Aesthetics/AI |
| mathvista | 790 | 3.417 | 3.753 | 0.339 | Vision-Heavy |
| ChartQA | 400 | **3.521** | **3.821** | 0.261 | Vision-Heavy |
| infographicsVQA | 398 | **3.584** | **3.837** | 0.162 | Vision-Heavy |

**Same pattern as LLaVA-Critic:** Aesthetics easiest (1.98), infographicsVQA hardest (3.58). Task-dependent uncertainty is consistent across judges.

### Per-Dataset Width Comparison (Phi-4 vs LLaVA-Critic)

| Dataset | Phi-4 Width(raw) | LLaVA Width(raw) | Difference |
|---------|-----------------|------------------|------------|
| AesBench | **1.975** | 2.082 | -0.107 |
| mm-vet | 2.385 | **2.180** | +0.205 |
| WIT | 2.484 | **2.377** | +0.107 |
| coco | 2.512 | **2.427** | +0.085 |
| Concept Caption | **2.698** | 2.703 | -0.005 |
| mind2web | 2.744 | **2.690** | +0.054 |
| VisitBench | **2.900** | 2.959 | -0.059 |
| llava_bench | 3.068 | **2.920** | +0.148 |
| textVQA | 3.100 | **2.812** | +0.288 |
| ScienceQA | **3.245** | 3.269 | -0.024 |
| diffusiondb | **3.215** | 3.414 | -0.199 |
| mathvista | 3.417 | **3.369** | +0.048 |
| ChartQA | 3.521 | **3.079** | +0.442 |
| infographicsVQA | 3.584 | **3.504** | +0.080 |

Phi-4 is narrower on 5/14 datasets (AesBench, Concept Caption, VisitBench, ScienceQA, diffusiondb) but wider on the rest. Overall, LLaVA-Critic produces slightly narrower intervals.

---

## 6. CP Value Analysis (Error Bins, R2CCP, 10 seeds)

| Error | Cov(raw) | Cov(adjusted) |
|-------|----------|---------------|
| Exact (0) | 99.6%±0.2% | 99.8%±0.1% |
| ±1 | 96.8%±1.5% | 99.9%±0.1% |
| ±2 | 78.8%±3.4% | 99.5%±0.9% |
| ±3 | 34.3%±3.9% | 86.3%±4.6% |
| ±4 | 30.3%±8.0% | 67.4%±10.6% |

**Comparison with LLaVA-Critic:**
- ±1 errors: Phi-4 96.8% vs LLaVA 98.7% raw coverage
- ±2 errors: Phi-4 78.8% vs LLaVA 84.3% raw coverage
- ±3 errors: Phi-4 34.3% vs LLaVA 25.4% raw coverage (Phi-4 better here!)
- After adjustment, both achieve >99% for ±1 and ±2

---

## 7. Per-Dataset ±1 Accuracy

| Dataset | Phi-4 ±1 | LLaVA ±1 | Phi-4 MAE | LLaVA MAE |
|---------|---------|---------|-----------|-----------|
| AesBench | **90.6%** | 88.3% | 0.755 | **0.732** |
| WIT | **89.5%** | 78.7% | **0.659** | 0.912 |
| coco | 84.7% | 84.9% | **0.794** | 0.854 |
| Concept Caption | 82.3% | 81.7% | 0.904 | **0.872** |
| llava_bench | **82.6%** | 81.1% | **0.864** | 0.922 |
| VisitBench | 81.0% | 75.1% | **0.871** | 1.018 |
| diffusiondb | **76.2%** | 57.2% | **0.950** | 1.385 |
| ScienceQA | **72.7%** | 69.7% | **1.093** | 1.199 |
| infographicsVQA | 61.8% | 69.3% | 1.336 | **1.191** |
| ChartQA | 60.9% | **73.2%** | 1.308 | **1.085** |
| mathvista | 67.9% | **65.1%** | **1.175** | 1.234 |

**Key finding:** Phi-4 is notably better on diffusiondb (76.2% vs 57.2% ±1) and WIT (89.5% vs 78.7%), but worse on ChartQA (60.9% vs 73.2%) and infographicsVQA (61.8% vs 69.3%). The reasoning model helps on creative/general tasks but hurts on chart/infographic reading.

---

## Files

| File | Description |
|------|-------------|
| `features_s2.csv` | S2 features (5,717 samples) |
| `point_prediction_comparison.csv` | Phi-4 vs LLaVA-Critic point metrics |
| `confusion_matrix.csv` | 5×5 confusion matrix |
| `error_analysis_per_dataset.csv` | ±1 accuracy and MAE per dataset |
| `r2ccp_per_dataset.csv` | R2CCP width per 14 datasets |
| `error_bins_cp_coverage.csv` | CP coverage per error bin |
| `cp_methods_results.csv` | Key CP methods results (pending CHR/BoostedLCP) |
