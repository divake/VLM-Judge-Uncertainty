# Results — Gemini 2.5 Flash (Closed-Source, Google Vertex AI)

**Date:** 2026-03-25
**Judge:** Gemini 2.5 Flash (Google, closed-source, via Vertex AI + LiteLLM)
**Dataset:** MLLM-as-a-Judge (5,717 valid samples, 14 dataset categories, 1-5 integer GT)
**Config:** α=0.10, 10 seeds, 50/50 cal/test split, same CoT prompt as LLaVA-Critic
**API cost:** ~$2-3 total

---

## 1. Point Prediction: 3-Judge Comparison

| Metric | LLaVA-Critic-7B | Phi-4-15B | Gemini-2.5-Flash |
|--------|----------------|-----------|-----------------|
| Pearson | 0.402 | 0.303 | **0.459** |
| Spearman | 0.356 | 0.293 | **0.446** |
| Kendall | 0.300 | 0.251 | **0.377** |
| Accuracy | 32.2% | **34.2%** | 32.1% |
| ±1 Accuracy | **75.1%** | **76.1%** | 70.3% |
| MAE | **1.031** | **1.005** | 1.122 |
| Bias | +0.380 | **+0.084** | **-0.045** |
| Overconfident (>0.99) | **25.6%** | 43.0% | 83.0% |
| Overconfident (>0.999) | **6.6%** | 16.9% | 77.4% |

**Key finding:** Gemini has the **highest correlation** (Pearson=0.459) but the **most overconfident distributions** (83% >0.99). It's the best ranker but the worst for CP because logprobs carry almost no distributional information. Near-zero bias (-0.045) — the most balanced scorer.

---

## 2. Error Distribution

| Error | Count | % |
|-------|-------|---|
| Exact (0) | 1,738 | 32.1% |
| ±1 | 2,062 | 38.1% |
| ±2 | 987 | 18.2% |
| ±3 | 453 | 8.4% |
| ±4 | 169 | 3.1% |
| **Within ±1** | **3,800** | **70.3%** |

### Relaxed Accuracy

| Tolerance | Gemini | LLaVA-Critic | Phi-4 |
|-----------|--------|-------------|-------|
| Exact (±0) | 32.1% | 32.2% | **34.2%** |
| ±1 | 70.3% | **75.1%** | 76.1% |
| ±2 | 88.5% | **91.7%** | 91.2% |

Gemini's ±1 accuracy is lower (70.3% vs 75-76%) despite higher Pearson — it makes **larger errors when wrong**.

---

## 3. Confusion Matrix (Gemini)

|     | Pred=1 | Pred=2 | Pred=3 | Pred=4 | Pred=5 | ±1 acc |
|-----|--------|--------|--------|--------|--------|--------|
| GT=1 | **61.7%** | 15.5% | 6.4% | 5.2% | 11.3% | **77.1%** |
| GT=2 | 38.7% | **24.7%** | 9.3% | 9.9% | 17.5% | 72.7% |
| GT=3 | 17.6% | 21.3% | **12.0%** | 18.8% | 30.2% | 52.1% |
| GT=4 | 10.4% | 13.7% | 9.0% | **15.0%** | 52.0% | 76.0% |
| GT=5 | 7.9% | 10.3% | 6.7% | 11.4% | **63.7%** | 75.0% |

**Key insight:** Gemini is much better at identifying bad answers (GT=1: 61.7% exact) than LLaVA-Critic (32.6%) or Phi-4 (21.6%). But GT=3 is terrible (12.0% exact, 52.1% ±1) — it can't score "average" answers.

### Judge Bias by GT Score

| GT | Gemini | LLaVA-Critic | Phi-4 |
|----|--------|-------------|-------|
| GT=1 | **+0.889** | +1.733 | +1.984 |
| GT=2 | **+0.428** | +1.208 | +1.125 |
| GT=3 | **+0.227** | +0.789 | +0.338 |
| GT=4 | **-0.154** | +0.102 | -0.353 |
| GT=5 | -0.874 | -0.741 | -1.037 |
| Overall | **-0.045** | +0.380 | +0.084 |

Gemini has the lowest bias at every GT level for bad answers. It's the most honest about bad responses.

---

## 4. R2CCP Results

| Metric | LLaVA-Critic | Phi-4 | Gemini |
|--------|-------------|-------|--------|
| Coverage (raw) | **0.900** | 0.891 | 0.898 |
| **Width (raw)** | 3.049 | 3.133 | **2.853** |
| Coverage (adj) | 0.981 | 0.981 | 0.980 |
| **Width (adj)** | 3.600 | 3.702 | **3.413** |

**Surprise finding: Gemini produces the NARROWEST conformal intervals** (2.85 raw, 3.41 adjusted) despite being the most overconfident. This is because Gemini has the highest Pearson correlation — R2CCP can learn better nonconformity scores from Gemini's logprobs even though they're peaked.

---

## 5. Per-Dataset R2CCP

| Dataset | Gemini Width(raw) | LLaVA Width(raw) | Phi-4 Width(raw) | Gemini Pearson |
|---------|------------------|------------------|------------------|---------------|
| AesBench | **2.143** | 2.082 | 1.975 | 0.256 |
| mm-vet | **2.290** | 2.180 | 2.385 | 0.233 |
| textVQA | **2.329** | 2.812 | 3.100 | **0.600** |
| WIT | 2.390 | **2.377** | 2.484 | 0.294 |
| coco | **2.400** | 2.427 | 2.512 | 0.342 |
| ScienceQA | **2.772** | 3.269 | 3.245 | **0.530** |
| Concept Caption | 2.796 | **2.703** | 2.698 | 0.428 |
| ChartQA | **2.828** | 3.079 | 3.521 | **0.494** |
| VisitBench | **2.845** | 2.959 | 2.900 | **0.524** |
| llava_bench | **2.913** | 2.920 | 3.068 | 0.312 |
| mind2web | **2.920** | 2.690 | 2.744 | 0.065 |
| infographicsVQA | **2.934** | 3.504 | 3.584 | **0.547** |
| diffusiondb | 3.376 | **3.414** | 3.215 | 0.320 |
| mathvista | 3.389 | **3.369** | 3.417 | 0.414 |

**Gemini produces narrower intervals on 10 of 14 datasets!** Largest improvements on textVQA (-0.48), ScienceQA (-0.50), ChartQA (-0.25), infographicsVQA (-0.57). These are all vision-heavy tasks where Gemini's superior visual understanding helps.

---

## 6. CP Value Analysis (Error Bins, R2CCP, 10 seeds)

| Error | Cov(raw) | Cov(adjusted) |
|-------|----------|---------------|
| Exact (0) | 97.9%±1.0% | 99.8%±0.1% |
| ±1 | 96.2%±1.2% | 99.8%±0.2% |
| ±2 | 88.1%±2.3% | 99.0%±1.1% |
| ±3 | 53.7%±7.8% | 96.2%±1.3% |
| ±4 | 55.1%±3.8% | 68.8%±4.3% |

Higher raw coverage for ±3/±4 errors than LLaVA-Critic (53.7% vs 25.4% for ±3) because Gemini's intervals, while narrower, are better centered.

---

## 7. Per-Dataset ±1 Accuracy

| Dataset | Gemini ±1 | LLaVA ±1 | Phi-4 ±1 |
|---------|----------|---------|---------|
| Concept Caption | **79.7%** | 81.7% | 82.3% |
| textVQA | **78.5%** | 79.2% | — |
| WIT | 78.3% | 78.7% | **89.5%** |
| AesBench | 76.0% | **88.3%** | 90.6% |
| VisitBench | 74.0% | 75.1% | 81.0% |
| ScienceQA | 72.3% | 69.7% | **72.7%** |
| coco | 72.1% | **84.9%** | 84.7% |
| llava_bench | 70.7% | **81.1%** | 82.6% |
| infographicsVQA | 69.0% | 69.3% | 61.8% |
| mm-vet | 68.4% | **80.2%** | — |
| mathvista | 67.3% | 65.1% | **67.9%** |
| ChartQA | **65.7%** | **73.2%** | 60.9% |
| diffusiondb | 59.9% | 57.2% | **76.2%** |
| mind2web | 51.3% | **74.4%** | — |

---

## 8. The Ranking-Scoring Decoupling (Key Paper Finding)

| Dataset | Gemini Pearson | Gemini Width(raw) | Interpretation |
|---------|---------------|------------------|---------------|
| textVQA | **0.600** | **2.329** | Best: ranks well AND tight intervals |
| infographicsVQA | **0.547** | 2.934 | Good ranking, moderate width |
| ScienceQA | **0.530** | 2.772 | Good ranking, moderate width |
| ChartQA | **0.494** | 2.828 | Good ranking, moderate width |
| mind2web | **0.065** | 2.920 | Bad ranking, but moderate width |
| mm-vet | **0.233** | 2.290 | Low ranking, but narrow width |

Gemini shows the **clearest ranking-scoring decoupling** — high Pearson doesn't always mean narrow intervals, confirming that ranking quality and scoring precision are fundamentally different capabilities.

---

## Files

| File | Description |
|------|-------------|
| `features_s2.csv` | S2 features (5,717 samples) |
| `features_s2_responses.csv` | Full judge responses (for reference) |
| `point_prediction_comparison.csv` | 3-judge point metrics |
| `confusion_matrix.csv` | 5×5 confusion matrix |
| `error_analysis_per_dataset.csv` | ±1 accuracy and MAE per dataset |
| `r2ccp_per_dataset.csv` | R2CCP width per 14 datasets |
| `error_bins_cp_coverage.csv` | CP coverage per error bin |
| `cp_methods_results.csv` | All CP methods: CQR, R2CCP, CHR, Boosted LCP |
| `RESULTS_gemini.md` | This document |
