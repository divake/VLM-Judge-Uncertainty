# Current Project Status

**Last updated:** 2026-03-27
**Version:** 4.0
**Target venue:** CoLM 2026 (Conference on Language Modeling)
**Paper title:** "How Reliable Are VLM Judges? Conformal Prediction Reveals Task-Dependent Uncertainty in Multimodal Evaluation"
**Timeline:** ~7 days to submission

---

## One-Line Summary

First systematic study of conformal prediction for VLM-as-a-Judge: 3 judges × 2 datasets × 8 CP methods + 3 novelty contributions (Mondrian CP, CP-Reliability Diagnostic, Multi-Judge CP analysis). Task-dependent uncertainty varies 40-70% across 14 visual task types.

---

## What's Complete

### Models (3 VLM Judges)
| Judge | Type | Size | Env | Dataset |
|-------|------|------|-----|---------|
| LLaVA-Critic-7B | Open-source | 7B | env_py311 | MLLM-Judge + Polaris |
| Phi-4-reasoning-vision-15B | Open-source | 15B | phi4_env | MLLM-Judge + Polaris |
| Gemini 2.5 Flash | Closed-source | Unknown | env_py311 + Vertex AI | MLLM-Judge |

### Datasets (2)
| Dataset | Samples | GT | Task | Judges Run |
|---------|---------|-----|------|-----------|
| MLLM-as-a-Judge | 5,717 | 1-5 (1 annotator) | VQA (14 types) | All 3 |
| Polaris | 8,726 | 0-1 (3+ annotators) | Captioning | LLaVA + Phi-4 |

### Analyses Complete
- [x] All 8 CP methods + boundary adjustment (LLaVA-Critic on MLLM-Judge)
- [x] Per-dataset R2CCP (14 categories × 3 judges)
- [x] Error analysis (confusion matrix, relaxed accuracy, judge bias, all judges comparison)
- [x] CP value analysis (error bins, "CP saved you", conditional coverage, informativeness)
- [x] Cross-judge comparison (3 judges on MLLM-Judge)
- [x] Polaris analysis (LLaVA-Critic: 4.5x narrower intervals)
- [x] Phi-4 on Polaris (complete)
- [x] CLIPScore pilot (negative result)

### Novelty Contributions (3)
- [x] **Mondrian CP** — Task-conditional CP: easy tasks 16.6% narrower intervals
- [x] **CP-Reliability Diagnostic** — Ranking-Scoring Gap metric for practitioner guidance
- [x] **Multi-Judge CP** — Negative result: single best judge > feature fusion

### Paper
- [x] Abstract drafted (Option F selected)
- [ ] Paper writing in progress (separate Claude agent)

---

## Key Results (Raw Coverage = α=0.10 target)

### R2CCP on MLLM-Judge (raw / adjusted)
| Judge | Coverage (raw) | Width (raw) | Coverage (adj) | Width (adj) |
|-------|:-:|:-:|:-:|:-:|
| Gemini 2.5 Flash | 0.898 | **2.853** | 0.980 | **3.413** |
| LLaVA-Critic-7B | **0.900** | 3.049 | 0.981 | 3.600 |
| Phi-4-15B | 0.891 | 3.133 | 0.981 | 3.702 |

### R2CCP on Polaris (LLaVA-Critic)
| Dataset | Coverage (raw) | Width (raw) |
|---------|:-:|:-:|
| MLLM-Judge | 0.900 | 3.049 |
| **Polaris** | **0.899** | **0.678** (4.5x narrower) |

---

## All Result Files

### MLLM-Judge (LLaVA-Critic)
`results/v2/` — features_s2.csv, RESULTS_v3.0.md, all_methods_results.csv, r2ccp_per_dataset.csv, confusion_matrix.csv, error_analysis*.csv, analysis1-6*.csv, all_judges_comparison.csv, naive_cp_per_dataset.csv

### MLLM-Judge (Phi-4)
`results/v2_phi4/` — features_s2.csv, RESULTS_phi4.md, point_prediction_comparison.csv, r2ccp_per_dataset.csv, confusion_matrix.csv, error_analysis_per_dataset.csv, error_bins_cp_coverage.csv, cp_methods_results.csv

### MLLM-Judge (Gemini)
`results/v2_gemini/` — features_s2.csv, features_s2_responses.csv, RESULTS_gemini.md, point_prediction_comparison.csv, r2ccp_per_dataset.csv, confusion_matrix.csv, error_analysis_per_dataset.csv, error_bins_cp_coverage.csv, cp_methods_results.csv

### Polaris (LLaVA-Critic)
`results/v2_polaris/` — features_s2.csv, RESULTS_polaris.md, point_prediction.csv, confusion_matrix.csv, r2ccp_results.csv, error_bins_cp_coverage.csv, dataset_comparison.csv

### Polaris (Phi-4)
`results/v2_polaris_phi4/` — features_s2.csv (analysis pending)

### Novelty Contributions
`results/v2_mondrian/` — mondrian_comparison.csv, cp_reliability_diagnostic.csv
`results/v2_multijudge/` — multijudge_comparison.csv
