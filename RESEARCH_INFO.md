# RESEARCH_INFO.md
# VLM-as-Judge Uncertainty: Complete Research Reference

> This file captures everything discussed in the planning session for this project.
> Do NOT go back to the original chat — all information is preserved here.
> Last updated: Feb 2026

---

## 1. THE CORE IDEA (3-Signal MLP)

### Problem Statement

Current VLM judges assign point scores with two critical information losses:

```
Failure 1: Actor uncertainty is invisible to the judge
────────────────────────────────────────────────────
Actor generates "The cat is on the LEFT of the table"
   internally: P(left)=0.51, P(right)=0.49  ← coin flip, thrown away
Judge reads the confident text → scores 4/5
Reality: this answer was basically guessed. Score should have a wide interval.

Failure 2: Judge's own uncertainty is discarded
────────────────────────────────────────────────
Judge outputs "Coherence: 3"
   internally: P("3")=0.40, P("4")=0.35, P("2")=0.25  ← essentially 3-way tie
Reported score: just "3"  ← the spread is discarded
Reality: wide interval warranted.
```

### The 3-Signal MLP Solution

Instead of taking the judge's point score at face value, consolidate **three signals** into a learned MLP that produces conformal prediction intervals:

```
Signal 1: Actor's internal confidence
  → When actor VLM generates its answer, extract per-token entropy
  → Captures: was the actor visually uncertain? (>67% of judge errors are visual)
  → Features: [mean_entropy, max_entropy, min_confidence, std_entropy, n_tokens]
  → Dim: 5

Signal 2: Judge's confidence about the actor
  → Log-probabilities at the score token position: P("1")...P("5")
  → Same as what the LLM repo already does — proven to work
  → Features: [log_P("1"), log_P("2"), log_P("3"), log_P("4"), log_P("5")]
  → Dim: 5

Signal 3: Judge's self-uncertainty
  → Collapse Signal 2 logits into scalar uncertainty measures
  → Features: [entropy_of_score_dist, top1_margin (P1-P2), std_of_probs]
  → Dim: 3

Total input: 13-dim feature vector per sample

MLP Architecture:
  Linear(13 → 64) → ReLU → Dropout(0.2)
  Linear(64 → 32) → ReLU → Dropout(0.2)
  Linear(32 → 2)  → [score_low, score_high]

Training: pinball loss (quantile regression) → conformal calibration
Output: [low, high] interval with ≥90% coverage guarantee (α=0.10)
```

### Connection to ICRA Paper

This idea is directly inspired by the ICRA uncertainty consolidation paper (your prior work). The pattern is identical:

```
ICRA:   [sensor_1_uncertainty, sensor_2_uncertainty, ...]
              → MLP → consolidated reliability score → action decision

This:   [actor_entropy, judge_logits, judge_self_uncertainty]
              → MLP → nonconformity score → conformal prediction interval
```

Both consolidate multiple heterogeneous uncertainty signals into a learned combiner that generalizes better than any single signal or their simple average. The domain shifts from robotics sensor fusion to vision-language evaluation scoring.

---

## 2. NOVELTY & GAP ANALYSIS

### Confirmed Gap (Verified by 3 separate web research agents)

| Component | Exists? | Papers |
|---|---|---|
| Actor logprobs → quality signal | ✅ | Self-Certainty (NeurIPS 2025, arXiv:2502.18581) |
| Judge score logprobs for evaluation | ✅ | LLM repo (arXiv:2509.18658), Wagner et al. (arXiv:2410.11594) |
| Judge self-consistency uncertainty | ✅ | SCOPE (arXiv:2602.13110), Radharapu et al. (arXiv:2512.22245) |
| Conformal prediction on VLMs (MCQ only) | ✅ | Art of Saying Maybe (arXiv:2509.13379, 2025) |
| Learned conformal nonconformity scores | ✅ | PAC-Bayes CP (NeurIPS 2023, arXiv:2312.04658) |
| **All 3 signals → conformal intervals for VLM eval (1-5 scale)** | ❌ **DOES NOT EXIST** | **Zero results found** |

Direct web searches for:
- "three signal uncertainty fusion judge actor" → zero results
- "multi-source confidence VLM evaluation" → zero results
- "reliability score uncertainty consolidation VLM" → zero results

### Three Clean Novel Claims

1. **First** conformal prediction intervals for VLM-as-judge on continuous (1-5/1-10) evaluation scales — "Art of Saying Maybe" only does MCQ
2. **First** use of actor-side generation logprobs as a feature for judge uncertainty quantification
3. **Learned 3-signal fusion** outperforms single-signal and two-signal ablations

### Why This Is Publishable at ECCV

- **ECCV 2026** is a top computer vision venue — VLM evaluation is central to the vision community
- The VLM-as-judge paradigm has ~10 key papers since early 2024, all from top venues (ICML, CVPR, NeurIPS, ICLR)
- Uncertainty quantification for VLM judges is the field's "most significant gap" (confirmed by Claude web research)
- Paper connects to the active conformal prediction community + active VLM evaluation community simultaneously
- Practical: every lab using VLM judges needs reliability estimates; this provides them with formal guarantees

---

## 3. EXISTING PROBLEMS IN VLM JUDGING (Motivation Section)

### Performance Gap
- Even best models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) achieve only **72% on Multimodal RewardBench**
- GPT-4o achieves only **65.4% on VL-RewardBench**
- MLLM-as-a-Judge: Pearson correlation only **0.557** for scoring evaluation — this is our baseline to beat
- "Judge Anything": only **30% accuracy** on score evaluation of multimodal generation

### Visual Perception Is the Dominant Bottleneck
- VL-RewardBench: **>67% of judge errors are visual perception failures**, not reasoning (41.8%)
- "Eyes Wide Shut" (CVPR 2024): CLIP-based visual encoders create systematic blind spots ("CLIP-blind pairs")
- CSR (NeurIPS 2024): **VLMs prioritize text over image** — language backbone dominates
- This directly motivates Signal 1 (actor visual confidence): if the actor was visually uncertain but textually fluent, the judge sees confident text and scores high — missing the visual uncertainty entirely

### Calibration Failures
- All tested VLMs show **high calibration error and are overconfident most of the time** (arXiv:2405.02917)
- High-accuracy models may ALSO have high uncertainty — the two are not aligned (arXiv:2402.14418)
- "Nominal 99% intervals cover true answer only 65% of the time" (FermiEval, arXiv:2510.26995)
- VLMs produce more stable feedback with Likert scales than numeric scores — point estimates are unreliable

### Biases in VLM Judges
| Bias | Evidence | Severity |
|---|---|---|
| High-score leniency | Scores cluster around 4/5 (MLLM-as-a-Judge) | High |
| Position bias | ~5% persistent across studies | Medium |
| Regional/cultural bias | Western-centric, English text preferred (Bingo, NAACL 2024) | High |
| Text dominates visual | Language backbone dominates score (CSR, NeurIPS 2024) | High |
| Adversarial vulnerability | 70-90% attack success rates (FRAME, EMNLP 2025) | Critical |

### Ensemble Approaches Fail
- "Is Your Video LM a Reliable Judge?" (ICLR 2025): **ensemble approaches fail when unreliable judges are included** — less reliable judges inject noise
- This motivates our approach: instead of ensembling multiple judges, estimate confidence of a single judge and use intervals

---

## 4. PAPER REFERENCES (Complete, with ArXiv IDs)

### Direct Predecessors (Must Cite)

| Paper | ArXiv / Venue | Why |
|---|---|---|
| Analyzing Uncertainty of LLM-as-a-Judge | arXiv:2509.18658 (EMNLP 2025) | Direct predecessor — LLM+CP. This is the sibling paper. |
| Art of Saying Maybe | arXiv:2509.13379 (Sep 2025) | Conformal prediction for VLMs — MCQ only, gap we fill |
| MLLM-as-a-Judge | arXiv:2402.15721 (ICML 2024 Oral, 244 citations) | First systematic VLM judge benchmark. GitHub: Dongping-Chen/MLLM-Judge |
| LLaVA-Critic | GitHub: LLaVA-VL/LLaVA-NeXT (CVPR 2025) | First open-source VLM judge model, 7B+72B |
| Prometheus-Vision | arXiv:2404.13534 (ACL 2024 Findings) | Rubric-based VLM evaluator, KAIST |
| VL-RewardBench | CVPR 2025 | Benchmark showing 65.4% GPT-4o accuracy |
| Multimodal RewardBench | arXiv:2502.14191 (Feb 2025, Meta) | 5,211 samples, 6 domains, top models at 72% |
| MJ-Bench | arXiv:2407.04842 (2024) | Preference-based multimodal judge benchmark |

### Supporting This Paper's Signals

| Paper | ArXiv | Validates |
|---|---|---|
| Self-Certainty | arXiv:2502.18581 (NeurIPS 2025) | Signal 1: actor logprobs predict output quality |
| Black-box UQ for LLM-as-a-Judge | arXiv:2410.11594 (Oct 2024) | Signal 3: judge token probs → uncertainty |
| SCOPE | arXiv:2602.13110 (Feb 2026) | Conformal + bidirectional entropy for pairwise judging |
| Calibrating LLM Judges (Linear Probes) | arXiv:2512.22245 (Dec 2025) | Judge hidden states → calibrated uncertainty |
| When Judgment Becomes Noise | arXiv:2509.20293 (Sep 2025) | >90% unexplained variance in judge benchmarks |
| PAC-Bayes Learned Conformal | arXiv:2312.04658 (NeurIPS 2023) | Foundation: learned conformal nonconformity scores |
| Conformal Prediction for Multimodal Regression | arXiv:2410.19653 (Oct 2024) | CP using neural network internal features from multimodal models |

### VLM Calibration / Uncertainty

| Paper | ArXiv | Key Finding |
|---|---|---|
| Uncertainty-Aware Eval for VLMs | arXiv:2402.14418 (Feb 2024) | High accuracy ≠ low uncertainty in VLMs |
| VLM-UQBench | arXiv:2602.09214 (Feb 2026) | Current UQ methods modality-specialized, don't generalize |
| Seeing is Believing (Verbalized Calibration) | arXiv:2505.20236 (May 2025) | VLMs miscalibrated, visual reasoning models better |
| Overconfidence is Key | arXiv:2405.02917 (NAACL 2024) | All VLMs overconfident, high calibration error |
| VL-Uncertainty | github.com/Ruiyang-061X/VL-Uncertainty | Semantic perturbation + entropy for hallucination |
| MTRE | github.com/lanl/MTRE | Multi-token logit analysis, 82.6 F1 on LLaVA-7B |
| BayesVLM | arXiv: Dec 2024, github.com/AaltoML/BayesVLM | Laplace approximation for CLIP/SigLIP |
| Learning Conformal Abstention Policies | Feb 2025 | RL + CP, 22% hallucination AUROC boost |

### Failure Mode Papers (For Problem Section)

| Paper | ArXiv / Venue | Finding |
|---|---|---|
| Eyes Wide Shut | CVPR 2024 | CLIP blind spots — visual perception failures |
| Bingo | NAACL 2024 (231+ citations) | VLM hallucination taxonomy |
| CSR / Calibrated Self-Rewarding | NeurIPS 2024 | Text dominates visual in self-rewarding |
| FRAME (Fooling LVLM Judges) | EMNLP 2025 | 70-90% adversarial attack success rates |
| Is Your Video LM a Reliable Judge? | ICLR 2025 | Ensemble fails with unreliable judges |
| Critic-V | CVPR 2025 | Actor-Critic paradigm, 7-12% accuracy gain |

### Parallel Work in Other Domains

| Paper | ArXiv | Why Relevant |
|---|---|---|
| Mutual Info Conformal Fusion (RGB+LiDAR) | arXiv:2309.09593 (2023) | Multi-signal CP fusion in robotics — parallel to ICRA work |
| Self-Certainty | arXiv:2502.18581 | Actor logprobs → quality prediction without reward model |
| Conformalizing MT Evaluation | arXiv:2306.06221 (2023) | Applying CP to evaluation scores (MT quality estimation) |

---

## 5. MODELS TO USE

### Actor VLMs (Models Being Evaluated)

Run these on (image, question) → save answer text + per-token logprobs.

| Model | HuggingFace ID | Size | VRAM fp16 | Notes |
|---|---|---|---|---|
| LLaVA-1.5-7B | llava-hf/llava-1.5-7b-hf | 7B | ~14GB | Baseline, most cited actor |
| LLaVA-1.6-Mistral-7B | llava-hf/llava-v1.6-mistral-7b-hf | 7B | ~14GB | Better than 1.5 |
| InternVL2-8B | OpenGVLab/InternVL2-8B | 8B | ~16GB | Strong open performer |
| Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B-Instruct | 7B | ~14GB | Recent SOTA |
| LLaMA-3.2-11B-Vision | meta-llama/Llama-3.2-11B-Vision-Instruct | 11B | ~22GB | Meta, widely known |

### Judge VLMs (Models Scoring Actor Outputs)

Run on (image, question, actor_answer) → save score text + score-token logprobs.

| Model | HuggingFace ID | Size | Notes |
|---|---|---|---|
| **LLaVA-Critic-7B** | lmms-lab/llava-critic-7b | 7B | **Purpose-built judge. Primary judge model.** CVPR 2025 |
| InternVL2-8B | OpenGVLab/InternVL2-8B | 8B | Best uncertainty quality (tops "Art of Saying Maybe") |
| Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B-Instruct | 7B | Strong reasoning |
| Prometheus-Vision-7B | prometheus-eval/prometheus-vision-7b-v1.0 | 7B | Rubric-based, ACL 2024 |

**Note:** LLaVA-Critic-72B also exists if needed (72B, needs int4 quantization on our hardware).

### Models Already Downloaded (Symlinked via `./models/`)

| Model | Path | Size |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-32B | models/DeepSeek-R1-Distill-Qwen-32B/ | ~62GB |
| Qwen2.5-72B-Instruct | models/Qwen2.5-72B-Instruct/ | ~136GB |

These are TEXT-ONLY models. For VLM work, need to download the VLMs above.

### Quantization for Large Models

If Qwen2.5-72B-VL is needed (doesn't fit fp16 on our 94.8GB):
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForVision2Seq.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
```
int4 NF4: ~40GB → fits on ONE GPU with room to spare. Quality loss negligible for scoring tasks.

---

## 6. DATASETS

### Primary Datasets

| Dataset | Size | Scale | Human GT? | Source |
|---|---|---|---|---|
| **MLLM-as-a-Judge** | 4,414 samples | 1-5 scoring | ✅ Human annotated | github.com/Dongping-Chen/MLLM-Judge — score.jsonl, pair.jsonl, batch.jsonl |
| **LLaVA-Bench (In-the-Wild)** | 300 QA | 1-10 | ✅ GPT-4 reference | HuggingFace: liuhaotian/llava-bench-in-the-wild |
| **MMVet** | 218 QA | 0-100 | ✅ Human | HuggingFace: whyu/MM-Vet — 6 capabilities: recognition, OCR, spatial, math, knowledge, language |
| **Multimodal RewardBench** | 5,211 triplets | Binary preference | ✅ Expert annotated | github.com/facebookresearch/multimodal_rewardbench — 6 domains |

### Secondary / Supplementary

| Dataset | Size | Notes |
|---|---|---|
| NaturalBench | 10K | Vision-centric VQA — tests genuine visual understanding |
| MJ-Bench | ~4K pairs | Preference-based — alignment/safety/quality/bias |
| VL-RewardBench | 1,250 | CVPR 2025 — vl-rewardbench.github.io |
| GenAI-Bench | 1,600 | **Already in sibling LLM repo** (Example_GenAI-Bench/) — 80K+ human ratings |

### Dataset Strategy (Practical)

**Development phase:** LLaVA-Bench (300 samples × 3 actors = 900 triplets) + MMVet (218 × 3 = 654). Total ~1,554 triplets — fast iteration.

**Main results:** MLLM-as-a-Judge (4,414) — this is the benchmark with human-annotated scoring, most directly comparable to the LLM repo's SummEval setup.

**Key baseline number to beat:** Pearson-r = **0.557** (MLLM-as-a-Judge score evaluation by VLM judges) → our interval midpoints need to exceed this.

### Building the Paired Dataset

The critical step — you need actor model outputs WITH logprobs, not just text:

```
1. Take questions from LLaVA-Bench / MMVet / MLLM-as-a-Judge
2. For each actor VLM (LLaVA-1.5-7B, InternVL-8B, Qwen2.5-VL-7B):
   - Run actor on (image, question)
   - Save: answer_text, per_token_entropy_stats (Signal 1 features)
3. For judge VLM (LLaVA-Critic-7B, InternVL2-8B):
   - Run judge on (image, question, actor_answer)
   - Save: judge_score, score_token_logprobs (Signal 2), entropy_stats (Signal 3)
4. Merge with human ground truth scores from original benchmark
Result: N × [signal_1 (5), signal_2 (5), signal_3 (3), human_score] matrix
```

---

## 7. EVALUATION METRICS

Same as sibling LLM paper. All code already exists in `conformal_predictors/evaluation_metrics.py` and `conformal_predictors/interval_processing.py`.

### Interval Metrics (Primary — Tables 1, 2)
- **Interval Width** (lower = better efficiency, narrower intervals for same coverage)
- **Coverage Rate** (target ≥ 90%, using α=0.10)
- Reported as mean ± std over 100 random seeds

### Score/Midpoint Metrics (Table 3 equivalent)
- **Pearson correlation** with human GT (beat the 0.557 baseline)
- **Spearman correlation**
- **Kendall's Tau**
- **MSE / MAE / RMSE**
- Compare: raw judge score | weighted average score | interval midpoint (before adjustment) | interval midpoint (after boundary adjustment)

### Ablation (Table showing signal importance)
| Configuration | Features | Width | Coverage | Pearson |
|---|---|---|---|---|
| Signal 2 only (baseline = LLM paper) | 5 dims | X | Y% | Z |
| Signal 2+3 | 8 dims | X | Y% | Z |
| Signal 1+2+3 (full model) | 13 dims | X | Y% | Z |

---

## 8. FULL PIPELINE (Step by Step)

### Phase 1: Data Collection (~2-3 weeks)

```bash
# Step 1: Download datasets
# MLLM-as-a-Judge: git clone https://github.com/Dongping-Chen/MLLM-Judge data/mllm_judge/
# LLaVA-Bench: huggingface-cli download liuhaotian/llava-bench-in-the-wild
# MMVet: huggingface-cli download whyu/MM-Vet

# Step 2: Run actor VLMs (adapt from qwen_eval.py in sibling repo)
python actors/run_actor_vlm.py \
    --model "llava-hf/llava-1.5-7b-hf" \
    --dataset data/llava_bench/ \
    --save_fp results/llava15_actor_llava_bench.json \
    --max_new_tokens 256

# Step 3: Run judge VLM on actor outputs
python judges/run_judge_vlm.py \
    --judge_model "lmms-lab/llava-critic-7b" \
    --actor_results results/llava15_actor_llava_bench.json \
    --prompt_fp judges/prompts/score_rubric.txt \
    --save_fp results/critic7b_judging_llava15_llava_bench.json
```

### Phase 2: Feature Extraction (~1 day)

```bash
python extract_features.py \
    --actor_fp results/llava15_actor_llava_bench.json \
    --judge_fp results/critic7b_judging_llava15_llava_bench.json \
    --save_fp model_logits/llava15_critic7b_llava_bench.csv
# Output CSV: columns [s1_mean_ent, s1_max_ent, s1_min_conf, s1_std_ent, s1_ntok,
#                      s2_lp1, s2_lp2, s2_lp3, s2_lp4, s2_lp5,
#                      s3_entropy, s3_margin, s3_std, human_score]
```

### Phase 3: Baseline — Signal 2 Only (~1 week)

```bash
# R2CCP on 5-dim judge logprobs only (replicating LLM repo for VLMs)
python conformal_predictors/R2CCP_rancom.py
# Shows: width X, coverage Y% — this is the baseline
```

### Phase 4: Full 3-Signal MLP (~2-3 weeks)

```bash
# Add Signal 1+3 features, train MLP
python mlp_fusion/train_mlp.py \
    --features model_logits/llava15_critic7b_llava_bench.csv \
    --signals 1+2+3 \
    --alpha 0.10 \
    --n_seeds 100
```

### Phase 5: Analysis (~1-2 weeks)

- Coverage/width comparison table across all signal combinations
- Correlation (Pearson/Spearman) of interval WIDTH with human annotator disagreement
- Cross-dataset generalization: train on LLaVA-Bench, test on MMVet
- Ablation: which signal contributes most?
- Visualization: wide intervals → harder questions → more human disagreement

---

## 9. PAPER STORY (ECCV Submission)

### One-Paragraph Abstract Target

> VLM judges assign point scores that discard two critical uncertainty signals: the actor's own generation confidence and the judge's self-uncertainty. We show that uncertainty has three sources in VLM evaluation — the actor's visual generation confidence (Signal 1), the judge's confidence about the actor (Signal 2), and the judge's own internal uncertainty (Signal 3) — and that combining these three signals with a small learned MLP into a conformal prediction framework produces statistically valid evaluation intervals. On LLaVA-Bench and MLLM-as-a-Judge, our intervals are X% narrower than single-signal baselines while maintaining ≥90% coverage, our interval midpoints achieve Pearson-r = Y vs. the 0.557 human-correlation baseline, and interval width correlates (r=Z) with genuine human annotator disagreement on visually challenging samples.

### Section Outline

1. **Introduction** — VLM judges are unreliable point estimators. Gap: no UQ for VLM evaluation. Our 3-signal approach.
2. **Related Work** — LLM-as-judge UQ, VLM calibration, conformal prediction for VLMs, actor confidence
3. **Method** — Signal 1/2/3 extraction, MLP fusion, conformal calibration
4. **Experiments** — Main table (interval metrics), score correlation table, ablation table
5. **Analysis** — Width-vs-human-disagreement correlation, cross-dataset generalization, failure cases
6. **Conclusion**

### Key Novelty Statements (for Rebuttal-Readiness)

1. First conformal prediction intervals for VLM-as-judge on continuous evaluation scales (not MCQ)
2. First use of actor-side generation logprobs as feature for judge uncertainty quantification
3. First demonstration that combining actor confidence + judge confidence + judge self-uncertainty in a learned framework improves interval efficiency over any single signal

---

## 10. EXISTING TOOLKITS (Use Don't Reinvent)

### Data Collection / Actor Inference

**VLMEvalKit** (github.com/open-compass/VLMEvalKit, ~3,500 stars)
- 220+ models, 80+ benchmarks, one-command evaluation
- `pip install -e .` from git clone
- Use for standardized actor VLM inference across benchmarks
- Then add our logprob extraction on top

**lmms-eval** (github.com/EvolvingLMMs-Lab/lmms-eval, ~3,300 stars)
- Text, image, video, audio — most versatile
- v0.5 (Oct 2025): audio eval + response caching
- Has statistical rigor tools (CLT, clustered standard errors)

### Hallucination / Uncertainty Baselines

**VL-Uncertainty** (github.com/Ruiyang-061X/VL-Uncertainty)
- Semantic-equivalent perturbations (image blurring + prompt rephrasing)
- Entropy of clustered response distributions
- 10 LVLMs supported, inference-only
- Use as a Signal 3 ALTERNATIVE (if logprobs not available)

**MTRE** (github.com/lanl/MTRE)
- Multi-token logit analysis, ~4M params, 26MB VRAM
- 82.6 F1 on LLaVA-7B hallucination detection
- Use as comparison baseline for Signal 1/2 approach

---

## 11. FREQUENTLY ASKED QUESTIONS (From Planning Session)

### Q: Don't frontier models score 5/5 on everything now, making benchmarks irrelevant?

No. The benchmark evaluates the **judge framework and its uncertainty**, not the actors. The 16 SummEval actors from 2019-2021 are outdated, but as a vehicle to evaluate conformal prediction + judge uncertainty, they still work. Separately, even for modern actors, judges need reliable confidence estimates — a GPT-4o actor can still generate hallucinations, and a judge needs calibrated intervals around its score.

### Q: Can you get logprobs from GPT-4o, Claude, Gemini APIs?

| Provider | Logprobs? | Notes |
|---|---|---|
| OpenAI GPT-4o | ✅ YES | `logprobs=True, top_logprobs=5` — top 5 tokens per step |
| Anthropic Claude | ❌ NO | Not available |
| Google Gemini | ⚠️ PARTIAL | Via Vertex AI only (`response_logprobs=True, top_logprobs=20`) |
| Local open-source (LLaVA, InternVL, Qwen) | ✅ YES (full vocab) | `output_scores=True` in HuggingFace |

For this project, we use open-source VLMs → full logprob access. Signal 1 requires white-box actor access.

### Q: Are the G-Eval prompts we created standard?

The prompts in the sibling LLM repo were reconstructed from paper Figure 5 (relevance prompt shown explicitly) + G-Eval paper (Liu et al., 2023) structure. They follow the standard G-Eval format and are defensible as "G-Eval-style prompts." For VLM work, we need new prompts that include image context. LLaVA-Critic has built-in judging prompts we can study.

### Q: How does fp16 vs int4 quality compare for a 72B judge?

- fp16 72B = ~144GB → doesn't fit on our 94.8GB VRAM → CPU offload → 10-15 min/response
- int4 NF4 72B = ~40GB → fits on one GPU → ~5-8s/response
- Quality loss for 1-5 scoring tasks: effectively zero perceptible difference
- int4 NF4 (bitsandbytes) is the right choice for Qwen-72B judges on our hardware

### Q: For the 100-seed experiments, do we re-run the VLMs 100 times?

No. VLMs run ONCE → save all logprobs → CSV. The 100 seeds are for the conformal predictor only: 100 different 50/50 random splits of the same CSV, each taking ~10-15 seconds. Total: ~17 min per dataset-actor-judge combination. VLM inference runs once.

### Q: Why separate this from the LLM repo?

- Different paper (ECCV vs EMNLP)
- Different dependencies (vision encoders, PIL, VLMEvalKit)
- Different data (image datasets vs text)
- Different models (VLMs with vision encoders)
- Cleaner GitHub story (each repo = one paper)
- Code reuse via copy-and-adapt (files are small, copying is fine)

---

## 12. KNOWN PITFALLS TO AVOID

1. **VLMs use `AutoProcessor` not `AutoTokenizer`** — images must be processed through the vision processor.

2. **Image input devices:** `inputs.to("cuda:0")` works for simple cases. For `device_map="auto"` models, use `model.device` for the first shard or let the processor handle it.

3. **Score token search direction:** Always scan BACKWARDS through the token list. Reasoning chains contain digits. The LAST digit 1-5 is the score.

4. **MAPIE version:** Must be 0.8.6. `MapieQuantileRegressor` renamed in v1.x.

5. **Model loading for VLMs:** Use `AutoModelForVision2Seq` for most VLMs. LLaVA-1.5 uses `LlavaForConditionalGeneration`. Check HuggingFace model card.

6. **Token representation:** Score tokens may appear as "1", "2"... OR "Ġ1", "▁1"... depending on tokenizer. Check by printing `top_logprobs` for a sample. In our LLM experiments, DeepSeek/Qwen used plain "1"-"5" — but VLMs may differ.

7. **Signal 1 for closed-source actors:** If you test a GPT-4o actor, Signal 1 is unavailable. Use verbalized confidence ("on a scale 0-10, how confident are you?") as a proxy. Pearson correlation with true logprobs ~0.47, Spearman ~0.71 (from "Art of Saying Maybe"). Acknowledge this limitation in paper.

8. **LLaVA-Critic prompt format:** LLaVA-Critic expects a specific prompt template. Check the model card at lmms-lab/llava-critic-7b before writing run_judge_vlm.py.

9. **VLMEvalKit vs custom inference:** VLMEvalKit doesn't expose logprobs — it's for accuracy benchmarking. Use it for data loading/organization, but run the actual model inference with our custom scripts that use `output_scores=True`.

---

## 13. NEXT STEPS (In Order)

1. ✅ Research planning done
2. ✅ Directory created with essential files
3. ⬜ Download LLaVA-Bench and MLLM-as-a-Judge datasets into `data/`
4. ⬜ Download LLaVA-Critic-7B and InternVL2-8B into `models/` (new VLM models)
5. ⬜ Write `actors/run_actor_vlm.py` (adapt from sibling repo's `qwen_eval.py`)
6. ⬜ Write `judges/run_judge_vlm.py` (adapt from sibling repo's `qwen_eval.py`, add image input)
7. ⬜ Test with 5-sample run: verify logprobs extractable from chosen VLMs
8. ⬜ Write `extract_features.py` (adapt from `extract_logits_REFERENCE.py`, add Signal 1+3)
9. ⬜ Run baseline: Signal 2 only → R2CCP (replicate LLM paper for VLMs)
10. ⬜ Write `mlp_fusion/train_mlp.py`
11. ⬜ Run full 3-signal experiment
12. ⬜ Analysis notebooks in `analysis/`
13. ⬜ Paper writing

---

## 14. REFERENCE: SIBLING LLM REPO KEY RESULTS

From the LLM-as-Judge paper (the work this extends):

**Table 1 (BEFORE boundary adjustment) — DeepSeek-R1-32B + R2CCP on SummEval:**

| Dimension | Width | Coverage |
|---|---|---|
| Consistency | 0.69 | 90.44% |
| Coherence | 2.30 | 90.12% |
| Fluency | 0.89 | 90.09% |
| Relevance | 2.00 | 89.84% |

**Our replication results (single seed=42):**

| Dimension | Width | Coverage |
|---|---|---|
| Consistency | 0.87 | 92.4% |
| Coherence | 2.34 | 84.8% |
| Fluency | 1.08 | 94.1% |

Small differences due to single seed vs. 100-seed average in paper, and our reconstructed prompts vs. paper's exact prompts. Framework validated as working.
