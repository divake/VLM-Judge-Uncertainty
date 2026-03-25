# Project Progress Log

## Version History

| Version | Date | Summary |
|---------|------|---------|
| [1.0](#10) | 2026-02-18 | Initial setup, 3-signal MLP plan for ECCV |
| [1.1](#11) | 2026-02-20 | Pilot run (200 samples), all 3 signals + multi-signal ablation |
| [1.2](#12) | 2026-02-20 | Full dataset run (5,713 samples), S2-only |
| [2.0](#20) | 2026-03-18 | Pivot: ECCV → CoLM, drop S1/S3, refocus on extending Sheng et al. to VLMs |

---

## 1.0
**Date:** ~2026-02-18
**Goal:** Build end-to-end VLM judge uncertainty framework targeting ECCV 2026.

**Original plan:** 3-signal MLP fusion for conformal prediction
- Signal 1 (5-dim): Actor VLM generation confidence (per-token entropy stats)
- Signal 2 (5-dim): Judge VLM score-token logprobs for "1"-"5"
- Signal 3 (3-dim): Judge self-uncertainty (entropy, margin, std of score dist)
- Total: 13-dim feature vector → MLP → conformal prediction

**What was built:**
- Full VLM inference pipeline (`src/models/llava.py`) supporting LLaVA-1.5 (HF) and LLaVA-Critic (LLaVA-NeXT)
- Judge inference script with parallel GPU support (`scripts/run_judge.py`)
- 3-signal feature extractor (`src/signals/extractor.py`)
- Conformal prediction runner wrapping R2CCP, CQR, OrdinalAPS (`src/conformal/runner.py`)
- MLLM-Judge dataset loader (5,719 samples, 1-5 scale, 1,373 images)

**Critical bugs fixed:**
- LLaVA-Critic requires `attn_implementation="sdpa"` (not "eager") — eager corrupts sliding window attention → garbage output
- `input_length` offset for LLaVA-NeXT: `actual_input_length = sequences.shape[1] - len(generation.scores)`
- R2CCP index error: float32 rounding edge case → `torch.clamp(indices, 0, len-1)`
- NaN logprobs from degenerate repetition → `if not (lp == lp): lp = -100.0`
- Score token extraction: 3-strategy approach (Score:+digit > keyword+digit > backward scan)

---

## 1.1
**Date:** ~2026-02-20
**Goal:** Pilot run on 200 samples to validate pipeline works end-to-end.

**What was done:**
- Ran LLaVA-Critic-7B as judge on 200 MLLM-Judge samples (using dataset answers, not actor-generated)
- Extracted all 3 signals
- Ran multi-signal ablation (S2, S2+S3, S1+S2, S1+S2+S3) with R2CCP, CQR, OrdinalAPS over 100 seeds

**Results (200 samples, 100 seeds):**

| Method | Signals | Coverage | Width |
|--------|---------|----------|-------|
| R2CCP | S2 only | 0.864±0.077 | 2.514±0.443 |
| R2CCP | S2+S3 | 0.863±0.083 | 2.617±0.513 |
| R2CCP | S1+S2 | 0.870±0.077 | 2.605±0.498 |
| R2CCP | S1+S2+S3 | 0.869±0.070 | 2.676±0.498 |
| CQR | S2 only | 0.968±0.038 | 3.382±0.560 |
| OrdinalAPS | S2 only | 0.654±0.032 | 1.535±0.048 |

**Key findings:**
- Adding S1/S3 to S2 does NOT improve results — coverage/width essentially unchanged
- S3 is a deterministic function of S2, so it adds zero new information
- S1 requires ground truth for actor answers to be meaningful, which we don't have
- **Decision: S1/S3 approach is a dead end**

---

## 1.2
**Date:** ~2026-02-20
**Goal:** Scale to full dataset (all 5,713 samples) with S2-only.

**What was done:**
- Ran LLaVA-Critic-7B judge on all 5,713 MLLM-Judge samples (split across 2 GPUs)
- Disabled `save_full_scores` for memory optimization
- Extracted S2 features, ran R2CCP, CQR, OrdinalAPS over 100 seeds

**Results (5,713 samples, 100 seeds, S2-only):**

Point prediction:
- Pearson=0.380, Spearman=0.314, Kendall=0.270, Accuracy=33.9%, MAE=1.059, RMSE=1.474

Conformal prediction (α=0.10, target=0.90):

| Method | Coverage | Width | Notes |
|--------|----------|-------|-------|
| **R2CCP** | **0.900±0.013** | **3.018±0.092** | Hits 90% target exactly |
| CQR | 0.993±0.007 | 3.904±0.128 | Over-covers; weak features → full-range intervals |
| OrdinalAPS | 0.567±0.007 | 1.410±0.009 | Under-covers badly |

**Analysis of CQR over-coverage:**
CQR's base quantile regressor (GradientBoosting) produces intervals spanning nearly the full [1,5] range because S2 features are weak predictors (Pearson 0.38). The conformal correction can only widen, never narrow. 99.6% of intervals are effectively [1.0, 5.0]. This is not a bug — CQR is the wrong tool when features are weak.

**R2CCP works** because it models the conditional distribution directly via neural network + softmax, which is more suitable for weak-signal settings.

---

## 2.0
**Date:** 2026-03-18
**Major pivot: ECCV → CoLM**

**What changed:**
- Target venue changed from ECCV 2026 to **CoLM** (Conference on Language Modeling)
- Dropped the 3-signal MLP fusion idea entirely (S1/S3 don't help)
- New framing: Extend Sheng et al.'s "Analyzing Uncertainty of LLM-as-a-Judge" framework to **VLM judges on multimodal tasks**

**Reference framework:**
- Paper: Sheng et al. "Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction"
- GitHub: https://github.com/BruceSheng1202/Analyzing_Uncertainty_of_LLM-as-a-Judge
- Local copy: `/ssd_4TB/divake/Analyzing_Uncertainty_of_LLM-as-a-Judge/`

**What Sheng et al. covers:**
- 9 conformal methods (CQR, AsymCQR, CHR, LVD, BoostedCQR, BoostedLCP, R2CCP, OrdinalAPS, OrdinalRC)
- 3 LLM judges (GPT-4o mini, DeepSeek-R1-32B, Qwen2.5-72B)
- Text-only datasets: SummEval (1,600), DialSumm (1,400), ROSCOE subsets (~200 each)
- 2 judge frameworks: G-Eval, SocREval
- Boundary adjustment (continuous → discrete intervals)
- Midpoint scoring (interval midpoint as calibrated estimate)
- 30 seeds, 50/50 cal/test split

**What we have so far:**
- 1 VLM judge (LLaVA-Critic-7B)
- 1 multimodal dataset (MLLM-Judge, 5,713 samples)
- 3/9 methods run (R2CCP, CQR, OrdinalAPS)
- No boundary adjustment applied
- No midpoint analysis
- Core pipeline validated and working end-to-end

**What's needed before expanding:**
- A novel angle/contribution beyond "same thing but VLMs" — user will decide this
- Then: more methods, more judges, more datasets, boundary adjustment, midpoint analysis

**Current status:** Paused, waiting for novel idea before scaling experiments.

---

## 3.0
**Date:** 2026-03-20
**Major progress: CoT prompt + all 9 methods + per-dataset analysis**

### What was done:

**1. New CoT prompt (adapted from MLLM-Judge dataset)**
- Replaced generic scoring prompt with structured CoT: Figure Analysis → Response Evaluation → Score
- Added detailed 5-level rubric from MLLM-Judge paper
- Kept `Score: X` format (simpler than dataset's `[[X]]` for extraction)
- Prompt: `configs/prompts/mllm_judge_cot.yaml`

**2. Lean judge runner (`scripts/run_judge_lean.py`)**
- Single-pass: inference → S2 extraction → CSV row (no intermediate JSON/PT files)
- ~5s per sample per GPU, flush after each row for crash safety
- Ran on 2 GPUs (1 model each), completed 5,717 samples in ~5 hours
- Output: `results/v2/features_s2.csv`

**3. Point prediction improvement (New CoT vs Old generic prompt)**

| Metric | OLD | NEW CoT | Change |
|--------|-----|---------|--------|
| Pearson | 0.380 | **0.402** | +5.8% |
| Spearman | 0.314 | **0.356** | +13.4% |
| Kendall | 0.270 | **0.300** | +11.1% |
| MAE | 1.058 | **1.031** | -2.6% |
| Overconfident (>0.99) | 51.4% | **25.6%** | -50% |

**4. OrdinalAPS bug analysis**
- Discovered why OrdinalAPS under-covered (56.7%): qhat grid capped at 0.999, too low for confident-but-wrong VLM judge
- Also discovered 13-class interpolation is wrong for our integer GT (only uses 5 of 13 classes)
- Fixed: 5-class direct + extended grid (up to 1-1e-8)
- Even with fixes, OrdinalAPS over-covers (~99.9%) because 5 discrete classes are too coarse

**5. All 9 conformal methods (COMPLETE)**

| Method | Cov(raw) | Width(raw) | Cov(adjusted) | Width(adjusted) |
|--------|----------|------------|---------------|-----------------|
| **CHR** | 0.880±0.010 | **2.972±0.069** | 0.965±0.006 | **3.431±0.098** |
| **Boosted LCP** | 0.863±0.012 | **3.018±0.031** | 0.977±0.004 | **3.510±0.044** |
| **R2CCP** | **0.900±0.016** | **3.049±0.097** | 0.981±0.005 | 3.600±0.089 |
| Naive Split CP | 0.895±0.009 | 3.226±0.056 | 0.990±0.003 | 3.781±0.035 |
| LVD | 0.894±0.009 | 3.207±0.112 | 0.988±0.005 | 3.713±0.111 |
| Boosted CQR | 0.878±0.009 | 3.540±0.086 | 0.997±0.002 | 3.932±0.036 |
| CQR | 0.996±0.005 | 3.951±0.061 | 1.000±0.000 | 4.000±0.000 |
| AsymCQR | 0.996±0.005 | 3.951±0.061 | 1.000±0.000 | 4.000±0.000 |
| OrdinalAPS | 0.999±0.000 | 3.993±0.001 | same | same |
| OrdinalRC | FAILED (import issue) | — | — | — |

**6. Boundary adjustment analysis**
- Boundary adjustment (floor lower, ceil upper) is critical for our discrete integer GT
- Increases coverage significantly (e.g., CHR: 88.0% → 96.5%)
- Also increases width (CHR: 2.97 → 3.43)

**7. Dataset comparison: Sheng et al. vs Ours**
- Their GT: average of 3 annotators → 13 fractional values (quasi-continuous)
- Our GT: single annotator → 5 integer values (discrete)
- Their judges: 32B-72B models; ours: 7B model
- Their best Pearson ~0.65; our best ~0.40
- This explains wider intervals (3.0 vs 0.6-2.5)

**8. CLIPScore pilot (negative result)**
- Tested CLIPScore(image, answer) and CLIPScore(image, question) on 5,371 samples
- Pearson with judge error: -0.017 (essentially zero)
- CLIPScore does NOT predict when the VLM judge will be wrong
- Interesting per-dataset findings: aesthetics/natural photos easiest, math/science/AI-generated hardest

**9. Per-dataset R2CCP analysis (COMPLETE, all 14 datasets)**
- Vision-Heavy tasks (math, charts, infographics): Width=3.21 avg
- General VQA/Knowledge tasks: Width=2.59-2.62 avg
- R2CCP beats Naive Split CP on ALL 14 datasets

**10. Error analysis**
- 75.1% of samples within ±1 of human GT (the "32% accuracy" is misleading)
- 3-class accuracy (Bad/Medium/Good) = 58.0%
- Judge has positive bias (+0.38) — overscores bad answers, underscores excellent ones
- All judges in dataset show same ±1 pattern (72-83% within 1 point)

**11. CP Value Analysis (error bins, "CP saved you", conditional coverage)**
- CP recovers 97.8% of judge errors (1,885 out of 1,927 wrong predictions)
- For ±1 errors: CP coverage = 99.9% after boundary adjustment
- For ±2 errors: CP coverage = 99.4%
- For ±3 errors: CP coverage = 91.5%
- GT=1 is weakest (88.9% coverage) — judge overscores bad answers
- ~31% of intervals are "moderate" (width 2-3), useful for decisions
- CP coverage holds across all 14 dataset types (≥96% after adjustment)
- Midpoint does NOT improve over raw judge score (unlike Sheng et al.)

### Key files:
- `configs/prompts/mllm_judge_cot.yaml` — new CoT prompt
- `scripts/run_judge_lean.py` — lean inference runner
- `scripts/run_all_conformal.py` — all 9 methods
- `scripts/run_r2ccp_per_dataset.py` — per-dataset analysis
- `scripts/clip_pilot.py` — CLIPScore analysis (negative result)
- `results/v2/features_s2.csv` — new CoT features (5,717 samples)
- `results/v2/conformal_results_intermediate.md` — results summary

---

---

## 3.1
**Date:** 2026-03-25
**Adding more VLM judges: Phi-4 + Gemini**

### What was done:

**1. Phi-4-reasoning-vision-15B (Microsoft, 15B params)**
- Downloaded model (~29GB) to `models/Phi-4-reasoning-vision-15B/`
- Created separate conda env `phi4_env` (needs transformers>=4.57.1, incompatible with LLaVA)
- Script: `scripts/run_judge_phi4.py`
- Key: Must append `<think>` to force reasoning mode (otherwise outputs 6-token non-reasoning responses)
- Uses `dtype=torch.float16`, `attn_implementation="sdpa"`, `device_map="auto"`
- GPU memory: 28.2GB per model — fits one per 48GB GPU
- Speed: ~25s per sample with reasoning (~300-1100 tokens generated)
- Preliminary results (3,149 samples): Pearson=0.243, Accuracy=34.6%, ±1=76.8%, MAE=0.979
- Less biased than LLaVA-Critic (+0.13 vs +0.38) but more overconfident (45.3% vs 25.6% >0.99)
- Status: RUNNING on 2 GPUs (~8-9 hours remaining)

**2. Gemini 2.5 Flash (Google, closed-source, Vertex AI)**
- Set up Vertex AI authentication: `gcloud auth application-default login`
- Project: `gen-lang-client-0896152357`, Location: `us-central1`
- Uses LiteLLM wrapper: `model="vertex_ai/gemini-2.5-flash"`
- Logprobs via Vertex AI: `logprobs=True, top_logprobs=20` (NOT available on Google AI Studio!)
- Script: `scripts/run_judge_gemini.py` — saves both S2 CSV and full responses CSV
- Speed: ~8-10s per sample, ~$2-3 for full dataset
- Preliminary results (10 samples): Accuracy=30%, very overconfident (90% samples >0.99)
- Status: RUNNING 4 parallel API processes (~5-6 hours remaining)
- Cost estimate: ~$2-3 total

**3. Paper abstract drafted**
- File: `paper/abstract_draft.md`
- Title: "How Reliable Are VLM Judges? Conformal Prediction Reveals Task-Dependent Uncertainty in Multimodal Evaluation"
- Multiple options (A-F), Option F selected for funding agency submission
- Key contribution framing: task-dependent uncertainty + ranking-scoring decoupling

**4. Gemini API learnings**
- Google AI Studio (`generativelanguage.googleapis.com`) does NOT support logprobs on 2.5 models
- Vertex AI (`aiplatform.googleapis.com`) DOES support logprobs via LiteLLM
- gemini-2.0-flash deprecated ("no longer available to new users")
- Need `gcloud auth application-default login` + quota project set
- Reference: TRACER paper (arxiv:2602.11409) from same lab uses same LiteLLM + Vertex AI approach

---

## Next Steps
- [x] Complete all 9 CP methods + boundary adjustment (LLaVA-Critic)
- [x] Per-dataset R2CCP analysis (14 categories)
- [x] Error analysis (confusion matrix, relaxed accuracy, judge bias)
- [x] CP value analysis (error bins, conditional coverage, informativeness)
- [x] Paper abstract drafted
- [x] Phi-4 full run completion + analysis (DONE)
- [x] Gemini full run completion + analysis (DONE)
- [x] Polaris dataset downloaded (8,726 captioning samples, running on LLaVA-Critic)
- [ ] Polaris analysis completion
- [ ] Cross-judge comparison paper table
- [ ] Decide novel contribution direction
- [ ] Paper writing (~8 days to deadline)
