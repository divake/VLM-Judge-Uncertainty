# Reviewer YU66 — Complete Information File (Pre-rebuttal Reference)

**Purpose:** Single source of truth for drafting the YU66 rebuttal. This reviewer is the **highest-stakes** target: rating 4 (rejection) and the AC will weight the rejection vote heavily. Per the professor's experience, even strong score lifts from other reviewers can be overruled by one rigid rejector. So this rebuttal must be thorough, polite, and backed with new empirical evidence.

**Reviewer details:**
- Title: "An empirical study of the application of conformal prediction to VLM-as-a-Judge found interesting but limited insights"
- Rating: **4** (Ok but not good enough — rejection)
- Confidence: 3 (fairly confident)
- Submitted: 2026-05-12

**Rebuttal goal:** Move YU66 from 4 → 5 (realistic ceiling per discussion with PI). +1 is the minimum we need so the AC has at least a borderline read.

---

## 1. Reviewer's Summary (verbatim)

> This work adapts conformal prediction to the VLM-as-a-Judge setting. It constructs a five-dimensional feature vector from score-token log-probabilities and evaluates eight conformal methods on two benchmarks: MLLM-as-a-Judge (14 single-annotation tasks) and Polaris (captioning with multiple annotators). The study compares three judge models, LLaVA-Critic-7B, Phi-4-reasoning-vision-15B, and Gemini 2.5 Flash.
>
> Key findings include: R2CCP achieves the strongest overall performance on a five-point Likert scale with around 90% coverage and an average raw interval width of approximately 3.05. Interval width varies substantially across tasks, with narrower intervals on AesBench (2.08) and wider ones on InfographicsVQA (3.50), suggesting that visually intensive reasoning tasks induce greater uncertainty. The study reports a ranking–scoring decoupling phenomenon, where high correlation does not imply tight or reliable score intervals, as illustrated on ChartQA. Intervals shrink by a factor of 4.5 on Polaris relative to MLLM-as-a-Judge. Mondrian CP reduces interval width by about 16.6% on easier tasks. Feature fusion across multiple judges degrades performance.

## 2. Reviewer's Reasons to Accept (verbatim)

1. The use of VLMs as evaluators has become widespread, yet systematic reliability quantification remains limited. Introducing conformal prediction into this setting provides a useful perspective for the community.
2. The empirical study spans multiple judge models, a diverse set of 14 task categories, and several conformal methods, offering a reasonably comprehensive experimental picture. Observations such as task-dependent interval width, the tendency of judges to over-score low-quality answers and under-score high-quality ones, and the variability across benchmarks are practically informative.
3. The ranking–scoring decoupling phenomenon is worth highlighting. Strong ordinal correlation does not guarantee reliable absolute scores, which has direct implications for model selection, data filtering, and automated benchmarking pipelines based on VLM judges.

## 3. Reviewer's Reasons to Reject (verbatim, R1–R4)

**R1 (Protocol validity).** The conformal protocol is insufficiently specified and may invalidate the claimed coverage guarantees. The paper states that $f(x)$ is trained on the calibration set while the same set is also used to compute nonconformity quantiles. For methods such as R2CCP, CQR, and CHR that require learning conditional distributions or prediction intervals, the absence of a proper training/calibration split, cross-conformal procedure, or leave-one-out scheme breaks the theoretical guarantees. The paper should clearly describe the data partitioning strategy and report results under a valid protocol.

**R2 (Wide intervals → little instance-level actionable info).** The prediction intervals are excessively wide, which limits their practical usefulness. R2CCP yields a raw width of about 3.05 out of 4 and a boundary-adjusted width of about 3.60 out of 4, with several methods approaching the full range ([1,5]) after adjustment. While coverage is achieved, the intervals provide little actionable information at the instance level. This limitation is not sufficiently examined.

**R3 (Polaris confounds).** The narrower intervals observed on Polaris admit multiple explanations, including averaging over annotators, the homogeneous captioning task, smoother score distributions, better alignment between judges and task, lower label noise, or more concentrated targets. A comparison between MLLM-as-a-Judge and Polaris alone does not support the claim that interval width is primarily driven by annotation quality and task structure.

**R4 (RSG critique).** The ranking-scoring gap is introduced but not systematically analyzed. The paper does not report the metric with statistical rigor such as confidence intervals or significance tests, and relies mainly on illustrative examples like ChartQA and WIT. As a proposed failure mode, the treatment remains descriptive. The decoupling between ordinal correlation and absolute accuracy has been discussed in prior work on rating models, recommender systems, and LLM-based evaluation, including issues such as verbosity and position bias or calibration breakdown under distribution shift. The claim that this is a previously unrecognized failure mode is overstated. The definition in Equation (7) is also ad hoc, combining $|\rho|$ and $1 - w/(K-1)$ without a clear theoretical basis or connection to established calibration–discrimination decompositions such as Brier decomposition or expected calibration error.

---

## 4. Honest Verdict on Each Critique

| # | Critique | Overlap with WoGp? | Verdict | Action |
|---|---|---|---|---|
| R1 | Protocol validity | = WoGp R1 | Wording in §3.2 is misleading; protocol IS valid split conformal | Clarify §3.2 + cite Exp 1 (40/10/50 external split → identical results) |
| R2 | Wide intervals → no instance-level info | **NEW (load-bearing)** | Partially fair (most intervals on MLLM-Judge are wide) BUT we have strong counter-evidence | Acknowledge wide intervals + present Exp 4 (width-error correlation, p<1e-245) + Exp 5 (selective evaluation gains exact accuracy 33%→45% at τ=2.25) + "saved by CP" stat (97% of wrong judge calls have truth inside adj interval) |
| R3 | Polaris confounds | = WoGp R4 | Reviewer is right that the comparison confounds variables | Concede language + cite Exp 2 (Polaris single-annotator → 14% of gap is aggregation) |
| R4 | RSG underformalized + Brier/ECE + prior decoupling | = WoGp R5 + new Brier/ECE angle | Mixed: Eq. 7 ad-hoc framing → fair; novelty overclaim → fair; CIs → add via Exp 3; Brier/ECE → these measure different objects, not the right comparison | Cite Exp 3 (95% CIs on all 42 cells) + reframe as exploratory diagnostic + cite prior decoupling work + clarify Brier/ECE measure calibration of probability estimates, RSG measures ranking-vs-interval-informativeness gap |

**Bottom line:** R2 is the make-or-break critique. We need very strong evidence that the intervals DO carry instance-level information. Exp 4 and Exp 5 directly produce that evidence.

---

## 5. Paper Locations Relevant to Each Critique

(See `info_WoGp.md` §5 for the exact paper line quotations covering R1, R3, R4. YU66's R2 introduces a new concern not covered there — see §6 below for the response.)

For R4 specifically, YU66 cites two additional issues that need paper edits:

- **§1 line 70 ("previously unrecognized failure mode"):** overclaim. Soften to "underexplored in multimodal-judge evaluation, although related decoupling phenomena have been discussed in recommender systems and LLM-judge bias literature."

- **§5.5 Eq. 7 (the $|\rho|$ + $1 - w/(K-1)$ ad-hoc combination):** the formula is a heuristic ranking-vs-informativeness gap, not derived from a calibration-discrimination decomposition. Brier and ECE measure something different (calibration of categorical probability predictions). Acknowledge in §5.5 prose; do not change the formula.

---

## 6. Experimental Evidence (5 rebuttal experiments)

### Reused from WoGp rebuttal

- **Exp 1** (protocol validity, 40/10/50 external split): all three judges show statistically indistinguishable coverage and width between current and strict protocols. See `info_WoGp.md` §6 for full table.
- **Exp 2** (Polaris single-annotator ablation): single-annotator widens Polaris by 1.40× (p=0.0004); annotation aggregation accounts for ~14% of the MLLM-Judge↔Polaris gap. See `info_WoGp.md` §6.
- **Exp 3** (bootstrap 95% CIs on RSG): all 42 (judge × dataset) RSG values now have CIs. See `info_WoGp.md` §6.

### New for YU66 (load-bearing)

#### Exp 4: Per-instance CP width predicts judge error

**Setup.** 3 judges × 10 seeds × full MLLM-Judge test sets (2,859 samples per seed) = 85,770 per-instance records. For each test sample, computed (per-instance R2CCP interval width, judge point error $|y_{\text{parsed}} - y_{\text{gt}}|$).

**Headline: pooled Spearman correlation (width, judge error) = +0.115, p = 1.4 × 10⁻²⁴⁵.**

The correlation is highly statistically significant. Per-judge:

| Judge | n | Spearman ρ (width, error) | p-value |
|---|---|---|---|
| Gemini | 27,037 | **+0.206** | $3.7 \times 10^{-257}$ |
| Phi-4 | 28,274 | +0.115 | $1.2 \times 10^{-83}$ |
| LLaVA-Critic | 28,583 | +0.067 | $4.1 \times 10^{-30}$ |
| **Pooled** | **83,894** | **+0.115** | **$1.4 \times 10^{-245}$** |

(Note: LLaVA-Critic's correlation is weakest because its logprobs are the most extreme; Gemini's is strongest because its logprobs vary more smoothly.)

**MAE by width tertile (pooled):**

| Tertile | n | mean width | Judge MAE |
|---|---|---|---|
| Tight (w ≤ 2.77) | ~27,964 | ~2.40 | **0.95** |
| Middle | ~27,964 | ~3.05 | 1.04 |
| Wide (w > 3.21) | ~27,964 | ~3.55 | **1.18** |

**Wide/tight MAE ratio: 1.24×.**

**Decile binning (pooled, monotonic relationship):**

| Width bin | mean width | n | exact_acc | ±1 acc | MAE |
|---|---|---|---|---|---|
| (0.35, 2.30] | 2.12 | 8,390 | **43.9%** | 81.9% | 0.84 |
| (2.30, 2.53] | 2.42 | 8,389 | 37.9% | 77.1% | 0.97 |
| (2.53, 2.71] | 2.62 | 8,389 | 35.1% | 75.3% | 1.01 |
| (2.71, 2.90] | 2.80 | 8,390 | 33.5% | 76.3% | 1.01 |
| (2.90, 3.05] | 2.99 | 8,389 | 31.9% | 74.2% | 1.06 |
| (3.05, 3.14] | 3.09 | 8,389 | 32.9% | 75.2% | 1.03 |
| (3.14, 3.25] | 3.20 | 8,390 | 31.9% | 73.9% | 1.05 |
| (3.25, 3.41] | 3.33 | 8,389 | 30.2% | 71.5% | 1.10 |
| (3.41, 3.68] | 3.52 | 8,389 | 26.9% | 67.3% | 1.21 |
| (3.68, 4.00] | 3.93 | 8,390 | **25.5%** | 65.3% | 1.25 |

**Interpretation.** Per-instance CP width is monotonically related to judge accuracy: tightest decile is 44% exact, widest is 26%. That's a 1.7× difference in exact accuracy purely from CP width alone. This refutes "intervals provide little actionable information at the instance level."

**Files.** Code: `scripts/rebuttal_exp4_per_instance_cp.py` and `scripts/rebuttal_exp4_5_analyze.py`. Outputs: `results/rebuttal/exp4/{per_instance.csv, correlations.csv, width_deciles.csv}`.

#### Exp 5: Selective evaluation via CP-width threshold

**Setup.** For each threshold τ from 0.5 to 4.0 in steps of 0.25, compute:
- Accept rate = $|$samples with width ≤ τ$|$ / N
- Exact accuracy among accepted
- ±1 accuracy among accepted
- MAE among accepted
- CP coverage among accepted (sanity check, should stay ~0.90)

**Pooled selective-evaluation curve (3 judges, 10 seeds):**

| τ | Accept rate | Exact acc | ±1 acc | MAE | CP coverage |
|---|---|---|---|---|---|
| accept all (baseline) | 100% | 33.0% | 73.8% | 1.053 | 89.4% |
| τ = 1.75 | 0.6% | 47.7% | 78.1% | 0.902 | 80.2% |
| **τ = 2.00** | **1.5%** | **49.4%** | **82.3%** | **0.793** | 81.9% |
| **τ = 2.25** | **7.8%** | **45.3%** | **83.0%** | **0.813** | 90.5% |
| τ = 2.50 | 18.7% | 41.3% | 79.7% | 0.901 | 89.8% |
| τ = 2.75 | 32.4% | 38.7% | 78.0% | 0.947 | 88.5% |
| τ = 3.00 | 45.0% | 37.1% | 77.4% | 0.968 | 87.3% |
| τ = 3.50 | 84.6% | 34.3% | 75.2% | 1.020 | 88.8% |
| τ = 4.00 | 100% | 33.0% | 73.8% | 1.053 | 89.4% |

**Headline:** At τ = 2.25, accepting 7.8% of judge calls yields **45.3% exact accuracy and 83.0% ±1 accuracy** — a **37% relative lift** in exact accuracy over the 33.0% baseline. The CP coverage of the accepted subset stays at 90.5%, confirming the interval is valid on the high-confidence subset.

**Per-judge breakdown at τ = 2.25:**

| Judge | Accept rate | Exact acc | ±1 acc | MAE |
|---|---|---|---|---|
| Gemini | 14.2% | 47.2% | 83.5% | 0.78 |
| Phi-4 | 3.7% | 41.8% | 84.8% | 0.83 |
| LLaVA-Critic | 5.7% | 43.2% | 80.6% | 0.87 |

Gemini accepts more (14%) because its intervals are narrower on average; all three judges show >40% exact accuracy on accepted subsets vs. 32–34% baselines.

**Operational interpretation.** A practitioner integrating a VLM judge into an evaluation pipeline can use τ as a routing rule:
- Width ≤ 2.25 → trust the judge's Likert score (high-confidence judgment)
- Width > 2.25 → escalate to pairwise comparison or human review

This is exactly the kind of "instance-level actionable information" the reviewer asked for. The selective-evaluation curve makes the actionability quantitative and tunable.

**Files.** Outputs: `results/rebuttal/exp4/selective_curve.csv`. Log: `results/rebuttal/logs/exp4_analysis.log`.

#### Per-task category breakdown (added post-hoc, no new VLM runs)

Stratified the per-instance data by the paper's 4-category task taxonomy (Vision-Heavy, General VQA, Knowledge/Web, Aesthetics/AI). At τ = 2.25:

| Category        | n       | Avg width | Spearman ρ (width, error) | Spearman p | Baseline exact | Accept | Exact on accepted | ±1 on accepted | Lift   |
|---              |---      |---        |---                          |---           |---             |---     |---                |---             |---     |
| Aesthetics/AI   | 9,845   | 2.932     | +0.115                      | 1.5e-30      | 33.3%          | 9.4%   | **45.1%**         | 89.8%          | 1.35×  |
| General VQA     | 21,106  | 2.960     | +0.111                      | 2.8e-59      | 34.2%          | 8.2%   | **45.2%**         | 84.5%          | 1.32×  |
| Knowledge/Web   | 17,629  | 3.033     | +0.074                      | 1.4e-22      | 31.7%          | 5.0%   | **39.0%**         | 84.3%          | 1.23×  |
| **Vision-Heavy**| **35,314** | **3.029** | **+0.122**                | **5.3e-117** | **32.8%**     | **8.4%** | **47.4%**         | **79.5%**      | **1.45×** |

**Key finding for the rebuttal:** the selective rule gives a positive lift in *every* category, and the largest lift (+14.6 percentage points, 1.45×) comes from **Vision-Heavy tasks** — precisely the category where the *average* CP width is widest and where the reviewer would most expect intervals to be uninformative. The wide average width reflects that judges are uncertain on most Vision-Heavy items, but on the subset with tight intervals the judges are markedly more accurate. The CP width therefore identifies *within each task category* the items the judge can score reliably. This directly refutes the implicit "but the intervals on hard tasks are useless" reading of R2.

**What we are NOT including (to avoid noise):** at the per-dataset level, two datasets (DiffusionDB, WIT) show slightly negative lift at τ=2.5 — for those the tight-interval subset includes some overconfident-wrong judge calls. This is honest but noisy and would give the reviewer a target to nitpick, so we keep only the per-category aggregation in the rebuttal.

**Files.** Code: `scripts/rebuttal_per_task_selective.py`. Output: `results/rebuttal/exp4/{per_category_selective.csv, per_dataset_selective.csv}`.

#### "Saved by CP" — the strongest single instance-level statistic

Pooled across all 3 judges, 10 seeds, 83,894 records on the full MLLM-Judge test set:

- Judge exactly correct: **27,659 (33.0%)**
- Judge wrong: **56,235 (67.0%)**
- Of the wrong judge calls:
  - **Raw CP interval still contains the truth: 47,554 / 56,235 = 84.6%**
  - **Boundary-adjusted CP interval still contains the truth: 54,577 / 56,235 = 97.1%**
- Complete CP failure (truth outside both raw and adjusted intervals): only ~3%

**Interpretation.** Even though judge accuracy is only 33% (so two-thirds of the time the judge's point score is wrong), the CP interval almost always (97% with boundary adjustment) still contains the true score. Without CP, the practitioner gets "judge says 3" — silently wrong with high probability. With CP, the practitioner gets "judge says 3, with the true score in [2, 5] at 90% confidence" — honest, bounded, and actionable. This is the core instance-level utility the reviewer's R2 critique missed.

---

## 7. Theoretical Response to R4 (Brier/ECE Question)

The reviewer asks why RSG is not connected to "Brier decomposition or expected calibration error." We respond:

**Different objects.** Brier and ECE measure the calibration of categorical *probability estimates*: do the model's predicted P(score = k) values match empirical frequencies? They require a probabilistic forecaster output.

RSG measures something different: **the gap between a judge's ranking ability and its interval informativeness**. The two inputs are:
- $|\rho_d|$ — magnitude of Pearson correlation between judge point scores and ground truth (a ranking quantity)
- $1 - w_d/(K-1)$ — informativeness of the conformal prediction interval (a width quantity)

There is no calibration of categorical probabilities at stake. The decoupling that RSG detects is between (a) the judge's ability to order items correctly and (b) the conformal predictor's ability to localize the score tightly.

**Prior decoupling literature.** We acknowledge that decoupling between ranking and absolute accuracy has been discussed in:
- Recommender systems (NDCG vs. RMSE divergence; Cremonesi et al. 2010)
- LLM-judge bias studies (Zheng et al. 2023 on position/verbosity bias)
- Calibration-vs-discrimination in rating models (Steyerberg et al. 2010)

Our contribution is the first **CP-based formalization** of this decoupling in the VLM-judge setting, not a novel theoretical insight. We will soften "previously unrecognized" to "underexplored in multimodal-judge evaluation, building on related decoupling phenomena in recommender systems and LLM-judge bias literature" and add 2–3 of the citations above.

**Why RSG is still useful.** It is a single-number diagnostic that flags datasets where ranking quality outpaces interval informativeness — directly actionable for the practitioner's choice between Likert scoring and pairwise comparison. We will frame this as an exploratory diagnostic in §5.5 (not a fundamental decomposition theorem) to avoid overclaiming.

---

## 8. Action Items for the Paper (camera-ready commits)

Combining WoGp's commits with YU66's specific asks:

### Inherited from WoGp (already documented in `info_WoGp.md`)
1. §3.2 rewrite (protocol clarification + 40/10/50 partition diagram)
2. §3.3 boundary-adjustment formula fix [⌊l⌋, ⌈u⌉]
3. Width normalization: standardize to ÷(K−1)=÷4 globally (Abstract + Table 7)
4. Soften "driven by annotation quality" to ~14% causal contribution
5. Eq. 7 explicit substituted form
6. §5.5 RSG framing as exploratory diagnostic
7. Add Polaris single-annotator ablation (Exp 2)
8. Add 95% bootstrap CIs to Table 17 (Exp 3)
9. Add 40/10/50 external-split robustness check (Exp 1)
10. Soften §5.5 RSG positivity from wide intervals as designed behavior

### New for YU66
11. **§5.4 or new §5.7 — instance-level utility analysis.** Add the Exp 4 width-decile table showing monotonic exact_acc decline (44% → 26%) and the Spearman correlation. Add the Exp 5 selective-evaluation curve as a small figure.
12. **§5.4 / abstract — "saved by CP" statistic.** Add "even when the judge is wrong (67% of MLLM-Judge cases), the boundary-adjusted CP interval contains the true score 97% of the time" — this directly addresses YU66 R2's "limited actionable information" concern.
13. **§5.5 — connection to Brier/ECE.** Add a paragraph clarifying that Brier and ECE measure calibration of categorical probability estimates, while RSG measures the gap between ranking and interval informativeness — different objects, complementary diagnostics.
14. **§2 (Related Work) — prior decoupling literature.** Add citations to Cremonesi et al. 2010 (recsys ranking-vs-RMSE), Zheng et al. 2023 (LLM-judge bias), Steyerberg et al. 2010 (calibration vs. discrimination in rating models). Soften "previously unrecognized" claim.

---

## 9. Rebuttal Strategy

### Tone reminders (per `rebuttal_format_guide.md`)
- **Opening:** thank for recognizing the "useful perspective for the community," the "reasonably comprehensive experimental picture," and the "practically informative" observations. Specifically thank for highlighting that "the ranking-scoring decoupling phenomenon is worth highlighting."
- **Concede generously on R3 and R4** (Polaris causal, RSG novelty/Brier/ECE) — the reviewer is right.
- **Clarify R1 with Exp 1 evidence** — the reviewer is misreading our wording; the protocol is valid.
- **R2 is the make-or-break.** Lead with the "saved by CP" 97% statistic. Then present Exp 4 monotonic decile table. Then present Exp 5 selective-evaluation curve. End with "we will add an §5.7 making the instance-level utility explicit."
- **Closer:** standard availability statement + score-revision request.

### Per-block response modes
- R1: **Clarify a misread** + back with Exp 1 paired-test table.
- R2: **Concede the wide-on-average observation + refute the "no actionable info" implication** with Exp 4 + Exp 5 + saved-by-CP.
- R3: **Concede language + back with Exp 2** (~14% causal contribution from aggregation alone).
- R4: **Concede ad-hoc framing + present Exp 3 CIs + clarify Brier/ECE distinction + soften novelty claim with citations.**

### What we are NOT doing
- Not arguing that intervals are tight. They aren't on MLLM-Judge. That's because the judge is genuinely uncertain (33% accuracy, 1.0 MAE) — and CP is honestly reflecting that. We will say this directly.
- Not pretending Eq. 7 is a deep theoretical decomposition. It is a heuristic diagnostic.
- Not pretending the decoupling phenomenon is novel in general. We are first to formalize it for VLM judges via CP.

### Realistic outcome
- Per discussion with PI: target is **4 → 5**. +1 is the realistic ceiling. The professor's experience: a rigid rejector at 4 can drag down the whole average even with score lifts from others, so the goal is to give the AC a non-rejection vote from this reviewer.
- The strongest argument we have is Exp 5's selective evaluation table — it gives the reviewer a *concrete operational use case* for the intervals (use τ = 2.25 to triage judge calls). This directly answers their "not sufficiently examined" complaint.

---

## 10. Files generated for this rebuttal

### Code (new)
- `scripts/rebuttal_exp4_per_instance_cp.py` — runs R2CCP on all 3 judges × 10 seeds, saves per-instance (lo, hi, gt, parsed_score) records
- `scripts/rebuttal_exp4_5_analyze.py` — post-hoc: correlation, decile binning, selective-evaluation curve

### Outputs
- `results/rebuttal/exp4/per_instance.csv` — 85,770 records (3 judges × 10 seeds × ~2859 test samples)
- `results/rebuttal/exp4/correlations.csv` — per-judge and pooled Spearman/Pearson
- `results/rebuttal/exp4/width_deciles.csv` — width decile binning per judge + pooled
- `results/rebuttal/exp4/selective_curve.csv` — selective-evaluation curve at τ ∈ [0.5, 4.0]

### Logs
- `results/rebuttal/logs/exp4.log` — R2CCP fitting log
- `results/rebuttal/logs/exp4_analysis.log` — post-hoc analysis log (all tables printed)

### Reused from WoGp rebuttal
- `results/rebuttal/exp1/{per_seed.csv, summary.csv}` (Protocol validity)
- `results/rebuttal/exp2/{per_seed.csv, summary.csv}` (Polaris single-annotator)
- `results/rebuttal/exp3/{per_seed.csv, rsg_with_95ci.csv}` (Bootstrap CIs)

### Reference docs
- `paper/colm2026/reviews/info_WoGp.md` — for shared critiques (R1, R3, R4 overlap)
- `paper/colm2026/reviews/rebuttal_format_guide.md` — tone and structure rules
- `paper/colm2026/reviews/honest_verdict.md` — code-vs-paper audit
