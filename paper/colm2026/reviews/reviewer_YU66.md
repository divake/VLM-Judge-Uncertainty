# Reviewer YU66 — Rating: 4 (Ok but not good enough — rejection)

**Confidence:** 3 (fairly confident)
**Submitted:** 12 May 2026, 02:11 (modified 22 May 2026, 08:53)
**Ethics flag:** No
**Headline:** "An empirical study of the application of conformal prediction to VLM-as-a-Judge found interesting but limited insights"

## Summary (from reviewer)

This work adapts conformal prediction to the VLM-as-a-Judge setting. It constructs a five-dimensional feature vector from score-token log-probabilities and evaluates eight conformal methods on two benchmarks: MLLM-as-a-Judge (14 single-annotation tasks) and Polaris (captioning with multiple annotators). The study compares three judge models, LLaVA-Critic-7B, Phi-4-reasoning-vision-15B, and Gemini 2.5 Flash.

Key findings include: R2CCP achieves the strongest overall performance on a five-point Likert scale with around 90% coverage and an average raw interval width of approximately 3.05. Interval width varies substantially across tasks, with narrower intervals on AesBench (2.08) and wider ones on InfographicsVQA (3.50), suggesting that visually intensive reasoning tasks induce greater uncertainty. The study reports a ranking–scoring decoupling phenomenon, where high correlation does not imply tight or reliable score intervals, as illustrated on ChartQA. Intervals shrink by a factor of 4.5 on Polaris relative to MLLM-as-a-Judge. Mondrian CP reduces interval width by about 16.6% on easier tasks. Feature fusion across multiple judges degrades performance.

## Reasons To Accept

- The use of VLMs as evaluators has become widespread, yet systematic reliability quantification remains limited. Introducing conformal prediction into this setting provides a useful perspective for the community.
- The empirical study spans multiple judge models, a diverse set of 14 task categories, and several conformal methods, offering a reasonably comprehensive experimental picture. Observations such as task-dependent interval width, the tendency of judges to over-score low-quality answers and under-score high-quality ones, and the variability across benchmarks are practically informative.
- The ranking–scoring decoupling phenomenon is worth highlighting. Strong ordinal correlation does not guarantee reliable absolute scores, which has direct implications for model selection, data filtering, and automated benchmarking pipelines based on VLM judges.

## Reasons To Reject

### R1. Conformal protocol may invalidate coverage guarantees
> The conformal protocol is insufficiently specified and may invalidate the claimed coverage guarantees. The paper states that $\hat{f}$ is trained on the calibration set while the same set is also used to compute nonconformity quantiles. For methods such as R2CCP, CQR, and CHR that require learning conditional distributions or prediction intervals, the absence of a proper training/calibration split, cross-conformal procedure, or leave-one-out scheme breaks the theoretical guarantees. The paper should clearly describe the data partitioning strategy and report results under a valid protocol.

### R2. Intervals are excessively wide and uninformative at instance level
> The prediction intervals are excessively wide, which limits their practical usefulness. R2CCP yields a raw width of about 3.05 out of 4 and a boundary-adjusted width of about 3.60 out of 4, with several methods approaching the full range ([1,5]) after adjustment. While coverage is achieved, the intervals provide little actionable information at the instance level. This limitation is not sufficiently examined.

### R3. Polaris-vs-MLLM comparison cannot isolate annotation quality
> The narrower intervals observed on Polaris admit multiple explanations, including averaging over annotators, the homogeneous captioning task, smoother score distributions, better alignment between judges and task, lower label noise, or more concentrated targets. A comparison between MLLM-as-a-Judge and Polaris alone does not support the claim that interval width is primarily driven by annotation quality and task structure.

### R4. Ranking-Scoring Gap not systematically analyzed
> The ranking–scoring gap is introduced but not systematically analyzed. The paper does not report the metric with statistical rigor such as confidence intervals or significance tests, and relies mainly on illustrative examples like ChartQA and WIT. As a proposed failure mode, the treatment remains descriptive. The decoupling between ordinal correlation and absolute accuracy has been discussed in prior work on rating models, recommender systems, and LLM-based evaluation, including issues such as verbosity and position bias or calibration breakdown under distribution shift. The claim that this is a previously unrecognized failure mode is overstated. The definition in Equation (7) is also ad hoc, combining $\rho$ and $w$ without a clear theoretical basis or connection to established calibration–discrimination decompositions such as Brier decomposition or expected calibration error.

## Rebuttal notes (draft thinking)

- **R1 (CRITICAL):** Need to clarify in §3.2 / §4 exactly how the calibration split works. If R2CCP is trained on a *subset* of calibration data and the nonconformity quantile is computed on a held-out portion, state this explicitly with sample counts. If we accidentally trained and quantile-fit on the same set, this is a real bug — verify in code.
- **R2:** Reframe wide intervals as *information about the task*, not a flaw of the method. The paper already does this in §5.5 — make this argument more prominent in the rebuttal. Also note Polaris (width 0.68 = 14% of range) demonstrates the method *can* produce tight intervals when data quality permits.
- **R3:** Acknowledge the confound. Could propose an additional ablation that controls for one factor (e.g., re-aggregate MLLM-as-a-Judge ratings if any subset has multiple annotators; or down-sample Polaris to single-annotator).
- **R4:** Soften "previously unrecognized" → "underexplored in the multimodal-judge setting." Cite verbosity/position-bias literature. Add CIs/significance tests to RSG numbers (easy fix from 10 seeds).
