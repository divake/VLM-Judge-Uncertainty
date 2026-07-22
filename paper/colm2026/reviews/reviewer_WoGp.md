# Reviewer WoGp — Rating: 5 (Marginally below acceptance threshold)

**Confidence:** 3 (fairly confident)
**Submitted:** 9 May 2026, 09:56 (modified 22 May 2026, 08:53)
**Ethics flag:** No
**Headline:** "a useful empirical study of conformal intervals for VLM judges, but unclear conformal validity"

## Summary (from reviewer)

The paper studies whether VLM-as-a-Judge scores can be accompanied by calibrated prediction intervals using conformal prediction over score-token log-probabilities, evaluated on MLLM-as-a-Judge and Polaris with LLaVA-Critic-7B, Phi-4-reasoning-vision-15B, and Gemini 2.5 Flash. The work is timely and potentially significant: Tables 1–3 show that point-score metrics hide substantial task-dependent uncertainty, and the ranking-scoring decoupling analysis in §5.5 is a useful warning against relying only on Pearson/Spearman correlations. However, the paper's methodological contribution is incremental relative to existing conformal LLM-as-a-judge work, which already introduced interval evaluation, ordinal boundary adjustment, and midpoint analysis for text-only judges; the novelty is mainly the multimodal empirical stratification rather than a new conformal framework. More importantly, the paper has unresolved issues around the validity of the conformal protocol, inconsistent descriptions of boundary adjustment, and several numerical inconsistencies in interval-width normalization, which make the main reliability claims harder to trust in their current form.

## Reasons To Accept

- The paper targets an important failure mode in multimodal evaluation: Table 1 shows that all three VLM judges have low exact agreement with human scores, around 32–34%, and modest correlations, with Pearson $\rho$ ranging from .303 to .459, so adding uncertainty estimates is practically motivated rather than cosmetic.
- The empirical scope is stronger than a single-model case study: §4 evaluates three heterogeneous judges, two datasets, eight conformal methods, and 10 random splits, while Tables 2 and 9 provide a useful comparison showing that R2CCP and CHR dominate degenerate alternatives such as CQR and OrdinalAPS, which often collapse to nearly full-range intervals.
- The task-level analysis is the paper's most compelling contribution: Table 3 and Figure 1 show that raw R2CCP widths vary from 2.08 on AesBench to 3.50 on InfographicsVQA for LLaVA-Critic, and Table 6 shows wider intervals for Vision-Heavy tasks than Knowledge/Web tasks, supporting the central claim that VLM-judge reliability is task-dependent.

## Reasons To Reject

### R1. Conformal protocol may break coverage guarantee
> The conformal prediction protocol is not specified clearly enough to justify the claimed coverage guarantees: §3.2 says the point predictor $f(x)$ is "trained on the calibration set," and §4 says the data are split only 50/50 into calibration and test sets, but standard split conformal validity requires that the model used to produce nonconformity scores be fixed before calibration, or that training/calibration be separated; if R2CCP or other predictors are trained and calibrated on the same examples, the guarantee in Equation 4 does not directly apply.

### R2. Boundary-adjustment numbers contradict the formula
> The boundary-adjustment description appears internally inconsistent with the reported results: §3.3 states that the interval $[l,u]$ is mapped to $[\lceil l \rceil, \lfloor u \rfloor]$, which should weakly shrink intervals on an integer rating scale, yet Tables 1 and 2 report boundary-adjusted intervals that are much wider than raw intervals, e.g. LLaVA-Critic R2CCP width increases from 3.05 to 3.60 and coverage increases from .900 to .981; this suggests either the formula, implementation, or explanation is wrong.

### R3. Inconsistent width normalization
> Several quantitative claims about interval width normalization are inconsistent: §4 defines width on a 1–5 scale as ranging from 0 to 4, and §5.4 correctly describes AesBench width 2.08 as 52% of the score range and InfographicsVQA width 3.50 as 88%, but Table 7 describes width 3.05 as 61% rather than $3.05/4 = 76\%$, and the abstract's "~40%" figure for aesthetics/natural images is not supported by Table 3.

### R4. Over-causal claim about annotation quality
> The claim that interval width is "primarily driven by task difficulty and annotation quality" is over-causal relative to the evidence: the Polaris comparison in §5.6 and Table 7 changes multiple factors simultaneously, including task type, score aggregation, number of annotators, label continuity, score distribution, and benchmark construction, so the 4.5× width reduction cannot isolate annotation quality without controlled ablations.

### R5. Ranking-Scoring Gap underformalized
> The ranking-scoring gap is interesting but underformalized: Equation 7 is missing necessary parentheses, since the values in Table 17 appear to use $|\rho_d - (1 - w_d/(K-1))|$, not the printed $|\rho_d| - (1 - w_d/(K-1))$; moreover, the metric is not validated against downstream decisions and can become positive simply because intervals are very wide, so it should be presented as an exploratory diagnostic rather than a "fundamental" failure mode.

## Rebuttal notes (draft thinking)

- **R1 (CRITICAL — same as YU66 R1):** Clarify the protocol. If we currently train and calibrate on the same set, this is the most damaging issue. Two options: (a) restructure code to use a train/calibrate/test 3-way split and re-run, (b) cite cross-conformal or jackknife+ literature if applicable. Verify what we actually do in code.
- **R2 (CRITICAL):** This is concrete and easily falsifiable. The formula in §3.3 *does* shrink intervals (ceil(l) ≥ l, floor(u) ≤ u, so u'-l' ≤ u-l). But our tables show width *increasing* after adjustment. Either (a) we report width-on-the-discrete-grid (counting integers in [⌈l⌉, ⌊u⌋] which is ⌊u⌋-⌈l⌉+1, not ⌊u⌋-⌈l⌉), or (b) the boundary adjustment we actually apply *expands* to the nearest integer rather than shrinking, or (c) there's a bug. Need to check the code in `conformal_predictors/interval_processing.py` and the actual implementation in our R2CCP script.
- **R3:** Genuine inconsistencies. The "~40%" in abstract probably refers to width covering ~40% of the *plausible* score range for clean tasks, but the abstract should be precise. Table 7 says width 3.05 = 61% — that's 3.05/5, treating the full range as 5 points, not 4 — we have an off-by-one ambiguity between (max-min) = 4 and the number of categories = 5. Need to pick one convention and use it everywhere.
- **R4 (same as YU66 R3):** Same response as YU66.
- **R5 (same as YU66 R4):** Add parentheses to Eq (7) — minor LaTeX fix. Soften the "fundamental failure mode" claim. Add stats to RSG.

## Critical bugs flagged that need code verification

1. **Whether R2CCP is trained on calibration data** — `conformal_predictors/R2CCP_rancom.py`
2. **Boundary adjustment direction (shrink vs expand)** — `conformal_predictors/interval_processing.py`
3. **Width normalization in Table 7** — denominator should be 4 or 5?
