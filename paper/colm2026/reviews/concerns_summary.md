# Consolidated concerns across all 3 reviewers

Grouped by theme, with reviewer attribution. Use this as the rebuttal scaffolding.

---

## TIER 1 — CRITICAL (must address convincingly to flip borderline → accept)

### C1. Conformal protocol validity — train/calibrate overlap

**Raised by:** YU66 (R1), WoGp (R1)

**The claim:** "Section 3.2 says $\hat{f}$ is trained on the calibration set while the same set is also used to compute nonconformity quantiles" → for R2CCP/CQR/CHR (which learn predictors), this breaks split-conformal validity, so Eq (4) doesn't apply.

**What we need to do:**
1. **Verify in code** what we actually do. Check `conformal_predictors/R2CCP_rancom.py`, our experiment scripts, and the 50/50 split implementation. Is R2CCP trained on the calibration portion, then quantile computed on the same portion? Or do we do a 3-way (train / cal / test) split?
2. If 2-way (cal-only training): we have a real problem. Need to either (a) rerun with a proper train/cal/test split and report new numbers, or (b) argue via cross-conformal / jackknife+ theory.
3. If 3-way already: fix the paper's prose in §3.2 and §4 to make this explicit, including exact sample counts at each stage.

**Severity:** Highest. Both reject/borderline reviewers cite this as a primary reason. Fixing this is the single highest-leverage change.

### C2. Boundary-adjustment numbers contradict the formula

**Raised by:** WoGp (R2)

**The claim:** §3.3 defines adjustment as $[l,u] \to [\lceil l\rceil, \lfloor u\rfloor]$, which can only shrink width on integer scales. But Tables 1 and 2 show adjusted width *larger* than raw (3.05 → 3.60). And coverage rises 0.900 → 0.981, which is consistent with *expanding* intervals.

**What we need to do:**
1. Inspect `conformal_predictors/interval_processing.py` (`range_modification` / `boundary_adjustment`) and see what the function actually does.
2. The most likely truth: we apply the *opposite* convention — expand outward to nearest integer ($[\lfloor l \rfloor, \lceil u \rceil]$) — which would explain both wider intervals and higher coverage. If so, fix §3.3 to match the implementation.
3. Update the explanation in §3.3 with the correct direction and rationale.

**Severity:** High. This is a concrete, falsifiable bug claim. Easy to fix but undermines trust.

---

## TIER 2 — IMPORTANT (need to address; affect interpretation of claims)

### C3. Polaris-vs-MLLM cannot isolate annotation quality

**Raised by:** YU66 (R3), WoGp (R4)

**The claim:** The 4.5× width reduction on Polaris conflates many factors: task type (captioning vs 14-way VQA), aggregation (multi-annotator avg vs single), continuous-vs-integer labels, score distribution shape, etc. The paper's headline claim that "annotation quality drives width" overreaches.

**What we need to do:**
1. **Soften the causal language** in abstract, intro, §5.6, and conclusion: "annotation quality and task structure are *correlated with* / *consistent with* narrower intervals" rather than "primarily driven by."
2. **Propose a controlled ablation** for rebuttal: e.g., simulate single-annotator Polaris by random sampling from the annotator pool and re-run. If width grows toward MLLM-level, that supports the claim partially.
3. Acknowledge the confound explicitly as a limitation.

### C4. Ranking-Scoring Gap underformalized / overclaimed novelty

**Raised by:** YU66 (R4), WoGp (R5)

**The claim:**
- "Previously unrecognized failure mode" is overstated — verbosity bias / position bias / calibration breakdown have been discussed in LLM-judge and recommender literature.
- Eq (7) is ad-hoc with no link to Brier decomposition or ECE.
- Missing parentheses: $|\rho| - (1 - w/(K-1))$ vs $|\rho - (1 - w/(K-1))|$ — the values in Table 17 match the latter.
- No statistical rigor (no CIs / significance tests).
- Could be artificially positive simply because intervals are wide.

**What we need to do:**
1. **Fix the equation typesetting** (parentheses) to match implementation.
2. **Soften novelty claim:** "underexplored in the multimodal-judge setting" or "first explicit characterization in VLM-judge evaluation."
3. **Add citations** to verbosity/position-bias and recommender-system rating-decoupling literature.
4. **Reframe RSG as exploratory diagnostic**, not "fundamental failure mode."
5. **Add CIs** from the 10 seeds (already available).
6. Discuss the limitation that RSG inflates when intervals are wide regardless of ranking quality.

### C5. Interval-width normalization inconsistent across paper

**Raised by:** WoGp (R3)

**The claim:**
- §4 says width range is 0–4 on a 1–5 scale.
- §5.4 reports AesBench width 2.08 = 52% and Infograph 3.50 = 88% (using denominator 4) ✓
- Table 7 reports width 3.05 = 61% (using denominator 5? 3.05/5 = 61%) ✗ inconsistent
- Abstract says "~40%" for aesthetics; with denominator 4 this is 1.6 in width units, not the reported 2.08; with denominator 5 it's 2.0. Neither matches cleanly.

**What we need to do:**
1. Pick **one** convention (recommend: width / 4, since the range *is* 4 on a 1–5 integer scale) and apply globally.
2. Fix Table 7, abstract, and any other percentage normalizations.
3. Verify the abstract's "40–70%" claim numerically against actual data.

---

## TIER 3 — MINOR (easy fixes, helpful for camera-ready)

### C6. Wide intervals look uninformative at instance level

**Raised by:** YU66 (R2)

**Response strategy:** Reframe as a feature, not a bug. Wide intervals *signal* that the task or data is unreliable for absolute scoring — that's the operational guideline (use ranking instead). Cite Polaris (width 0.68 = 14% of scale) as proof the method *can* produce tight intervals when warranted. Strengthen §5.5 / Conclusion language.

### C7. Methodological novelty incremental over text-only CP-for-LLM-judge work

**Raised by:** WoGp (preamble)

**Response strategy:** Position contribution as **empirical**, not methodological — first multimodal stratification, ranking-scoring decoupling identified empirically, task-conditional analysis. This matches our [paper_contribution_framing](paper_contribution_framing.md) memory.

### C8. Task taxonomy is arbitrary

**Raised by:** GBAF (R1)

**Response strategy:** Justify current grouping or adopt established taxonomy (MM-Vet, MMBench, VL-RewardBench's vision-capability axes). Add brief paragraph in §4 or §5.4 explaining the grouping rationale.

### C9. Missing citations for CP methods in §5.1

**Raised by:** GBAF (Q1)

**Response strategy:** Trivial — add citations: CQR (Romano 2019), CHR (Sesia & Romano 2021), Boosted CP (Wang et al. 2024), OrdinalAPS (Romano et al. 2020). LVD (Lin et al. 2021) and R2CCP (Guha et al. 2024) already cited.

---

## Rebuttal priority order

1. **C1 (protocol validity)** — investigate code first, then respond definitively
2. **C2 (boundary adjustment)** — investigate code, fix description, regenerate any wrong tables
3. **C5 (normalization)** — quick fix, recompute percentages globally
4. **C3 (causal claim about annotation quality)** — soften language + propose ablation
5. **C4 (RSG formalization)** — fix equation, soften claim, add CIs, add citations
6. **C8, C9** — easy concessions for camera-ready
7. **C6, C7** — reframe in response prose; no code changes needed

## Code files to audit for C1 and C2

- `conformal_predictors/R2CCP_rancom.py` — does R2CCP train on cal data?
- `conformal_predictors/CHR_random..py`, `CQR_random.py` — same question
- `conformal_predictors/interval_processing.py` — what does `boundary_adjustment` actually do?
- Top-level experiment script that does the 50/50 split — need to confirm it doesn't reuse cal data for training
