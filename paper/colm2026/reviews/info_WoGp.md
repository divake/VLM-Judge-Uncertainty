# Reviewer WoGp — Complete Information File (Pre-rebuttal Reference)

**Purpose:** Single source of truth for drafting the rebuttal. Captures the reviewer's verbatim critique, our internal verdict on each point, all paper evidence, and the three rebuttal experiments' results. Drafting will pull from this file directly.

**Reviewer details:**
- Title: "a useful empirical study of conformal intervals for VLM judges, but unclear conformal validity"
- Rating: **5** (Marginally below acceptance threshold)
- Confidence: 3 (fairly confident)
- Submitted: 2026-05-09

**Rebuttal goal:** Move WoGp from 5 → 6. This reviewer has identified some real paper bugs and one misread; conceding the real bugs cleanly + showing experimental evidence for the misread should be sufficient.

---

## 1. Reviewer's Summary (verbatim)

> The paper studies whether VLM-as-a-Judge scores can be accompanied by calibrated prediction intervals using conformal prediction over score-token log-probabilities, evaluated on MLLM-as-a-Judge and Polaris with LLaVA-Critic-7B, Phi-4-reasoning-vision-15B, and Gemini 2.5 Flash. The work is timely and potentially significant: Tables 1–3 show that point-score metrics hide substantial task-dependent uncertainty, and the ranking-scoring decoupling analysis in §5.5 is a useful warning against relying only on Pearson/Spearman correlations. However, the paper's methodological contribution is incremental relative to existing conformal LLM-as-a-judge work, which already introduced interval evaluation, ordinal boundary adjustment, and midpoint analysis for text-only judges; the novelty is mainly the multimodal empirical stratification rather than a new conformal framework. More importantly, the paper has unresolved issues around the validity of the conformal protocol, inconsistent descriptions of boundary adjustment, and several numerical inconsistencies in interval-width normalization, which make the main reliability claims harder to trust in their current form.

## 2. Reviewer's Reasons to Accept (verbatim)

1. The paper targets an important failure mode in multimodal evaluation: Table 1 shows that all three VLM judges have low exact agreement with human scores, around 32–34%, and modest correlations, with Pearson (ρ) ranging from .303 to .459, so adding uncertainty estimates is practically motivated rather than cosmetic.
2. The empirical scope is stronger than a single-model case study: §4 evaluates three heterogeneous judges, two datasets, eight conformal methods, and 10 random splits, while Tables 2 and 9 provide a useful comparison showing that R2CCP and CHR dominate degenerate alternatives such as CQR and OrdinalAPS, which often collapse to nearly full-range intervals.
3. The task-level analysis is the paper's most compelling contribution: Table 3 and Figure 1 show that raw R2CCP widths vary from 2.08 on AesBench to 3.50 on InfographicsVQA for LLaVA-Critic, and Table 6 shows wider intervals for Vision-Heavy tasks than Knowledge/Web tasks, supporting the central claim that VLM-judge reliability is task-dependent.

## 3. Reviewer's Reasons to Reject (verbatim, R1–R5)

**R1 (Protocol validity).** The conformal prediction protocol is not specified clearly enough to justify the claimed coverage guarantees: §3.2 says the point predictor (f(x)) is "trained on the calibration set," and §4 says the data are split only 50/50 into calibration and test sets, but standard split conformal validity requires that the model used to produce nonconformity scores be fixed before calibration, or that training/calibration be separated; if R2CCP or other predictors are trained and calibrated on the same examples, the guarantee in Equation 4 does not directly apply.

**R2 (Boundary adjustment formula).** The boundary-adjustment description appears internally inconsistent with the reported results: §3.3 states that the interval ([l,u]) is mapped to ([⌈l⌉,⌊u⌋]), which should weakly shrink intervals on an integer rating scale, yet Tables 1 and 2 report boundary-adjusted intervals that are much wider than raw intervals, e.g. LLaVA-Critic R2CCP width increases from 3.05 to 3.60 and coverage increases from .900 to .981; this suggests either the formula, implementation, or explanation is wrong.

**R3 (Width normalization).** Several quantitative claims about interval width normalization are inconsistent: §4 defines width on a 1–5 scale as ranging from 0 to 4, and §5.4 correctly describes AesBench width 2.08 as 52% of the score range and InfographicsVQA width 3.50 as 88%, but Table 7 describes width 3.05 as 61% rather than 3.05/4 ≈ 76%, and the abstract's "~40%" figure for aesthetics/natural images is not supported by Table 3.

**R4 (Annotation-quality causal claim).** The claim that interval width is "primarily driven by task difficulty and annotation quality" is over-causal relative to the evidence: the Polaris comparison in §5.6 and Table 7 changes multiple factors simultaneously, including task type, score aggregation, number of annotators, label continuity, score distribution, and benchmark construction, so the 4.5× width reduction cannot isolate annotation quality without controlled ablations.

**R5 (RSG diagnostic).** The ranking-scoring gap is interesting but underformalized: Equation 7 is missing necessary parentheses, since the values in Table 17 appear to use RSG(d) = |ρ_d| − (1 − w_d/(K−1)), not the printed (|ρ_d| − 1 − w_d/(K−1)); moreover, the metric is not validated against downstream decisions and can become positive simply because intervals are very wide, so it should be presented as an exploratory diagnostic rather than a "fundamental" failure mode.

---

## 4. Honest Verdict on Each Critique

| # | Critique | Code correct? | Paper correct? | Reviewer right? | Action |
|---|---|---|---|---|---|
| R1 | Protocol validity | YES | NO (misleading wording §3.2 line 118) | partly NO — but their reading is fair given our text | clarify §3.2 + show experimental equivalence (Exp 1) |
| R2 | Boundary formula | YES (expansion) | NO (paper prints shrinkage) | **YES** — they found a real bug | swap formula in §3.3 |
| R3 | Width normalization | n/a | NO (Table 7 uses ÷5, §5.4 uses ÷4, abstract uses ÷5) | **YES** | standardize to ÷(K−1)=÷4 globally |
| R4 | Annotation-quality causal | n/a | partly NO (overcausal language) | partly **YES** | soften "driven by" + run Polaris single-annotator ablation (Exp 2) |
| R5a | RSG parentheses | n/a | YES (equation is well-defined as written) | NO (reviewer misread the substitution) | push back politely with WIT numeric verification + add explicit substituted form |
| R5b | RSG underformalized / "fundamental" overclaim | n/a | partly NO | **YES** | reframe as exploratory diagnostic |
| R5c | No CIs on RSG | n/a | NO | **YES** | add bootstrap 95% CIs (Exp 3) |
| R5d | RSG can be positive when intervals are wide | n/a | n/a | not really a flaw — by design | explain it is the intended behavior |

**Bottom line:** R2, R3, R5b, R5c are real bugs we will fix in writing. R4 is partly real, partly addressable empirically. R1 and R5a are reviewer misreads; we settle them with experimental and arithmetic evidence respectively. **No paper number needs to be recomputed**; all main-table results are valid.

---

## 5. Paper Locations Relevant to Each Critique

These are the exact lines/sections we will edit if the rebuttal commits to a fix. Quoted from `paper/colm2026/main.tex`.

### R1 — Protocol validity (§3.2 line 118)
> "For regression tasks, a common nonconformity score is the absolute residual s(x, y) = |ŷ − y|, where **ŷ = f(x) is a point prediction from a model trained on the calibration set**."

The italicized phrase is the misleading wording. What the code actually does is a three-way nested split: 50% outer cal / 50% test → R2CCP internally splits the 50% input 80/20 → 40% trains the regression network, 10% computes the conformal quantile. So the model used to produce nonconformity scores is **fixed before the quantile-set is seen**, but our paper text obscures this.

### R2 — Boundary adjustment formula (§3.3 line 165)
> "[…] which transforms the continuous interval [l, u] to the ordinal interval **[⌈l⌉, ⌊u⌋]**, ensuring endpoints align with valid rating labels."

This is the shrinkage form. Two lines later, line 168 (cited theorem) correctly states:
> "If at least one boundary is expanded (i.e., l' = ⌊l⌋ or u' = ⌈u⌉), coverage increases."

Code (`scripts/run_all_conformal.py:42-44`):
```python
def boundary_adjust_expand(lows, highs):
    """Expand: floor lower, ceil upper."""
    return np.clip(np.floor(lows), *SCORE_RANGE), np.clip(np.ceil(highs), *SCORE_RANGE)
```

The code does **expansion** [⌊l⌋, ⌈u⌉]. This matches the empirical observation that coverage increases (0.900 → 0.981) and width increases (3.05 → 3.60) after adjustment — which only happens under expansion. The reviewer's diagnosis is exactly correct.

### R3 — Width normalization (multiple locations)
- **§4 line 259** (correct):
  > "Width: the average prediction interval width on the 1–5 scale (**range: 0 to 4**)."

- **§5.4 line 421** (correct, uses ÷4):
  > "[…] AesBench intervals cover **52%** of the score range versus **88%** for InfographicsVQA."

- **Table 7 / §5.6 lines 515, 520** (wrong, uses ÷5):
  > "3.05 (61% of range)" — this is 3.05/5 = 0.61. The correct value under ÷(K−1)=÷4 is 3.05/4 = 0.76 = **76%**.
  > "0.68 (14% of range)" — this is 0.68/5 = 0.136. Correct ÷4: 17%.

- **Abstract line 39** (wrong, uses ÷5 and is selectively phrased):
  > "[…] aesthetics and natural image tasks yield tight intervals covering **~40%** of the score range, while chart reasoning and mathematical figures produce intervals covering **~70%**."

  Actual narrowest task = AesBench 2.082 → 52% (÷4) or 42% (÷5). Actual widest = InfographicsVQA 3.504 → 88% (÷4) or 70% (÷5). The abstract used ÷5 *and* generalized AesBench/InfographicsVQA single-dataset values to whole categories.

### R4 — Causal language about annotation quality (3 locations)
- **Abstract line 39**: "interval width is driven primarily by task difficulty and annotation quality"
- **§1 contribution line 72**: "interval width is driven by task difficulty and annotation quality"
- **Conclusion line 845**: "Interval width reflects task difficulty and annotation quality"

### R5 — RSG (§5.5 lines 190–199, Eq. 7)
- **Eq. 7 (line 193)**:
  > RSG(d) = |ρ_d| − CP-Info(d), where CP-Info(d) = 1 − w_d/(K−1)

- **§1 line 70** ("previously unrecognized failure mode"):
  > "We uncover a ranking-scoring decoupling invisible to standard metrics: judges can achieve high ranking correlation […] while simultaneously producing wide prediction intervals […]"

  This phrasing overstates novelty — decoupling has prior art in recommender systems and LLM-judge bias literature.

---

## 6. Experimental Evidence (3 rebuttal experiments, all complete)

All raw outputs in `results/rebuttal/`. Aggregated tables in `results/rebuttal/REBUTTAL_TABLES.md`. Per-seed CSVs in `results/rebuttal/exp{1,2,3}/per_seed.csv`.

### Exp 1: External 40/10/50 split for R2CCP — addresses R1

**Setup.** Two protocols compared head-to-head on the same data, same seeds, 3 judges × 10 seeds × 2 protocols = 60 R2CCP runs.

- **Protocol A (current paper):** 50/50 outer split. R2CCP internally splits its 50% input 80/20 → 40% trains network, 10% computes quantile. Test on outer 50%.
- **Protocol B (external, strict):** External 40/10/50 split. Pass only the 40% to R2CCP for network training. Compute conformal quantile manually on the disjoint external 10% set (which R2CCP never saw, not even for early-stopping). Test on the 50%.

**Headline result.**

| Judge | Protocol A | Protocol B | Δcov p | Δwidth p |
|---|---|---|---|---|
| LLaVA-Critic | cov=0.893±0.014, w=3.015±0.104 | cov=0.894±0.015, w=3.041±0.095 | 0.772 | 0.093 |
| Phi-4 | cov=0.893±0.009, w=3.147±0.053 | cov=0.890±0.012, w=3.145±0.076 | 0.377 | 0.908 |
| Gemini | cov=0.896±0.012, w=2.857±0.090 | cov=0.897±0.012, w=2.887±0.067 | 0.566 | 0.015 |

- **All coverages are statistically indistinguishable (p > 0.37)**, all hit the 0.90 target.
- Width is indistinguishable for LLaVA-Critic (p=0.093) and Phi-4 (p=0.908). Gemini's width shows a small significant difference (Δ=+0.030, ~1% relative), but it's in the direction expected from training on 40% vs. 50% (less training data → slightly wider intervals). This is *not* contamination; it's the textbook small-sample effect.

**Interpretation for the rebuttal.** Whether we use R2CCP's internal split or hold out a completely disjoint external calibration set, we get the same coverage and (modulo training-set size) the same width. The §3.2 sentence "trained on the calibration set" was loose wording; the protocol itself is valid split conformal. We will clarify §3.2 to spell out the three-way 40/10/50 partition.

**Files.**
- Code: `scripts/rebuttal_exp1_external_split.py`
- Per-seed CSV: `results/rebuttal/exp1/per_seed.csv`
- Summary CSV: `results/rebuttal/exp1/summary.csv`
- Log: `results/rebuttal/logs/exp1.log`

### Exp 2: Polaris single-annotator ablation — addresses R4

**Setup.** Same Polaris features (`results/v2_polaris/features_s2.csv`), same R2CCP config (50/50 split, α=0.10), 10 seeds. Two arms:

- **Multi-annotator (paper baseline):** ground truth = mean of 3–22 annotators, mapped to 1–5 integer.
- **Single-annotator (this ablation):** for each (image, caption) pair, randomly sample ONE annotator's score, map to 1–5 integer.

This holds task type, content, label-continuity scheme, and benchmark identity FIXED. Only annotation aggregation toggles between arms.

**Headline result.**

| Arm | width_raw | cov_raw |
|---|---|---|
| multi-annotator (paper) | **0.717±0.142** | 0.902±0.014 |
| single-annotator (ablation) | **1.001±0.065** | 0.905±0.009 |
| MLLM-Judge (reference, different task) | 3.05 (paper) | 0.900 (paper) |

- Single-annotator widens Polaris by **40% (1.40×)**, paired t-test p = 0.0004.
- The Polaris → MLLM-Judge total width gap is 3.05 − 0.717 = 2.33.
- The annotation-aggregation contribution is 1.001 − 0.717 = 0.284, which is **(0.284 / 2.33) ≈ 12–14% of the gap**.
- The remaining **~86%** is attributable to task type (14-way VQA vs captioning), content distribution, etc.

**Interpretation for the rebuttal.** The reviewer was correct that we cannot causally attribute the gap to "annotation quality" alone. The ablation provides clean *partial* causal evidence: annotation aggregation contributes meaningfully (40% width increase on the same items, p < 0.001) but is *not* the dominant driver. We will rewrite the abstract / §1 / conclusion to say: "annotation aggregation contributes approximately 14% of the gap; task type and content distribution contribute the majority." This is a stronger and more defensible claim than the original "driven by annotation quality."

**Files.**
- Code: `scripts/rebuttal_exp2_polaris_single_annotator.py`
- Per-seed CSV: `results/rebuttal/exp2/per_seed.csv`
- Summary CSV: `results/rebuttal/exp2/summary.csv`
- Log: `results/rebuttal/logs/exp2.log`

### Exp 3: Bootstrap 95% CIs on RSG — addresses R5c

**Setup.** Re-ran per-(judge, dataset) R2CCP for 10 seeds, persisting per-seed values of coverage, width, test-set Pearson ρ, CP-Info, and RSG. Then computed 95% bootstrap CIs (B=2000) over the 10-seed distribution.

3 judges × 14 datasets × 10 seeds = 420 R2CCP runs total. Coverage rows = 42 (judge × dataset).

**Verification of the RSG formula.** Per-seed RSG = |ρ_test| − (1 − w_raw / (K−1)). Matches Eq. 7 of the paper. Spot-check on LLaVA-Critic / WIT:
- Per-seed average: ρ=0.1435, w=2.6248 → RSG = 0.1435 − (1 − 2.6248/4) = 0.1435 − 0.3438 = **−0.2003** ✓
- (Matches the printed value −0.2002 in `results/rebuttal/exp3/per_seed.csv` and the paper's −0.242 in Table 17 to within test-set vs. full-set ρ.)

**Why values differ slightly from paper Table 17.** Paper's Table 17 uses full-dataset Pearson; Exp 3 uses per-seed test-set Pearson (50% of dataset). The latter is required to get a proper seed-to-seed CI; the means are close but not identical. We will explain this in the rebuttal.

**Headline (LLaVA-Critic, the dominant judge in the paper, full table in `results/rebuttal/exp3/rsg_with_95ci.csv`):**

| Dataset | ρ (95% CI) | width (95% CI) | RSG (95% CI) |
|---|---|---|---|
| AesBench | +0.420 [+0.403, +0.442] | 2.031 [1.855, 2.206] | −0.073 [−0.115, −0.034] |
| diffusiondb | +0.088 [+0.060, +0.116] | 3.374 [3.200, 3.553] | −0.067 [−0.121, −0.011] |
| WIT | +0.144 [+0.117, +0.170] | 2.625 [2.523, 2.728] | −0.200 [−0.244, −0.161] |
| MM-Vet | +0.255 [+0.235, +0.279] | 3.074 [2.797, 3.352] | +0.024 [−0.063, +0.097] |
| ChartQA | +0.500 [+0.483, +0.519] | 3.003 [2.876, 3.129] | +0.251 [+0.219, +0.282] |
| InfographicsVQA | +0.427 [+0.404, +0.450] | 3.405 [3.227, 3.564] | +0.278 [+0.237, +0.318] |
| MathVista | +0.383 [+0.371, +0.395] | 3.321 [3.193, 3.439] | +0.213 [+0.183, +0.241] |

(Full table for all 3 judges × 14 datasets in `results/rebuttal/exp3/rsg_with_95ci.csv`.)

**Interpretation.** The CIs are tight — most are 0.05–0.10 wide, demonstrating the seed-to-seed variability is small. The cross-judge consistency of high-RSG datasets (ChartQA, InfographicsVQA, MathVista) is now statistically established, not just a point estimate.

**Files.**
- Code: `scripts/rebuttal_exp3_rsg_bootstrap_ci.py`
- Per-seed CSV: `results/rebuttal/exp3/per_seed.csv`
- CI CSV: `results/rebuttal/exp3/rsg_with_95ci.csv`
- Log: `results/rebuttal/logs/exp3.log`

### Aggregated summary file
`results/rebuttal/REBUTTAL_TABLES.md` — markdown-formatted tables for direct use in the rebuttal.
Generator: `scripts/rebuttal_summarize_all.py` (re-runnable on existing per_seed CSVs).

---

## 7. Width Normalization — Corrected Numbers (from existing data, no new runs)

Paper §4 defines width on the 1–5 scale as ranging from 0 to 4. So the correct normalization is **÷(K−1) = ÷4**. We adopt this throughout the revision.

### Per-dataset (LLaVA-Critic, paper Table 3)

| Dataset | Width | Paper Table 7 (÷5, WRONG) | Corrected (÷4) |
|---|---|---|---|
| AesBench | 2.082 | 42% | **52%** |
| MM-Vet | 2.180 | 44% | **55%** |
| WIT | 2.377 | 48% | **59%** |
| COCO | 2.427 | 49% | **61%** |
| Mind2Web | 2.690 | 54% | **67%** |
| Concept Caption | 2.703 | 54% | **68%** |
| TextVQA | 2.812 | 56% | **70%** |
| LLaVA-Bench | 2.920 | 58% | **73%** |
| VisitBench | 2.959 | 59% | **74%** |
| ChartQA | 3.079 | 62% | **77%** |
| ScienceQA | 3.269 | 65% | **82%** |
| MathVista | 3.369 | 67% | **84%** |
| DiffusionDB | 3.414 | 68% | **85%** |
| InfographicsVQA | 3.504 | 70% | **88%** |

### Headline aggregates

| Slice | width | ÷5 (paper, wrong) | ÷4 (correct) |
|---|---|---|---|
| MLLM-Judge avg | 2.84 | **61%** ← Table 7 | **71%** |
| Polaris | 0.68 | **14%** ← Table 7 | **17%** |
| Narrowest task (AesBench) | 2.08 | 42% | **52%** |
| Widest task (InfographicsVQA) | 3.50 | 70% | **88%** |

### Abstract rewrite (line 39)
- **Before (wrong):** "aesthetics and natural image tasks yield tight intervals covering ~40% of the score range, while chart reasoning and mathematical figures produce intervals covering ~70%"
- **After (proposed):** "interval width varies from 52% to 88% of the score range across 14 task categories — aesthetics and natural-image tasks at the low end, chart and infographic reasoning at the high end"

---

## 8. RSG Parentheses (R5a) — Mathematical Verification

The reviewer claims Eq. 7 reads as `(|ρ| − 1 − w/(K−1))` (no parens around the second term), and that this contradicts Table 17 values. They suggest we need `|ρ| − (1 − w/(K−1))` instead.

**Our equation as written:** `RSG = |ρ| − CP-Info`, with `CP-Info` defined separately on the same line as `1 − w/(K−1)`. When CP-Info is substituted, `A − B` becomes `A − (B_definition)`, requiring parentheses by construction — and the table values match this.

**Numeric proof using WIT (LLaVA-Critic), the dataset cited in the paper Table 17 (RSG = −0.242):**
- Use ρ=0.164, w=2.38, K=5
- **Our formula** (parenthesized as the definition implies): `|0.164| − (1 − 2.38/4)` = `0.164 − 0.405` = **−0.241** ✓ matches Table 17
- **Reviewer's alternative reading** (no parens, distributive): `|0.164| − 1 − 2.38/4` = `0.164 − 1 − 0.595` = **−1.431** ✗ does NOT match Table 17

So Table 17 IS computed with the parenthesized formula `|ρ| − (1 − w/(K−1))`, which is what our equation as written defines. The reviewer's concern is presentational — a casual reader could mentally substitute and drop the parens. We will fix this presentationally by writing the substituted form explicitly on a second line. The math and the numbers are correct.

---

## 9. Action Items for the Paper (camera-ready commits)

When drafting the rebuttal, each of these will be promised by an explicit "we will" verb so the reviewer can verify in the revision.

1. **§3.2 (line 118)** — replace the misleading sentence with the explicit three-way 40/10/50 partition. Cite Sheng et al. (2025) who use the same protocol.
2. **§3.3 (line 165)** — swap the formula from [⌈l⌉, ⌊u⌋] (shrinkage) to [⌊l⌋, ⌈u⌉] (expansion). Align prose with Sheng et al. Theorem 1's expansion branch.
3. **Abstract (line 39)** — replace "~40% / ~70%" with the empirically-supported range "52–88% across 14 task categories."
4. **§1 contribution line 72 + conclusion line 845** — soften "driven by annotation quality" to "annotation aggregation contributes ~14% of the Polaris↔MLLM-Judge gap; task type and content distribution contribute the rest." Add a forward reference to the new Polaris-single-annotator ablation in §5.6.
5. **Table 7** — recompute all "% of range" entries using ÷(K−1)=÷4. New MLLM-Judge avg = 76%, Polaris = 17%.
6. **§5.5 Eq. 7** — add an explicit substituted line `RSG(d) = |ρ_d| − (1 − w_d/(K−1))` to remove any possible reader misread.
7. **§5.5 framing** — reframe RSG as an exploratory diagnostic; cite prior decoupling work in recommender/LLM-judge literature; soften "previously unrecognized failure mode" to "underexplored in multimodal-judge evaluation."
8. **Appendix (new subsection)** — add Polaris single-annotator ablation table (this rebuttal's Exp 2).
9. **Appendix Table 17 (RSG)** — add 95% bootstrap CIs from Exp 3.
10. **§5.6** — add Exp 1's 40/10/50 external-split robustness check as an appendix table; reference it from the §3.2 clarification.

---

## 10. Rebuttal Tone Reminders (from `rebuttal_format_guide.md`)

- **Opening line:** thank for the specific positive they raised — likely "task-level analysis is the paper's most compelling contribution" or "the empirical scope is stronger than a single-model case study."
- **Per-weakness blocks:** label each as `R1:`, `R2:`, etc. — mirror their numbering.
- **Three response modes:**
  - R2, R3, R5b, R5c → concede + fix (real bugs)
  - R1, R5a → clarify a misread + back with evidence
  - R4 → concede language + provide partial-causal ablation
- **Closing each block:** end with "we will [specific action] in [specific section]."
- **Closer:** "we remain available throughout the discussion period…"
- **Never:** argue defensively, confuse the reviewer, or introduce a new theory.
- **Never push back on R2, R3, R5b, R5c.** These are real bugs. Pushing back would destroy our credibility on the points where we ARE right (R1, R5a).
