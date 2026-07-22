# Honest verdict on each reviewer critique

After auditing our code, the R2CCP source code, and the reference paper (Sheng et al. 2025 = `Analyzing_Uncertainty_of_LLM-as-a-Judge`), here is the honest verdict for each critique. **"Code: correct / Paper: wrong"** means we did the right thing in code but described it incorrectly in writing.

---

## C1. "Conformal protocol may break coverage guarantee" — YU66 R1 + WoGp R1

**Verdict: Reviewers are WRONG, but our paper wording IS misleading. Code is correct.**

### The reviewers' claim
"Section 3.2 says $\hat{f}$ is trained on the calibration set while the same set is also used to compute nonconformity quantiles" → split conformal validity broken.

### What our paper says (main.tex line 118)
> "$\hat{y} = f(\mathbf{x})$ is a point prediction from a model trained on the calibration set."

This is genuinely misleading wording.

### What the code actually does
Three-way nested split:

```
Total data
   ↓ outer split (src/conformal/runner.py:149)
50% "calibration" (X_cal)           50% test (X_test) — held out
   ↓ inner split inside R2CCP.fit()
80% inner train (40% of total)     20% inner cal (10% of total)
   → trains neural network        → computes conformal quantile
```

**The data used to compute the quantile (inner 10%) is DISJOINT from the data used to train R2CCP's network (inner 40%), and both are disjoint from the test set (50%).** This is a valid split-conformal protocol.

Evidence:
- `src/conformal/runner.py` line 149: `train_test_split(X_np, y_np, test_size=0.5)` — outer split
- `R2CCP/main.py` line 100: `train_loader, cal_loader = get_loaders(...)` — inner split
- `R2CCP/data.py` line 6: `train_test_split(X, y, test_size=args.test_size)` — inner split implementation
- `R2CCP/argparser.py` line 49: `--test_size` default = `0.2` (so 20% of what we pass is the inner cal set)
- `R2CCP/main.py` line 122: `get_intervals` uses `X_cal, y_cal` (the inner cal portion) to build the quantile, NOT what we passed in originally.

The reference paper Sheng et al. uses the exact same protocol (their §4.1: "50% calibration set and 50% test set with 30 random seeds"), and their paper was peer-reviewed without this objection — because R2CCP's internal split makes the protocol valid.

### Rebuttal action
- Code is fine. Don't change.
- **Fix paper §3.2 line 118**: replace "a model trained on the calibration set" with "a model trained on a held-out training split of the calibration data (R2CCP internally splits its input 80/20: 80% trains the regression-as-classification network; the remaining 20% computes the conformal quantile)."
- Add to §4 / Appendix the explicit three-way data partition (40% inner-train, 10% inner-cal, 50% test).

---

## C2. "Boundary-adjustment description contradicts the reported numbers" — WoGp R2

**Verdict: Reviewer is 100% CORRECT. Paper has a formula bug; code is right.**

### The reviewer's claim
§3.3 says interval maps to $[\lceil l\rceil, \lfloor u\rfloor]$ (shrinkage), yet the tables show widths *increasing* (3.05 → 3.60) and coverage *increasing* (0.900 → 0.981). The formula and the numbers don't match.

### What our paper says (main.tex line 165)
> "transforms the continuous interval $[l, u]$ to the ordinal interval $[\lceil l \rceil, \lfloor u \rfloor]$"

This is shrinkage (ceil-lower, floor-upper). But:
- It contradicts the very theorem cited two lines later (line 168: "If at least one boundary is expanded (i.e., $l' = \lfloor l \rfloor$ or $u' = \lceil u \rceil$), coverage increases").
- It contradicts our own tables (coverage goes up after adjustment, which only happens when intervals expand).

### What the code actually does (`scripts/run_all_conformal.py` lines 42–44)
```python
def boundary_adjust_expand(lows, highs):
    """Expand: floor lower, ceil upper."""
    return np.clip(np.floor(lows), *SCORE_RANGE), np.clip(np.ceil(highs), *SCORE_RANGE)
```

**Code does expansion: $[\lfloor l \rfloor, \lceil u \rceil]$.** This matches the reference paper Sheng et al. Eq (7) exactly.

### Origin of the bug
We took the formula from §3.3 of Sheng et al. — but looking again at their Eq (7): "$\{a : s'(z_{\text{test}}, a) \leq \hat{q}\} = [l, u] \to [l', u']$, where $l' = \lceil l \rceil$ and $u' = \lfloor u \rfloor$."

Wait, **Sheng's Eq (7) itself uses $[\lceil l \rceil, \lfloor u \rfloor]$** (shrinkage), and Sheng's prose immediately after says: "We shrink the boundaries to integer labels closest to the original continuous-valued boundaries by cutting excessive areas... On the other hand, we can also expand an interval to mitigate the marginal miscoverage... For example, assuming the interval [2.2, 3.9] only covers one possible rating 3 but can be expanded to [2, 4]."

So the reference paper acknowledges BOTH directions exist. Their *Eq (7)* is the shrinkage version, but Theorem 1 covers both shrinkage (coverage preserved) and expansion (coverage strictly increases).

Our code uses **expansion**, which is the variant Sheng's Theorem 1 says strictly increases coverage. Our paper §3.3 copied Sheng's Eq (7) (the shrinkage form) without noticing it doesn't match what we actually compute.

### Rebuttal action
- Code is fine. Don't change.
- **Fix paper §3.3 line 165**: change the formula to $[\lfloor l \rfloor, \lceil u \rceil]$ (expansion). Update surrounding prose to say we apply the expansion variant from Sheng et al., which by their Theorem 1 strictly increases coverage. This makes the formula, the cited theorem, and the empirical tables all consistent.

---

## C3. "Polaris-vs-MLLM comparison cannot isolate annotation quality" — YU66 R3 + WoGp R4

**Verdict: Reviewers are CORRECT. This is a fair critique of our causal language.**

The 4.5× width reduction confounds several variables simultaneously: task type (14-way VQA vs captioning), aggregation (single-annotator integer vs multi-annotator continuous mean), score distribution shape, benchmark construction. We cannot causally attribute the gap to "annotation quality" alone from this single comparison.

### Where the overclaim appears
- Abstract line 39: "interval width is driven primarily by task difficulty and annotation quality"
- §1 contribution line 72: "interval width is driven by task difficulty and annotation quality, not the conformal method"
- Conclusion line 845: "Interval width reflects task difficulty and annotation quality"

### Rebuttal action
- Soften causal language: replace "driven by" / "primarily driven by" with "consistent with" / "correlates with" / "is sensitive to".
- Acknowledge the confound as a limitation in §5.6.
- Propose an additional ablation for rebuttal: simulate single-annotator Polaris by sampling one rater per item (we have access to the per-annotator scores in Polaris). If width grows toward MLLM-level, that gives partial causal support.

---

## C4. "Ranking-Scoring Gap is underformalized and over-claimed" — YU66 R4 + WoGp R5

**Verdict: Mixed. Some reviewer critiques are CORRECT, one is WRONG.**

### Sub-critique 1 (WoGp): "Equation 7 missing parentheses"
**Reviewer is WRONG.** Our paper line 193:
$$\text{RSG}(d) = |\rho_d| - \text{CP-Info}(d), \quad \text{CP-Info}(d) = 1 - \frac{w_d}{K-1}$$

Substituted: $\text{RSG} = |\rho_d| - \left(1 - \frac{w_d}{K-1}\right)$ — this matches the reported numbers, which include negative values (WIT: −0.242).

Verification with WIT (ρ=0.164, w=2.38, K=5):
- Our formula: |0.164| − (1 − 2.38/4) = 0.164 − 0.405 = **−0.241** ✓ (matches reported −0.242)
- WoGp's interpretation (outer abs): |0.164 − 0.405| = +0.241 — would lose the sign and contradict the table.

WoGp got this one wrong; the parens are fine. We can address this directly in the rebuttal.

### Sub-critique 2 (YU66 + WoGp): "Overclaims novelty as previously-unrecognized failure mode"
**Reviewers are CORRECT.** Ranking-scoring decoupling has been discussed in recommender systems, LLM-judge bias (verbosity/position), and rating model literature. We should soften to "underexplored in multimodal-judge evaluation" or "first formalized via a CP-based diagnostic in the VLM-judge setting."

### Sub-critique 3 (YU66 + WoGp): "Eq (7) is ad-hoc, no link to Brier / ECE; could be artificially positive just because intervals are wide; no CIs"
**Reviewers are partially CORRECT.** The metric is a heuristic, not derived from a calibration-discrimination decomposition. We can:
- Acknowledge it as exploratory diagnostic, not "fundamental" framework.
- Add 10-seed CIs (data already available).
- Discuss limitation: yes, RSG inflates trivially when intervals approach full range — but this is the desired behavior (it warns the user that the task is unreliable for scoring). Make this design choice explicit.

### Rebuttal action
- Push back on WoGp's parentheses claim (cite the WIT value).
- Concede novelty overstatement; soften language.
- Add CIs; cite prior decoupling literature; reframe as diagnostic, not framework.

---

## C5. "Inconsistent width-normalization across paper" — WoGp R3

**Verdict: Reviewer is CORRECT. Paper has real inconsistencies.**

### What our paper actually uses

| Location | Statement | Implicit denominator |
|----------|-----------|----------------------|
| Abstract (line 39) | "~40% / ~70% of the score range" | unclear; doesn't match either ÷4 or ÷5 numerics |
| §5.4 (line 421) | "AesBench width 2.08 = 52% / InfographicsVQA 3.50 = 88%" | **÷4** (correct since max width on 1–5 scale is 4) |
| Table 7 / §5.6 (lines 515, 520) | "3.05 (61% of range)" | **÷5** (3.05/5 = 0.61) ✗ inconsistent |

### Verification against actual data
- AesBench: 2.08/4 = **52%** ✓ (§5.4 correct)
- InfographicsVQA: 3.50/4 = **88%** ✓ (§5.4 correct)
- MLLM-Judge avg R2CCP width: 3.05/4 = **76%**, NOT 61% (Table 7 wrong; should be 76%)
- Polaris: 0.68/4 = **17%**, NOT 14% (Table 7 wrong; should be 17%)
- Abstract "~40%": even the narrowest task (AesBench) is 52% with the correct ÷4 convention. There is NO task at ~40%. The abstract claim is unsupported.
- Abstract "~70%": ChartQA 77%, ScienceQA 82%, MathVista 84%, InfographicsVQA 88% — the actual range for "chart and math reasoning" is 77–88%, not ~70%.

### Rebuttal action
- **Pick one convention** (recommend: ÷(K−1) = ÷4, since width *can* range from 0 to 4 on a 1–5 integer scale) and apply globally.
- Fix Table 7: 3.05 → 76% (not 61%), 0.68 → 17% (not 14%).
- Fix abstract: "tight intervals covering ~50–60% of the score range / wide intervals covering ~75–90%" matches the actual data.

---

## C6. "Wide intervals are uninformative at instance level" — YU66 R2

**Verdict: Reviewer point is fair but the framing is wrong; this is a feature, not a bug.**

Wide intervals literally are the signal: they say "don't trust the absolute score on this task — switch to ranking." This is the operational guideline we already articulate in §5.5 and Conclusion. Polaris (width 0.68 = 17%) demonstrates the method *can* produce tight intervals when warranted.

### Rebuttal action
Strengthen this framing in the rebuttal. No code or numeric changes needed — just position the wide-interval finding as the *purpose* of the diagnostic, not a flaw.

---

## C7. "Methodological novelty is incremental over text-only CP-LLM work" — WoGp preamble

**Verdict: Reviewer is CORRECT.** We borrowed the conformal pipeline (R2CCP, boundary adjustment, midpoint analysis) from Sheng et al. directly. Our contribution is empirical (multimodal stratification, RSG, Polaris/MLLM comparison), not methodological.

This matches the [paper_contribution_framing](paper_contribution_framing.md) memory: position contribution as empirical. The rebuttal should embrace this rather than fight it.

---

## C8 & C9. Task taxonomy + missing CP method citations — GBAF

**Verdict: Reviewer is CORRECT, both are trivial camera-ready fixes.**

- Add citations for CQR (Romano 2019), CHR (Sesia 2021), Boosted CQR/LCP (Wang/Xie 2024), OrdinalAPS (Romano 2020), Naïve Split (Vovk 2005). Most are already in references_write_01.bib.
- Adopt a more principled task taxonomy (cite MM-Vet or VL-RewardBench's vision-capability axes; or empirically cluster tasks by feature similarity and report the resulting groups).

---

# Summary: who is right?

| Critique | Code correct? | Paper correct? | Reviewer correct? |
|----------|---------------|----------------|--------------------|
| C1 (conformal protocol) | **YES** | NO (misleading wording in §3.2) | NO (in fact), but confusion is on us |
| C2 (boundary adjustment formula) | **YES** | NO (wrong formula in §3.3) | **YES** — they found a real paper bug |
| C3 (annotation-quality causal claim) | n/a | NO (overcausal language) | **YES** |
| C4a (RSG missing parens) | n/a | YES | NO (verified against WIT data) |
| C4b (RSG novelty overclaim) | n/a | NO | **YES** |
| C4c (RSG ad-hoc, no CIs) | n/a | partly NO | partly **YES** |
| C5 (width normalization) | n/a | NO (inconsistent ÷4 vs ÷5; abstract numbers wrong) | **YES** |
| C6 (wide intervals uninformative) | n/a | partly NO (could frame better) | partly NO |
| C7 (methodological novelty) | n/a | NO (overclaimed) | **YES** |
| C8 (task taxonomy) | n/a | could be better | **YES** |
| C9 (CP citations missing) | n/a | NO | **YES** |

## Bottom line for the rebuttal

**We do NOT need to re-run experiments.** All our code is correct (matches the reference paper that we borrowed from). The reported numbers are valid.

**We DO need to fix the paper.** The bugs are all in writing/description, not in the underlying experiments:
1. §3.2 — clarify the three-way nested split for R2CCP.
2. §3.3 — swap the formula from $[\lceil l\rceil, \lfloor u\rfloor]$ to $[\lfloor l\rfloor, \lceil u\rceil]$.
3. Abstract + Table 7 — fix width-as-percentage to use the consistent ÷(K−1) = ÷4 convention.
4. Soften "primarily driven by annotation quality" → "consistent with annotation quality."
5. Soften "previously unrecognized failure mode" → "underexplored in multimodal-judge evaluation."
6. Add citations for the unfamiliar CP method names in §5.1.
7. Add 10-seed CIs to the RSG table.

These are all camera-ready-style fixes — a reviewer who saw the rebuttal acknowledging the bugs and committing to the fixes could reasonably move WoGp from 5 to 6, and possibly even YU66 from 4 to 5. Combined with GBAF's 7, that puts us around 6.0 — within striking distance of accept.
