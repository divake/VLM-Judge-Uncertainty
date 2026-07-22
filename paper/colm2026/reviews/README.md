# CoLM 2026 Reviews — Paper 991

**Paper:** "How Reliable Are VLM Judges? Conformal Prediction Reveals Task-Dependent Uncertainty in Multimodal Evaluation"
**Venue:** CoLM 2026 (under review, double-blind)
**Reviews received:** 2026-05-22
**Submitted PDF:** `../991_How_Reliable_Are_VLM_Judge_CoLM.pdf`

## Summary of scores

| Reviewer | Rating | Confidence | Stance |
|----------|--------|------------|--------|
| YU66     | 4 — Ok but not good enough (reject) | 3 | Negative |
| WoGp     | 5 — Marginally below acceptance     | 3 | Borderline |
| GBAF     | 7 — Good paper, accept              | 3 | Positive |

**Average rating:** 5.33 / 10. Mixed reviews; one accept, one borderline, one reject.

## Files

- `reviewer_YU66.md` — full reject review (4/10) — focus: conformal validity + interval too wide + RSG underdeveloped
- `reviewer_WoGp.md` — borderline review (5/10) — focus: conformal protocol clarity + boundary-adjustment inconsistency + numerical inconsistencies
- `reviewer_GBAF.md` — accept review (7/10) — focus: positive on motivation + suggests task taxonomy + minor citation requests
- `concerns_summary.md` — consolidated list of all reviewer concerns grouped by theme, for use when drafting rebuttal

## Common concerns across reviewers

1. **Conformal protocol clarity (YU66 + WoGp)** — both reviewers think we may train R2CCP on the same calibration set used for nonconformity quantiles, breaking the split-conformal guarantee. **This is the most critical issue to address.**
2. **Interval width interpretability (YU66 + WoGp)** — widths of 3.05/3.60 on a 0–4 scale look uninformative; need to defend or contextualize.
3. **Polaris vs MLLM-Judge causality (YU66 + WoGp)** — both push back on "annotation quality drives width" since multiple factors change between the benchmarks.
4. **Ranking-Scoring Gap formalism (YU66 + WoGp)** — Eq (7) is ad-hoc, missing parentheses, not validated against downstream decisions, novelty overclaimed.
5. **Numerical inconsistencies (WoGp)** — boundary-adjusted widths reported as *larger* than raw widths (contradicting the formula), and width % normalization is inconsistent across sections.

## Next steps

- [ ] Draft rebuttal addressing the 5 common concerns above (priority: conformal protocol)
- [ ] Verify in code whether R2CCP training and quantile computation actually share data
- [ ] Re-check boundary-adjustment numbers in Tables 1 and 2 against the formula in §3.3
- [ ] Re-check the 61% vs 76% width normalization in Table 7 / §5.4
