# Rebuttal Format Guide

**Source:** Extracted from the ICML 2026 TRACER paper rebuttal trajectory (Submission 32414), where the paper moved from initial weak-reject scores to an Accept after a carefully structured rebuttal. This is a *style/format template*, completely independent of TRACER's content. Reuse this exact structure for the CoLM 2026 VLM-Judge rebuttal.

**Goal mapping for our CoLM submission:**
- Reviewer YU66 (rating 4): aim for +1 (→5), +2 (→6) would be excellent.
- Reviewer WoGp (rating 5): aim for +1 (→6). Most addressable; has concrete fixable critiques.
- Reviewer GBAF (rating 7): already accept. Address minor asks cleanly; lock in champion status.

Average target: from 5.33 → ~6.0, putting us in striking distance of accept.

**Word budget per reviewer:** up to ~5,000 words allowed. Use the budget for *precision and clarity*, not padding. The prime directive: **do not confuse the reviewer at any cost.**

---

## The 7 Structural Patterns to Copy

### 1. Opening line, always
One sentence that thanks the reviewer **and names a specific positive they actually raised**. Don't say "thank you for the constructive feedback" — quote what *this* reviewer praised.

Examples from TRACER rebuttals:
- "Thank you for your constructive feedback and for recognizing the importance of trajectory-level uncertainty in agentic systems, as well as the cleanliness of the TRACER method."
- "Thank you for your detailed review and for highlighting the coherence of our pipeline and the strong empirical gains across domains."
- "Thank you for your exceptionally positive feedback and for recognizing the technical rigor, originality, and practical value of TRACER."

This sets a non-defensive tone before any pushback lands.

### 2. Per-weakness blocks with explicit headers
Use the reviewer's own labeling: `W1: <reviewer's phrasing>`, `W2: ...`, `Q1: ...`. The reviewer scrolls and sees their own list mirrored back — they don't hunt for which concern you answered.

Example header: `W1: Tuning Fairness and Validation Protocol:`

### 3. Three response modes — used cleanly
Pick one per concern. Don't mix.

**(a) Concede + fix.** Used for genuine bugs and oversights.
> "We agree with reviewer's point regarding the 'risk-dominates-hazard' and 'tail-sparsity' assumptions. … we will add an empirical analysis section in the Appendix to address your concerns."

**(b) Clarify a misread.** Used when the reviewer misunderstood; redirect without defensiveness.
> "Regarding the concern about trajectory-aware baselines, we would like to clarify that the SAUP method is inherently trajectory-aware."

**(c) Push back with evidence.** Cite the prior work, then state our distinct contribution. Never "you're wrong" — always "here is the additional context."
> "Notably, Zhao et al. (ACL 2025) introduced the SAUP baseline, which we directly compared TRACER against in our study, demonstrating that TRACER achieves superior, state-of-the-art performance. Furthermore, while the other prior works you mentioned explore semantic-level uncertainty… TRACER's distinct contribution lies in transitioning from single-step … to multi-turn, dual-control trajectory risk aggregation."

### 4. Always close each block with the concrete change
Every block ends with a verb the reviewer can verify in the revision: "We will add…", "We will expand…", "We will include…", "We will clarify in section A."

This converts a debate into a checklist the reviewer can grade.

### 5. Closing line, always
> "I remain available throughout the discussion period to provide any further clarifications or empirical details you might need. I truly appreciate your feedback and welcome the opportunity to work together to improve the paper."

Signals continued engagement — matters when reviewers post follow-up questions.

### 6. The follow-up reply pattern
When a reviewer says "promises aren't enough, show me results" — **produce the concrete artifact inside the discussion window, do not just promise camera-ready.**

In TRACER, reviewer 8w2A's follow-up said "would greatly aid Reviewer's re-evaluation … to see some of the preliminary results on a promised additional benchmark during this discussion phase." Authors responded with full tables on ToolHop and ComplexFuncBench. That's what moved the needle.

Format the new results as a markdown table directly in the comment. Include all models, all domains, all baselines side-by-side with our method last and bolded if possible.

### 7. AC summary comment at the end
Post a final consolidated note addressed to the Area Chair that:
- Lists which concerns each reviewer raised
- States what was resolved during discussion
- Names any reviewer who is championing
- Reaffirms camera-ready commitments

The AC reads dozens of papers; our summary becomes their reference document. Don't skip this.

---

## Tonal Rules (Hard Constraints)

1. **Never argue defensively.** "We disagree" → reframe as "We would like to clarify…" or "We would like to provide additional context…"
2. **Never confuse the reviewer.** If a sentence requires re-reading, cut it. Short sentences win.
3. **Concede real bugs immediately.** Reviewers reward intellectual honesty. Trying to defend an actual mistake destroys credibility on the points where we *are* right.
4. **No emojis, no hedging filler.** ("Indeed," "Of course," "It is worth noting that" — strip.)
5. **End every weakness block with a concrete action.** No standalone arguments.
6. **Cite specific section/figure numbers** when promising changes ("§3.3", "Table II", "Appendix A.8") — feels concrete, easy to verify.
7. **Address the reviewer who is closest to accepting first** in tone (warmth), but address the toughest reviewer with the most evidence (volume).

---

## The Template Skeleton

```
Thank you for [SPECIFIC POSITIVE THEY RAISED]. We appreciate your insights and address your concerns below.

W1: [Reviewer's exact concern label]
[Mode: concede / clarify / push back with evidence — one paragraph, 50–150 words].
We will [specific concrete action] in [specific section/appendix].

W2: [Next concern]
[Response paragraph].
We will [concrete action].

...

[If applicable] Q1: [Reviewer's question]
[Direct answer]. [Concrete addition to paper].

Please let me know if any further clarifications or data would be helpful during the rebuttal phase. We are committed to fully addressing your feedback and strengthening the paper.
```

---

## Follow-Up Reply Skeleton (when reviewer asks for more)

```
Thank you for the thoughtful follow-up and for emphasizing [WHAT THEY EMPHASIZED].

We are happy to share [NEW ARTIFACT — results, plot, analysis].

[New content: table, numbers, brief analysis tying it back to the original concern.]

Summary

[1–2 sentences on what the new evidence shows, restating that it addresses the specific concern they raised.]

We hope these additional results address [SPECIFIC CONCERN]. We kindly ask you to consider them in your final evaluation and update your score accordingly.
```

---

## AC Summary Skeleton

```
Summary for Area Chair

We thank the reviewers for their detailed and constructive feedback. Below is a concise summary of how we addressed the remaining concerns during the discussion phase:

Primary Remaining Concern (shared across reviewers X, Y, Z): [the cross-cutting issue]

Reviewer Signals:
- Reviewer X noted [outcome of their thread]
- Reviewer Y stated [outcome]
- Reviewer Z [outcome]

Our Response: [What we added during discussion]

Key Findings:
- [Bullet]
- [Bullet]

Alignment with Reviewer Requests: [How this maps to their explicit asks]

Positive Reviewer Signal: [Champion mention if any]

Camera-Ready Commitment: We will incorporate all new analyses and results in the camera-ready version.
```

---

## When to Use Each Mode for Our CoLM Reviewers

Mapping the audit findings in [[honest_verdict]] onto response modes:

**To WoGp (rating 5 — top priority for +1):**
- §3.3 boundary adjustment formula bug → **Concede + fix** ("the paper text incorrectly stated [⌈l⌉, ⌊u⌋]; the implementation uses expansion [⌊l⌋, ⌈u⌉], consistent with Sheng et al. Theorem 1. We will correct §3.3.")
- Width normalization inconsistency → **Concede + fix**.
- R2CCP internal split / "trained on calibration set" → **Clarify a misread** (the protocol *is* valid split conformal; only the wording is misleading).
- RSG parentheses critique → **Push back with evidence** (verified with WIT data, paper formula matches Table; reviewer's reading would contradict it).

**To YU66 (rating 4 — hardest to move):**
- Methodological novelty concerns → **Clarify** (frame as empirical contribution, not methodological).
- Single-judge / dataset breadth concerns → **Concede + commit** to expanded analysis in camera-ready.
- "Driven by annotation quality" overcausal claim → **Concede + soften wording**.

**To GBAF (rating 7 — champion-lock-in):**
- Thank profusely, address every minor ask, name them as our champion.
- This is the `kzjs`-equivalent reviewer. Easy points; don't squander them.

---

## Hard Reminders

- **Limit:** ~5,000 words per reviewer. Don't fill it. Precision beats volume.
- **Zero experiments need to be re-run** for our paper — every critique is addressable by writing fixes. This is a luxury; use it.
- **No GitHub pushes during rebuttal drafting.** Standing instruction from prior session: wait for explicit go-ahead before pushing any .tex or .bib changes.
- **Save every rebuttal draft** in this `paper/colm2026/reviews/` folder as `rebuttal_<reviewer>.md` so we maintain the track record.
