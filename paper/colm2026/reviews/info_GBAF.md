# Reviewer GBAF — Complete Information File (Pre-rebuttal Reference)

**Purpose:** Source of truth for drafting the GBAF rebuttal. This is the champion-candidate reviewer (rating 7, accept). Strategy is to **acknowledge cleanly and close fast** — do not introduce new analyses or counter-evidence that could re-open settled questions.

**Reviewer details:**
- Title: "Conformal Prediction to Better Evaluate of VLM Judge Reliability"
- Rating: **7** (Good paper, accept)
- Confidence: 3 (fairly confident)
- Submitted: 2026-05-07

**Rebuttal goal:** Lock in the +7 (do not lose it). Aim to convert the reviewer into an explicit champion (similar to TRACER's reviewer kzjs: "I'm happy to be the champion for it if this is not a unanimous opinion").

---

## 1. Reviewer's Summary (verbatim)

> This work explores the reliability of VLM judges, particularly diving into correlation with humans on ranking tasks does not equal reliability on point judgements. To formalize this, the authors use conformal prediction, where point scores are converted into conformal intervals. This gives the uncertainty around any given model judgement. They use two multimodal datasets with human-annotated scores for calibration.
>
> They find that there is a disparity between *ranking*, which is consistent amongst judges, and *scoring*, which is unreliable with variance. This distinction is seen in particular across different tasks and is dependent on the data annotation quality. They suggest using conformal prediction as a means of determining the best evaluation setting: when intervals are narrow, absolute scoring is appropriate; when intervals are wide, relative ranking is more reliable.

## 2. Reviewer's Reasons to Accept (verbatim) — to quote in the opening

1. **Well grounded motivation:** "I find this to be a really interesting motivation with informative results! VLM judges are being used widely to automate evaluation and correlation with humans is the most common way to validate the approach. This works tackles the importance of exploring *reliability*, beyond just high correlation numbers."

2. **Strong findings about correlation/accuracy versus reliability:** "As noted in the paper, we tend to rely on correlation with human ratings to justify automatic evaluations using judges. The findings in this work, particularly in **Section 5.2**, really add depth about how high performing models wrt ranking can still have very poor point score estimations."

3. **Depth of evaluation:** "The authors tested a range of different conformal prediction methods (Table 1). See questions section for additional take on this contribution."

4. **Range of VLM judges used:** "While only 3 VLMs are evaluated, I appreciate that these were intentionally chosen with one 'speciality' judge model, one slightly larger reasoning model and one higher performing (on general multimodal tasks) closed-source model."

## 3. Reviewer's Concerns (the only two)

### Reason to Reject (R1): Limited comparison of tasks (taxonomy critique)
> "I appreciate the number of tasks evaluated in Table 3. However, the categories feel a bit arbitrary — COCO is 'Gen. VQA' yet TextVQA is 'Vision'. To draw more structured conclusions about the task-dependent differences, I suggest making a more formal taxonomy to define the tasks, perhaps drawing on existing work."

### Question (Q1): Limited details on different conformal methods
> "Unless readers are already quite familiar with conformal prediction methods, the names in Section 5.1 might be unfamiliar. I suggest adding a citation to each. As a minor note, this contributes less to the overall narrative and could probably be moved to the conclusion."

---

## 4. Honest Verdict

Both asks are **presentational fixes, not empirical gaps**. The reviewer is fundamentally satisfied with the work. The rebuttal exists primarily to:
1. Show we read the review carefully
2. Acknowledge and commit to each concrete ask
3. Reinforce that they made the right call recommending accept

| Critique | Verdict | Response mode | Action |
|---|---|---|---|
| R1 (taxonomy arbitrary) | Reviewer is correct | Clean concede + commit | Adopt MM-Vet capability dimensions (or similar principled taxonomy) for the camera-ready |
| Q1 (CP method citations) | Reviewer is correct | Clean concede + commit | Add citations for all 8 CP methods in §5.1; agree §5.1 can be compressed in main text |

**Tone is everything here.** Warm, brief, no defensiveness, no extra results. The reviewer already gave us 7; our job is to not give them any reason to second-guess.

---

## 5. Concrete revisions to commit

### For R1 (taxonomy)
- Adopt **MM-Vet's 6 vision capability dimensions** (Yu et al., 2023) as the primary task taxonomy: Recognition, OCR, Knowledge, Language Generation, Spatial Awareness, Math reasoning.
- Re-categorize the 14 MLLM-Judge tasks under this taxonomy in Table 3.
- This is a published, well-cited taxonomy (~500+ citations) — directly addresses "drawing on existing work."

### For Q1 (citations)
Add a citation after each conformal method name in §5.1. Most are already in `references_write_01.bib`:
- Naive Split CP → Vovk et al. 2005
- CQR → Romano et al. 2019
- Asymmetric CQR → Sesia & Romano 2020
- CHR → Sesia & Romano 2021
- LVD → Lin et al. 2021
- Boosted CQR / Boosted LCP → Xie et al. 2024
- OrdinalAPS → Romano et al. 2020 (or Lu et al. 2022)
- R2CCP → Guha et al. 2024 (already cited)

Also: agree to compress §5.1's prose in main text and move the full per-method comparison details to the appendix, per their suggestion.

---

## 6. Rebuttal strategy

**Length target:** ~800–1200 words. Substantially shorter than the WoGp (~2500) and YU66 (~3000) rebuttals — matching the reviewer's lighter critique load.

**Tone:** warm, grateful, concise. No new tables, no new experiments, no over-engineering. Quote their specific praise for §5.2.

**What we are NOT doing:**
- Not adding new analysis tables (could re-open settled questions)
- Not running new VLM inference (the reviewer didn't ask for it, and over-engineering for a +7 reviewer looks like under-confidence in the rest of the paper)
- Not litigating any minor framing — just accept their suggestions

**Structure:**
1. Warm thank-you that names §5.2 specifically (per their praise)
2. R1 response: acknowledge taxonomy critique + commit to MM-Vet 6-axis taxonomy
3. Q1 response: acknowledge + list the specific citations we will add + agree to compress §5.1
4. Brief summary of revisions (2–3 bullets)
5. Standard availability closer
