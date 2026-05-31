# Reviewer GBAF — Rating: 7 (Good paper, accept)

**Confidence:** 3 (fairly confident)
**Submitted:** 7 May 2026, 16:34 (modified 22 May 2026, 08:53)
**Ethics flag:** No
**Headline:** "Conformal Prediction to Better Evaluate of VLM Judge Reliability"

## Summary (from reviewer)

This work explores the reliability of VLM judges, particularly diving into correlation with humans on ranking tasks does not equal reliability on point judgements. To formalize this, the authors use conformal prediction, where point scores are converted into conformal intervals. This gives the uncertainty around any given model judgement. They use two multimodal datasets with human-annotated scores for calibration.

They find that there is a disparity between ranking, which is consistent amongst judges, and scoring, which is unreliable with variance. This distinction is seen in particular across different tasks and is dependent on the data annotation quality. They suggest using conformal prediction as a means of determining the best evaluation setting: when intervals are narrow, absolute scoring is appropriate; when intervals are wide, relative ranking is more reliable.

## Reasons To Accept

- **Well grounded motivation:** I find this to be a really interesting motivation with informative results! VLM judges are being used widely to automate evaluation and correlation with humans is the most common way to validate the approach. This works tackles the importance of exploring reliability, beyond just high correlation numbers.
- **Strong findings about correlation/accuracy versus reliability:** As noted in the paper, we tend to rely on correlation with human ratings to justify automatic evaluations using judges. The findings in this work, particularly in Section 5.2, really add depth about how high performing models wrt ranking can still have very poor point score estimations.
- **Depth of evaluation:** The authors tested a range of different conformal prediction methods (Table 1). See questions section for additional take on this contribution.
- **Range of VLM judges used:** While only 3 VLMs are evaluated, I appreciate that these were intentionally chosen with one "speciality" judge model, one slightly larger reasoning model and one higher performing (on general multimodal tasks) closed-source model.

## Reasons To Reject

### R1. Task taxonomy feels arbitrary
> Limited comparison of tasks: I appreciate the number of tasks evaluated in Table 3. However, the categories feel a bit arbitrary — COCO is "Gen. VQA" yet TextVQA is "Vision". To draw more structured conclusions about the task-dependent differences, I suggest making a more formal taxonomy to define the tasks, perhaps drawing on existing work.

## Questions To Authors

### Q1. Citations missing for conformal methods
> Limited details on different conformal methods: Unless readers are already quite familiar with conformal prediction methods, the names in Section 5.1 might be unfamiliar. I suggest adding a citation to each. As a minor note, this contributes less to the overall narrative and could probably be moved to the conclusion.

## Rebuttal notes (draft thinking)

- **R1:** Reasonable critique. Could adopt MMBench / MM-Vet taxonomy or cite the visual capability taxonomy from VL-RewardBench. Alternatively, justify the existing grouping with a brief explanation: COCO = natural-image general VQA, TextVQA = reading-heavy vision = "Vision" because the model must extract text from pixels rather than answer about scene content. Even better: re-cluster empirically by feature similarity and show widths still group meaningfully.
- **Q1:** Trivial fix — add citations for CQR (Romano 2019), CHR (Sesia 2021), Boosted CP (Wang 2024), LVD (Lin 2021 — already cited), OrdinalAPS (Romano 2020), Naïve Split CP (Vovk 2005 — already cited), R2CCP (Guha 2024 — already cited). Will add in camera-ready.

## Overall

This is our strongest review and the one to leverage in the rebuttal. The reviewer "gets" the contribution and the motivation; both concerns are easily addressable.
