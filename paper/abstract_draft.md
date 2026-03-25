# Paper Title & Abstract Drafts

**Target venue:** CoLM 2026 (Conference on Language Modeling)
**Date:** 2026-03-24
**Status:** Draft for funding agency abstract submission

---

## Title Options

1. **Uncertainty-Aware VLM-as-a-Judge: Conformal Prediction for Multimodal Evaluation**
2. **How Reliable Are VLM Judges? Conformal Prediction Reveals Task-Dependent Uncertainty in Multimodal Evaluation**
3. **Conformal Prediction Intervals for VLM-as-a-Judge: Coverage Guarantees Across Visual Task Types**

---

## Abstract (Option A — Empirical Study Focus)

VLM-as-a-Judge has emerged as a scalable paradigm for evaluating multimodal AI systems, yet the reliability of these evaluations remains poorly understood. We present the first comprehensive study of uncertainty quantification for VLM-based judges using conformal prediction. Given a VLM judge's score-token logits on a 1–5 Likert scale, conformal prediction constructs prediction intervals with provable coverage guarantees — capturing the range of plausible human scores for each evaluation instance. We evaluate 8 conformal prediction methods across two VLM judges (LLaVA-Critic-7B and Phi-4-reasoning-vision-15B) on the MLLM-as-a-Judge benchmark spanning 14 multimodal task categories and 5,717 samples. Our analysis reveals three key findings. First, VLM judges produce substantially wider prediction intervals than their text-only LLM counterparts (width 3.0 vs 0.6–2.5 on a 5-point scale), reflecting fundamental challenges in cross-modal evaluation that conformal prediction makes visible. Second, interval width varies by up to 70% across visual task types — aesthetics and natural image tasks yield informative intervals (width ~2.1), while mathematical reasoning, chart understanding, and AI-generated image evaluation produce near-uninformative intervals (width ~3.5) — providing practitioners the first quantitative reliability map for VLM-based evaluation. Third, despite low exact-match accuracy (32–35%), conformal intervals recover 97.8% of judge errors, demonstrating that prediction intervals transform unreliable point scores into trustworthy interval evaluations. We further show that boundary adjustment for discrete rating scales and chain-of-thought prompting significantly improve interval calibration. Our results establish that conformal prediction is essential infrastructure for deploying VLM judges responsibly, and that visual task complexity is the primary driver of evaluation uncertainty.

---

## Abstract (Option B — Shorter, Punchier)

VLM-as-a-Judge has become a popular approach for evaluating multimodal AI, but how much should we trust these evaluations? We apply conformal prediction to VLM judges, producing prediction intervals with statistical coverage guarantees for each evaluation instance. Evaluating 8 conformal methods with two VLM judges across 14 multimodal task categories (5,717 samples), we find that VLM judges exhibit substantially higher uncertainty than text-only LLM judges — prediction intervals span 60% of the score range on average, compared to 12–50% for LLM judges. Critically, this uncertainty is task-dependent: aesthetics judgments produce tight, informative intervals, while mathematical reasoning and chart understanding yield nearly uninformative ones. Despite only 32–35% exact-match accuracy with human annotators, conformal intervals capture the true human score 97.8% of the time, transforming unreliable point predictions into trustworthy interval evaluations. Our analysis provides the first uncertainty-aware reliability map for VLM-based evaluation across visual domains and demonstrates that conformal prediction is necessary infrastructure for responsible deployment of VLM judges.

---

## Abstract (Option C — Method + Findings Balanced)

Large Vision-Language Models (VLMs) are increasingly used as automated judges to evaluate multimodal AI systems, yet their scoring reliability is largely unquantified. We present the first application of conformal prediction to VLM-as-a-Judge, constructing prediction intervals from judge score-token logits that provide distribution-free coverage guarantees. We systematically evaluate 8 regression and classification-based conformal methods — including R2CCP, CQR, CHR, and Boosted variants — with two VLM judges on the MLLM-as-a-Judge benchmark (5,717 samples, 14 visual task categories, 1–5 Likert scale). We adapt the framework of Sheng et al. (2025) from text-only LLM judges to the multimodal setting, identifying key challenges: discrete integer ground truth (vs. averaged continuous scores), stronger judge overconfidence, and task-dependent uncertainty. Our experiments reveal that (1) R2CCP achieves exact 90% marginal coverage with the narrowest intervals among well-calibrated methods, (2) prediction interval width varies 70% across task types — from 2.1 for aesthetics to 3.5 for mathematical reasoning, providing a quantitative reliability map, (3) conformal intervals recover 97.8% of judge scoring errors, and (4) boundary adjustment for integer scales is critical, increasing coverage from 88–90% to 96–100%. We release our code, features, and analysis to support uncertainty-aware multimodal evaluation.

---

## Abstract (Option D — Application-Motivated, Why VLM Judges Matter)

As multimodal AI agents are deployed in real-world applications — navigating web interfaces, booking flights, analyzing documents, and interpreting screenshots — reliably evaluating their visual reasoning becomes critical. Unlike text-only tasks, these scenarios demand judges that process both visual and linguistic modalities: a judge must see the screenshot, read the UI elements, and assess whether the agent's action was correct. This makes Vision-Language Models (VLMs) the natural choice for automated evaluation, yet a fundamental question remains unanswered: when can we trust a VLM judge's score, and when is it guessing?

We address this question through conformal prediction, a distribution-free framework that transforms a VLM judge's single-point score into a prediction interval with provable coverage guarantees. From the judge's score-token logits alone, conformal prediction constructs an interval that contains the true human score with at least 90% probability — no retraining, no additional data, and no assumptions about the judge's error distribution.

We conduct the first systematic study of conformal prediction for VLM-as-a-Judge, evaluating 8 methods across two VLM judges (LLaVA-Critic-7B, Phi-4-reasoning-vision-15B) on 5,717 multimodal evaluation instances spanning 14 visual task categories. Three critical findings emerge. First, conformal intervals recover 97.8% of judge scoring errors — when the judge predicts a score of 4 but the human says 2, the conformal interval [2, 5] still contains the truth, providing a safety net that point scores cannot offer. Second, uncertainty is strongly task-dependent: aesthetics and natural image tasks yield informative intervals (width 2.1 on a 5-point scale), while chart reasoning, mathematical figures, and AI-generated image evaluation produce intervals spanning 70% of the score range (width 3.5), quantifying precisely where VLM judges cannot be trusted. Third, VLM judges exhibit systematically wider intervals than text-only LLM judges (3.0 vs 0.6–2.5), revealing that the visual modality introduces a distinct source of evaluation uncertainty that existing text-only analyses miss entirely.

Our results establish conformal prediction as essential infrastructure for deploying VLM judges in high-stakes evaluation pipelines, and provide practitioners with the first task-specific reliability map for multimodal automated evaluation.

---

## Abstract (Option F — FINAL, Send This One)

**Title:** How Reliable Are VLM Judges? Conformal Prediction Reveals Task-Dependent Uncertainty in Multimodal Evaluation

As multimodal AI agents are deployed in real-world applications such as navigating web interfaces, interpreting screenshots, and analyzing documents, reliably evaluating their visual reasoning becomes critical. Vision-Language Models are the natural choice for automated evaluation since these tasks require understanding both visual and linguistic content, yet a fundamental question remains: when can we trust a VLM judge's score, and when is it unreliable? We address this through conformal prediction, a distribution-free framework that transforms a VLM judge's single-point score into a prediction interval with provable coverage guarantees, requiring only the judge's score-token logits with no retraining, no additional data, and no assumptions about the error distribution. We conduct the first systematic study of conformal prediction for VLM-as-a-Judge, evaluating multiple conformal methods across two VLM judges on the MLLM-as-a-Judge benchmark spanning 5,717 samples across 14 visual task categories. Our analysis reveals that evaluation uncertainty is strongly task-dependent: aesthetics and natural image tasks yield tight, actionable intervals covering roughly 40% of the score range, while chart reasoning, mathematical figures, and infographic understanding produce intervals covering 70% of the score range, providing practitioners with the first quantitative reliability map for VLM-based evaluation. We further show that conformal prediction exposes a failure mode invisible to standard metrics: on chart understanding, the judge achieves the highest ranking correlation across all 14 task types while simultaneously producing some of the widest prediction intervals, meaning it correctly identifies which answers are better but cannot assign reliable absolute scores. This decoupling of ranking quality from scoring precision has direct implications for practitioners choosing between pairwise comparison and absolute scoring protocols. Our findings establish conformal prediction as essential infrastructure for trustworthy deployment of VLM judges in multimodal evaluation pipelines.

---

## Notes for Abstract Selection

- **Option A** is the most complete — good for funding agency who wants to see breadth
- **Option B** is the shortest — good if there's a strict word limit
- **Option C** is the most technical — closest to final CoLM paper style
- **Option D** is application-motivated — explains WHY this matters (AI agents, real-world deployment) before diving into WHAT we found. Best for funding agencies who care about impact and practical relevance.

**Recommended for funding agency: Option D** — it answers "why should we care?" before "what did you find?" and grounds the work in real-world AI agent deployment scenarios.

All options share the same core contributions:
1. First CP study for VLM judges (new findings, not just engineering)
2. Task-dependent uncertainty (70% width variation across visual domains)
3. CP recovers 97.8% of judge errors (practical safety net)
4. VLM judges are fundamentally more uncertain than LLM judges (visual modality adds uncertainty)
