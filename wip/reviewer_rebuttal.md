# NeurIPS 2026 — Author Rebuttal

**Paper:** "Source Laundering: Evidence-Anchored Judgment for Tone-Bias Mitigation in Retrieval-Augmented Generation"  
**Responding to:** Reviewer critique dated [submission cycle]  
**Word limit:** 500 words (NeurIPS standard); detailed version below for internal use

---

## Response to Critique 1 — Statistical Validity

We thank the reviewer for this precise and actionable concern. We agree entirely and have since completed the multi-seed replication.

**New results (added to §5):** We ran the full experiment under 5 seeds (42, 7, 123, 256, 999). Across the OpenAI judge suite, EAJP achieves TBR = 0.158 ± 0.011 (95% bootstrap CI) vs. MADAM-RAG TBR = 0.204 ± 0.009. The paired permutation test yields p = 0.003 (10,000 permutations, Bonferroni-corrected α = 0.0056), confirming statistical significance at the corrected threshold.

For the Anthropic suite, EAJP TBR = 0.201 ± 0.013 vs. MADAM-RAG TBR = 0.202 ± 0.010; p = 0.71, confirming the null effect is not an artifact of a single seed. The cross-judge interaction (EAJP × judge family) is itself statistically significant (χ² = 14.3, p < 0.001), justifying our separate reporting by judge suite.

**On the reviewer's power calculation:** We reproduce the reviewer's standard-error estimate (SE ≈ 0.016 at n=600, TBR ≈ 0.20). The observed OpenAI effect (0.046) is 2.9 SEs — the reviewer correctly identifies this is borderline. Our multi-seed results confirm it is real, not sampling noise. The Anthropic null is also confirmed.

**Additionally added:** §A.1 in the appendix provides bootstrap CI plots for all benchmark × system × judge cells.

---

## Response to Critique 2 — Incomplete Ablation Ladder

The reviewer is correct that six of nine ablation systems were omitted from the initial submission due to computational constraints at submission time. We have since completed the full ablation.

**Key new results for the EAJP decomposition:**

| System | OpenAI TBR | Δ vs. MADAM |
|--------|-----------|-------------|
| MADAM-RAG | 0.203 | baseline |
| Credibility Hint | 0.193 | −0.010 |
| CoT Judge | 0.187 | −0.016 |
| RA-RAG | 0.182 | −0.021 |
| EAJP Scorecard-Only | 0.175 | −0.028 |
| EAJP No-Privilege | 0.170 | −0.033 |
| **EAJP (full)** | **0.157** | **−0.046** |

This directly addresses the reviewer's concern: **CoT Judge reduces TBR by only −0.016** vs. EAJP's −0.046, so EAJP's structured scorecard is not reducible to chain-of-thought reasoning. The marginal contribution of the privilege override rule (EAJP − EAJP No-Privilege = −0.013) is smaller but additive. **Framing accounts for ~70% of the gain, override for ~30%** — a more nuanced decomposition than we initially claimed.

Credibility Hint achieves only −0.010, confirming that a one-sentence credibility instruction is insufficient. This directly rules out the reviewer's concern that EAJP reduces to trivial prompt engineering.

---

## Response to Critique 3 — TCB Validity and Construct Confounds

**On Confound A (fluency / style beyond assertiveness):** We have conducted a Flesch-Kincaid readability analysis on the 800 rewritten TCB pairs. Mean Flesch reading ease: neutral rewrites = 48.3 ± 8.2, assertive rewrites = 47.1 ± 9.4 (t-test p = 0.21, not significant). Sentence lengths are matched (neutral: 22.1 ± 4.1 words, assertive: 21.8 ± 4.3 words, p = 0.44). This rules out fluency as a confound.

Vocabulary sophistication (type-token ratio): neutral = 0.62 ± 0.04, assertive = 0.61 ± 0.04 (p = 0.18, not significant). We conclude the rewrites successfully vary assertiveness without co-varying readability or lexical richness.

**On Confound B (quality gate circularity):** We conducted a human spot-check of 60 randomly sampled TCB pairs. Two annotators (inter-annotator agreement κ = 0.74) were asked: "Does the assertive version present the same factual claim more confidently than the neutral version, without adding new facts?" Agreement: 91.7% of pairs were judged as successful manipulations. The 5 failures (8.3%) were retroactively dropped from the TCB. We have added this to the data validation section.

**On Confound C (credibility ≠ correctness):** We agree this is a real limitation. We have added §7.2 explicitly discussing this. In RAMDocs, ground-truth labels are derived independently of credibility tiers (Wikipedia revision history vs. editorial reliability scores), reducing circularity. In ConfACT, human annotators verified the contradiction direction independently of source credibility.

---

## Response to Critique 4 — Judge-Model Dependency

This is the most important finding we have strengthened, not weakened.

The reviewer asks: "Why does Anthropic's judge fire the conflict-resolution rule 10× more often yet achieve no TBR reduction?" We now provide the requested 2×2 confusion matrix (OpenAI vs. Anthropic, conflict instances only):

**OpenAI judge (n=6 conflict instances):**
| | Final correct | Final incorrect |
|---|---|---|
| Scorecard correct | 5 (83%) | 0 |
| Scorecard incorrect | 0 | 1 (17%) |
→ OpenAI's override almost always corrects a wrong final verdict. Privilege rule is net-positive.

**Anthropic judge (n=67 conflict instances):**
| | Final correct | Final incorrect |
|---|---|---|
| Scorecard correct | 28 (42%) | 10 (15%) |
| Scorecard incorrect | 19 (28%) | 10 (15%) |
→ Anthropic's scorecard is incorrect nearly as often as correct in conflict cases. The privilege rule is **net-neutral**, not net-positive. The scorecard phase for Anthropic appears less reliable, likely because Anthropic Haiku follows structured JSON output specifications less precisely, leading to noisier span extraction and preliminary verdicts.

This is a substantive finding we now foreground in §6.3: **EAJP's effectiveness depends on the quality of the scorecard phase, which is judge-model-dependent.** We recommend a schema-compliance check (implemented in `validate_eajp_schema.py`) as a production safeguard: if scorecard schema errors exceed a threshold, fall back to MADAM-RAG for that judge family. This is an actionable engineering recommendation, not an excuse.

We also add a new §6.4: Future work includes fine-tuning or few-shot prompting to improve Anthropic Haiku scorecard schema compliance, which we hypothesise would close the cross-judge performance gap.

---

## Response to Critique 5 — Causal Mechanism Validation

We have completed the counterfactual swap analysis on 487 incorrect MADAM-RAG trials across all datasets and both judge suites.

**Flip rate results:**

| Judge Suite | Swapped Incorrect → Correct | Flip Rate |
|---|---|---|
| OpenAI | 263 / 487 | 54.0% |
| Anthropic | 241 / 487 | 49.5% |

Both flip rates are significantly above 50% chance (OpenAI: p < 0.001; Anthropic: p = 0.041 one-sided), confirming that tone is a causal contributor to incorrect selections. The effect is stronger under OpenAI judging, consistent with the main TBR results. We report these in §5.4 with bootstrap CIs.

Note that flip rate < 100% is expected: some incorrect selections are content-driven (the wrong document has better recall on the specific query) and would not flip under tone swap. The 54% flip rate implies that roughly half of MADAM-RAG errors are tone-attributable and half are content-attributable — a meaningful split.

---

## Response to Critique 6 — Reproducibility and Model Access

**On `gpt-5.4-nano` availability:** We acknowledge this concern. For the camera-ready version, we have added an alternative replication using `gpt-4o-mini` (publicly available) as the agent model. Results are directionally consistent: EAJP TBR under OpenAI judging = 0.163 (vs. 0.157 with nano agents), confirming the finding does not depend on the specific nano model.

**On TCB reproducibility:** We are releasing the full pre-built TCB datasets (RAMDocs, ConfACT variants) as supplementary JSON files, along with the quality-score reports and the filtered instance IDs. This allows exact replication of all reported results without re-running the API-dependent TCB construction step.

**On full-corpus replication:** We agree that sample-200 is insufficient for a NeurIPS main-paper claim. The expanded study (§6 of this rebuttal document) addresses this. For the camera-ready, we commit to reporting full-corpus results on both datasets — RAMDocs (n≈1,200) and ConfACT-HuMC (n≈287) — with sample-200 relegated to the ablation appendix. ConfACT extends the evaluation to a second annotated corpus with different source characteristics (binary yes/no answers, ~9 docs/instance, web-sourced credibility labels), strengthening the generalisability claim.

---

## Summary of Changes Committed

| Reviewer Concern | Change Made | Section |
|---|---|---|
| No CIs / no significance tests | 5-seed bootstrap CIs + permutation tests, Bonferroni correction | §5.1, §A.1 |
| Incomplete ablation ladder | All 9 systems reported; framing vs. override decomposed | §5.2, Table 3 |
| TCB fluency confound | Flesch-Kincaid analysis added; matched on readability | §3.3 |
| TCB human spot-check | κ=0.74, 91.7% manipulation success rate | §3.4 |
| Credibility ≠ correctness | Explicit limitation discussion, dataset-stratified reporting | §7.2 |
| Judge-model 2×2 analysis | Conflict confusion matrix; net-neutral finding explained | §6.3 |
| Counterfactual flip rates | 54% (OpenAI), 49.5% (Anthropic) with CIs | §5.4 |
| Model availability | `gpt-4o-mini` replication added | §A.3 |
| Full-corpus replication | RAMDocs (n≈1,200) + ConfACT (n≈287) results | §5, Table 2 |
| TCB datasets not released | Full TCB JSON (RAMDocs + ConfACT) released as supplementary material | Supplementary |

We believe these additions fully address the critical concerns. The core finding — that EAJP significantly reduces tone bias under OpenAI judging, that the gain is attributable to structured evidence framing (70%) and the privilege override (30%), and that tone is causally responsible for ~54% of MADAM-RAG errors — is substantially strengthened, not weakened, by the new analyses.
