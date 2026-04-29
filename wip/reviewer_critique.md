# NeurIPS 2026 — Reviewer Critique

**Paper:** "Source Laundering: Evidence-Anchored Judgment for Tone-Bias Mitigation in Retrieval-Augmented Generation"  
**Reviewer role:** Area Chair + two independent reviewers (simulated)  
**Recommendation:** Weak Reject (5/10) — significant revisions required before acceptance

---

## Summary

The paper introduces EAJP (Evidence-Anchored Judgment Protocol), a two-phase LLM-judge prompting strategy designed to reduce *tone bias* in multi-agent debate-based RAG systems. The authors define tone bias as the tendency to prefer assertive but weakly-evidenced claims over neutral but well-supported ones, and operationalise it via a Tone-Controlled Benchmark (TCB) constructed through LLM-driven tone rewrites. Across a 200-instance-per-benchmark evaluation on two source datasets (RAMDocs and ConfACT-HuMC) and their TCB variants with two judge suites, EAJP achieves a −23.2% relative reduction in TBR under OpenAI judging, with negligible improvement under Anthropic judging.

The problem is timely and the pipeline is technically competent. However, the paper has critical methodological gaps that substantially weaken the claims, and the inconsistent cross-judge performance raises questions about the generality of the proposed intervention.

---

## Critique 1 — Weak Statistical Validity (Critical)

**Score impact: −2 points**

The entire empirical contribution rests on point estimates from a single seed (seed=42) over 200 instances per benchmark. No confidence intervals, no significance tests, and no multi-seed variance analysis are reported anywhere in the paper.

The observed EAJP improvement under OpenAI judging is a TBR reduction from 0.203 to 0.157 — an absolute difference of 0.046 across 600 eligible trials. This corresponds to roughly 28 fewer incorrect selections. Without a standard error estimate, it is impossible to determine whether this difference is statistically reliable or whether it would replicate under a different random sample.

The contrast between OpenAI (−23.2% TBR) and Anthropic (+0.2% TBR, essentially unchanged) is the paper's central finding. Yet this cross-judge divergence is never subjected to formal analysis. Does the interaction between judge family and EAJP effectiveness reach statistical significance? The paper cannot answer this.

**Specific gap:** At n=600 per cell, assuming TBR ≈ 0.20, the standard error of TBR is approximately $\sqrt{0.20 \times 0.80 / 600} \approx 0.016$. The observed OpenAI improvement (0.046) is roughly 2.9 standard errors — likely significant, but not confirmed. The Anthropic "no-improvement" result (0.002) is trivially within noise and cannot distinguish a true null effect from insufficient power.

**Required:** Bootstrap confidence intervals (2,000 samples), paired permutation tests per cell, and a multi-seed replication covering at least 3 seeds.

---

## Critique 2 — Incomplete Ablation Ladder (Serious)

**Score impact: −1.5 points**

The paper advertises a 9-system ablation ladder as a core contribution, yet the Results section reports only three systems: Standard RAG, MADAM-RAG, and full EAJP. The intermediate ablations — Credibility Hint, CoT Judge, RA-RAG, EAJP Scorecard-Only, and EAJP No-Privilege — are described in the Methods but absent from the results tables.

This is not a minor omission. The mechanistic claims depend critically on these ablations. Specifically:

1. **EAJP No-Privilege** isolates whether TBR reduction comes from the structured scorecard framing or from the mechanical override rule. Without this data, the paper cannot attribute its improvement to EAJP's specific design vs. any structured evidence prompt.

2. **CoT Judge** tests whether chain-of-thought evidence reasoning (a much simpler intervention) achieves comparable TBR reduction. If CoT Judge ≈ EAJP on TBR, the two-phase scorecard apparatus is unnecessary complexity.

3. **Credibility Hint** tests whether simply telling the judge "prefer the higher-credibility source" is sufficient. If it is, the engineering contribution of EAJP collapses to a one-sentence prompt addition.

Without these ablations, the paper's argument that EAJP's specific design is responsible for its improvement is unsubstantiated.

---

## Critique 3 — TCB Validity and Construct Measurement Confounds (Serious)

**Score impact: −1.5 points**

The Tone-Controlled Benchmark construction introduces two confounds that are not adequately addressed.

**Confound A: Tone ≠ Assertiveness.** The TCB rewrites aim to vary "assertiveness," but the LLM rewriter (Anthropic `claude-haiku-4-5-20251001`) may introduce additional changes: sentence length, vocabulary sophistication, claim specificity, and hedging patterns. These factors are partially controlled by the BERTScore / cosine semantic quality gate, but cosine similarity ≥ 0.90 is a relatively permissive threshold for preserving exact factual content. The paper does not verify that fluency, reading level, or argument structure are matched between neutral and assertive rewrites.

If assertive rewrites are also more fluent or more specific, TBR may be driven by fluency bias rather than assertiveness per se, which is a different (and less novel) finding.

**Confound B: Circularity in quality gating.** The semantic quality gate uses the same model family (`roberta-large`, `all-mpnet-base-v2`) to filter rewrites, but these models may systematically miss style-preserving changes that alter factual tone. The gate validates "same meaning," not "different tone." Without a direct readability or assertiveness audit of the resulting TCB pairs, it is unclear whether the manipulation succeeds.

**Confound C: Credibility ≠ Correctness.** The paper conflates credibility tier (editorial signal in source data) with ground-truth correctness. In the ConfACT dataset, document credibility is assigned by the dataset creators, not verified independently. If credibility assignment errors exist, TBR measurements are systematically biased.

**Required:** (a) Readability analysis (Flesch-Kincaid or equivalent) comparing original vs. rewritten documents. (b) Human spot-check of 50 TCB pairs to verify successful tone manipulation. (c) Explicit discussion of credibility-correctness alignment and its limitations.

---

## Critique 4 — Judge-Model Dependency Makes Claims Non-Generalisable (Serious)

**Score impact: −1 point**

The paper's primary claim — "EAJP reduces tone bias" — holds only under one of two tested judge models. The Anthropic Haiku 4.5 judge shows near-zero improvement (TBR 0.202 vs. 0.203), and in some benchmarks shows regression (TCB-RAMDocs: +0.02 TBR). The paper acknowledges this as a "judge-model effect" but offers no explanation and draws conclusions about EAJP's general effectiveness.

This is a significant problem for deployment claims. The paper's §5.4 advises practitioners to "adopt evidence-anchored judging on high-risk queries" without qualifying that this advice depends on which LLM family is used as the judge. A practitioner using Claude-family models would see no benefit from EAJP.

The scorecard-final conflict rate (0.008 for OpenAI vs. 0.084 for Anthropic) is itself evidence that the two judges interact with the two-phase protocol differently. This is not a nuisance finding — it is a fundamental mechanistic question. Why does Anthropic's judge fire the conflict-resolution rule 10× more often yet achieve no TBR reduction? Is the privilege override actually harming Anthropic-judged performance? These questions are left unresolved.

**Required:** Analysis of conflict-resolution outcomes stratified by judge family. Test whether EAJP's privilege rule is net-positive, net-neutral, or net-negative for each judge model.

---

## Critique 5 — No Causal Mechanism Validation (Moderate)

**Score impact: −0.5 points**

The paper hypothesises that assertive tone is the *causal* mechanism behind incorrect selections, but presents only observational evidence. The counterfactual swap analysis (implemented in `build_counterfactual_swaps.py` and `analyze_counterfactual_swaps.py`) is described in the codebase but not reported in the paper.

Without counterfactual swap results, the paper cannot rule out that the lower-credibility document is selected because it happens to be topically more relevant, more recent, or structurally better-formatted — not because it is more assertive. The tone manipulation (TCB) partially controls for this by holding content constant, but only at the TCB-dataset level, not for the natural (non-TCB) benchmarks.

---

## Critique 6 — Limited Practical Scope and Reproducibility Concerns (Minor)

**Score impact: −0.5 points**

**Scope:** All debate agents use `gpt-5.4-nano`, a model not yet publicly released at the time of submission. Readers cannot reproduce the results without access to this model, and the finding that EAJP helps under OpenAI judging but not Anthropic judging may partly reflect that `gpt-5.4-nano`-generated debate arguments are stylistically aligned with `gpt-5.4-mini` judge preferences.

**Reproducibility:** The paper does not provide the actual TCB datasets as supplementary material, only construction scripts. The TCB quality depends on the specific API responses from `claude-haiku-4-5-20251001`, which are non-deterministic even at temperature=0. Different runs may produce different TCB instances, making exact replication impossible without the persisted JSON.

**Compute scale:** The sample-200 run is presented as the main result, with no indication of full-corpus replication. NeurIPS standards expect results on the full dataset or rigorous justification for subsampling.

---

## Summary Scorecard

| Criterion | Score | Comment |
|-----------|-------|---------|
| Novelty | 7/10 | TCB construction + EAJP design are original |
| Technical correctness | 7/10 | Pipeline is well-implemented |
| Statistical validity | 3/10 | Single seed, no CIs, no tests |
| Completeness of ablation | 4/10 | 3 of 9 systems reported |
| Claim scope vs. evidence | 4/10 | Claims generalise beyond supported evidence |
| Presentation | 7/10 | Clear writing, good architecture diagrams |
| **Overall** | **5/10** | **Weak reject — major revisions required** |

---

## Questions for Authors

1. Why are the 6 intermediate ablation systems omitted from the results? The codebase contains all necessary code and cache infrastructure; were these experiments run?

2. What is the EAJP No-Privilege TBR under OpenAI judging? This single number would allow readers to decompose the framing vs. override contributions.

3. Can you report the counterfactual swap flip rates? This is the most direct test of whether tone is truly causal.

4. The Anthropic conflict rate is 0.084 but TBR is unchanged. Does the privilege rule systematically override correct-scorecard verdicts with incorrect-final verdicts for this judge family? Please provide a 2×2 confusion matrix: (scorecard correct/incorrect) × (final verdict correct/incorrect).

5. What is the inter-seed variance in TBR? Even two seeds would substantially strengthen the statistical case.

---

## Recommendation

**Weak Reject.** The core idea is sound and the implementation is competent. The paper fails primarily on statistical completeness and ablation coverage — both of which appear to be execution gaps rather than design flaws, given that the infrastructure exists in the codebase. A revision that adds: (a) 3-seed bootstrap CIs, (b) the complete ablation ladder results, (c) counterfactual swap flip rates, and (d) judge-stratified privilege-override analysis, would address the critical concerns and likely warrant acceptance.
