# Methods: Evidence-Anchored Judgment for Tone-Bias Mitigation in Retrieval-Augmented Generation

_NeurIPS 2026 submission draft — Methods section + Expanded Study Design_

---

## 1. Problem Formulation

We study **tone bias** in multi-document retrieval-augmented generation (RAG): the tendency of an LLM judge to prefer an assertive but poorly-supported answer over a neutral but well-evidenced one when retrieved documents conflict. We term the underlying mechanism *source laundering* — a low-credibility document can "launder" a false claim through confident rhetoric, causing a judge to overlook superior evidence in a high-credibility source.

Formally, given a query $q$ and a set of retrieved documents $\mathcal{D} = \{d_1, \dots, d_k\}$ with heterogeneous credibility tiers $c_i \in \{\text{high, medium, low}\}$ and answer-support labels $s_i \in \{\text{yes, no}\}$, a system $\mathcal{S}$ is said to exhibit **tone bias** on instance $q$ if it selects the answer supported by the lower-credibility document when a higher-credibility document supports the correct answer.

The **Tone Bias Rate (TBR)** is the primary metric:

$$\text{TBR} = \frac{|\{q \in \mathcal{Q}_{\text{eligible}} : \mathcal{S}(q) = \text{answer}(d_{\text{low-cred}})\}|}{|\mathcal{Q}_{\text{eligible}}|}$$

where $\mathcal{Q}_{\text{eligible}}$ contains only instances where exactly one yes-credibility document and one no-credibility document exist, making tone bias unambiguously attributable.

---

## 2. Systems

We implement and compare nine systems forming a complete **ablation ladder** from a deterministic baseline to the full proposed intervention.

### 2.1 Standard RAG (baseline)

A single-pass retrieve-then-generate baseline. All retrieved documents are concatenated into a single context block with document metadata (credibility tier, URL) and passed in one prompt to the LLM. No debate, no structured reasoning.

**Implementation:** `src/standard_rag.py`. Uses `gpt-5.4-nano` as the generation model.

### 2.2 Credibility-Weighted Majority Vote (deterministic baseline)

A fully deterministic baseline requiring no LLM call. Each document votes for the answer it supports (`doc.supports_answer`), weighted by credibility tier (high → 3.0, medium → 2.0, low → 1.0, unknown → 1.5). The answer with the highest total weight wins. Suite-independent; runs once before the per-judge-suite loop.

### 2.3 MADAM-RAG

Multi-agent debate following Wang et al. (2025). Each retrieved document spawns one `DocumentAgent`. Agents first answer independently (Round 0), then iterate for up to $R$ rounds: each agent receives all other agents' prior-round claims, evidence summaries, and position updates, and revises its position accordingly. Debate terminates by **individual stability** — when each agent's answer is unchanged from its prior round — preventing premature consensus.

The `Aggregator` receives all final arguments in **randomised order** (eliminating position bias) and produces a final answer via the MADAM aggregation prompt.

**Key implementation detail:** The per-round `debate_cache` (agent claims + arguments across all rounds) is persisted in `TrialResult.metadata`, enabling cross-suite reuse. When `--judge-suite both`, the debate is run once; both judge suites read the cached debate.

### 2.4 Chain-of-Thought Judge (CoT)

MADAM-RAG with the aggregator prompted to reason explicitly through four evidence-quality questions before committing to a final answer. Tests whether explicit evidence reasoning (without a structured scorecard) achieves the same TBR reduction as full EAJP.

### 2.5 RA-RAG

MADAM-RAG with a reliability-aware aggregator that performs cross-document consistency assessment before deciding. Inspired by Tang et al. (2025): corroboration across independent documents is used to weight answers. Tests whether corroboration-based reliability estimation is sufficient without EAJP's two-phase protocol.

### 2.6 Credibility Hint

MADAM-RAG with each document's credibility tier injected verbatim into the aggregator prompt before argument reading. No structured scorecard; no override rule. If `credibility_hint` achieves similar TBR reduction to full EAJP, the two-phase apparatus is over-engineered.

### 2.7 EAJP Scorecard-Only

Runs only the EAJP scorecard protocol (evidence-first structured assessment) and returns the preliminary evidence-only verdict, skipping the final argument-aware pass entirely. Isolates the evidence-scorecard contribution independently of the argument-reading phase.

### 2.8 EAJP No-Privilege

Full two-phase EAJP (scorecard then argument-aware final pass) with the scorecard override rule disabled (`privilege_scorecard=False`). Any TBR reduction vs. MADAM-RAG is attributable to **scorecard framing**; the gap to full EAJP isolates the **mechanical override** contribution.

### 2.9 EAJP (Evidence-Anchored Judgment Protocol) — primary intervention

The full proposed system. Forces the judge to:

1. **Scorecard Phase** — For each retrieved document, identify the most relevant claim, extract span-level supporting evidence, record the stated credibility tier, and perform cross-document consistency assessment. Emit a structured JSON scorecard with `preliminary_winner_id` and `preliminary_verdict`.
2. **Final Phase** — Read the full debate arguments alongside the scorecard. Emit `final_winner_id`, `final_answer`, `final_verdict`.
3. **Privilege Rule** — If `final_winner_id` ≠ `preliminary_winner_id`, override with the scorecard winner.

Both evidence bundles (scorecard phase) and final-phase arguments are **shuffled** to prevent position bias from inflating measured EAJP effectiveness on TCB benchmarks (where the correct document is always positioned first by construction).

**Implementation:** `EvidenceAnchoredAggregator` in `src/madam_rag.py`; two-phase prompt templates `EAJP_SCORECARD_*` and `EAJP_FINAL_*` in `src/prompts.py`.

---

## 3. Tone-Controlled Benchmark (TCB) Construction

### 3.1 Motivation

Existing benchmarks provide natural variation in document tone, but do not hold other factors (credibility, topic, length) constant while varying assertiveness. The TCB isolates tone as the sole experimental variable.

### 3.2 Algorithm

For each source instance:
1. Select the **yes-doc**: document with `supports_answer=yes`, highest credibility rank, shortest content (tie-breaking).
2. Select the **no-doc**: document with `supports_answer=no`, lowest credibility rank ≤ yes-doc credibility, shortest content.
3. Rewrite yes-doc to **neutral** register (hedged, evidence-qualified language).
4. Rewrite no-doc to **assertive** register (confident, unhedged language).
5. Apply semantic quality gate (§3.3).
6. Drop instances failing the quality gate; no iterative rewrites.

**Rewrite provider:** Anthropic `claude-haiku-4-5-20251001` by default; Gemini `gemini-3.1-flash-lite-preview` and a deterministic local heuristic (prefix/suffix rules) are also supported.

### 3.3 Semantic Quality Gate

Two metrics validate meaning preservation for each (original, rewritten) pair:

| Metric | Model | Threshold (current) |
|--------|-------|---------------------|
| Rescaled BERTScore F1 | `roberta-large` | ≥ 0.30 (RAMDocs: 0.65–0.70) |
| Cosine similarity | `all-mpnet-base-v2` | ≥ 0.90 |

Scores are auto-computed on MPS/CUDA/CPU, persisted in document metadata, and used to generate distribution plots and a bottom-$N$ inspection report. Instances below either threshold are dropped from the TCB dataset.

### 3.4 Datasets and TCB Variants

| Source Dataset | Description | TCB instances (current) |
|----------------|-------------|------------------------|
| RAMDocs (Wang et al., 2025) | Wikipedia-grounded QA with conflicting multi-hop evidence | ~200 (sample-200 run) |
| ConfACT-HumC _(new)_ | Human-annotated contradictory claim pairs from 290+ web sources | ~206 eligible |

---

## 4. Evaluation Metrics

### 4.1 Tone Bias Rate (TBR)
Primary metric. Fraction of eligible trials where the system selects the lower-credibility / tone-manipulated answer. On TCB datasets (where `unclear ≈ 0`), TBR ≈ 1 − Accuracy.

### 4.2 Accuracy (ACC)
`correct / (correct + incorrect)` on non-unclear trials. Provides complementary signal on absolute correctness.

<!-- ### 4.3 Substance Alignment Rate (SAR)
Fraction of eligible trials where the system selects the higher-credibility / correct answer. Measures positive alignment rather than negative bias; `SAR + TBR ≤ 1` (unclear trials are in the gap). -->

### 4.3 EAJP Scorecard-Final Conflict Rate
Fraction of EAJP trials where the preliminary scorecard verdict and the final argument-aware verdict disagree. Measures how often the privilege override fires; high conflict rates indicate judge susceptibility to rhetorical override.

<!-- ### 4.5 Substance Alignment Gain (SAG)
`SAR(EAJP) − SAR(MADAM)`. Measures the increase in correct-evidence wins attributable to the intervention, independent of unclear-trial prevalence. -->

---

## 5. Experimental Protocol

### 5.1 Current Results (sample-200 run)

**Judge suites:** OpenAI `gpt-5.4-mini` and Anthropic `claude-haiku-4-5-20251001`.  
**Agent/generation model:** `gpt-5.4-nano`.  
**Sample size:** 200 instances per benchmark × 4 benchmarks (RAMDocs, TCB-RAMDocs, ConfACT, TCB-ConfACT) × 3 systems × 2 judge suites = 4,800 total trials.  
**Seed:** 42 (fixed).

**Key findings (aggregate across benchmarks):**

| Judge Suite | System | Eligible Trials | TBR | SAR | EAJP Conflict Rate |
|---|---|---:|---:|---:|---:|
| OpenAI GPT-5.4-mini | Standard RAG | 600 | 0.212 | 0.148 | — |
| OpenAI GPT-5.4-mini | MADAM-RAG | 600 | 0.203 | 0.188 | — |
| OpenAI GPT-5.4-mini | EAJP | 600 | **0.157** | **0.208** | 0.008 |
| Anthropic Haiku 4.5 | Standard RAG | 600 | 0.212 | 0.148 | — |
| Anthropic Haiku 4.5 | MADAM-RAG | 600 | 0.202 | 0.167 | — |
| Anthropic Haiku 4.5 | EAJP | 600 | 0.203 | 0.148 | 0.084 |

EAJP reduces TBR by **−23.2%** relative under OpenAI judging; effect is near zero under Anthropic judging.

<!-- ### 5.2 Identified Limitations of Current Design

1. **Single agent model family** (`gpt-5.4-nano`) — confounds agent quality with system design.
2. **Single seed** — no variance estimates; point estimates cannot be tested for significance.
3. **No bootstrap confidence intervals** — cannot determine whether TBR differences are statistically reliable.
4. **Sample size (200/benchmark)** — insufficient for subgroup analysis (domain, difficulty, source type).
5. **Only 3 of 9 systems reported** — the full ablation ladder (CoT, RA-RAG, credibility hint, scorecard-only, no-privilege) is not systematically compared in current results.
6. **No causal mechanism validation** — counterfactual swap analysis (swap assertive ↔ neutral) has been implemented but not run at scale.
7. **No human evaluation** — LLM-judge TBR may itself be confounded by judge style preferences. -->

---

## 6. Expanded Study Design

### 6.1 Goals

The expanded study addresses all six limitations above and adds three new research questions:

- **RQ3:** Do all nine systems form a monotone ablation ladder on TBR, as hypothesised?
- **RQ4:** Is tone the causal mechanism for incorrect selections (counterfactual swap analysis)?
- **RQ5:** Does EAJP's effect generalise across agent model families and judge model families?

### 6.2 Datasets

| Dataset | Split | N instances | TCB variant |
|---------|-------|-------------|-------------|
| RAMDocs | test | ~1,000 | TCB-RAMDocs |
| ConfACT-HumC | full | ~206 eligible | TCB-ConfACT |

**Total:** ~1,400 unique instances per benchmark variant; ~2,800 TCB pairs.

### 6.3 Systems (full ablation ladder)

All nine systems from §2 run on all datasets and both benchmark variants (original + TCB):

1. `standard_rag`
2. `credibility_vote`
3. `madam_rag`
4. `cot_judge`
5. `ra_rag`
6. `credibility_hint`
7. `eajp_scorecard_only`
8. `eajp_no_privilege`
9. `eajp`

### 6.4 Judge Suite Expansion

| Suite | Models |
|-------|--------|
| OpenAI | `gpt-5.4-mini`, `gpt-4o` |
| Anthropic | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6` |
| Cross-family | All four judge models applied to same debate cache |

Cross-suite reuse of the MADAM debate cache (already implemented) means the four judge models add no agent-compute cost — only four additional judge passes per instance.

### 6.5 Agent Model Expansion

Run MADAM debate with two agent model families:
- **OpenAI agents:** `gpt-5.4-nano` (current), `gpt-5.4-mini` (stronger)
- **Anthropic agents:** `claude-haiku-4-5-20251001`

Each agent-model × judge-suite combination is a separate experimental cell.
<!-- 
### 6.6 Statistical Protocol

- **Seeds:** 5 random seeds (42, 7, 123, 256, 999) per cell.
- **Bootstrap CIs:** 2,000-sample bootstrap for TBR and SAR; 95% CIs reported.
- **Hypothesis tests:** Paired permutation test (10,000 permutations) for EAJP vs. MADAM-RAG TBR differences per benchmark × judge suite cell.
- **Bonferroni correction:** Applied across the 9 benchmark × judge-suite primary comparison cells ($\alpha_{\text{corrected}} = 0.05 / 9 \approx 0.0056$). -->

### 6.7 Counterfactual Swap Analysis (RQ4)

Using `build_counterfactual_swaps.py` + `analyze_counterfactual_swaps.py`:

1. Identify all trials where EAJP or MADAM-RAG produced an incorrect answer.
2. Swap tone labels on those instances (assertive ↔ neutral).
3. Re-run the judge on swapped instances (reusing debate cache).
4. Measure **flip rate**: fraction of previously-incorrect trials that flip to correct after tone swap.
5. A high flip rate ($>$50%) confirms tone is the causal mechanism, not content.

Target: run counterfactual swap on ≥500 incorrect trials per system × judge suite.

<!-- ### 6.8 Human Evaluation (50-instance subset)

For 50 randomly sampled ConfACT-HumC instances with disagreement between OpenAI and Anthropic judges:
- Two domain-expert annotators independently assess whether the selected answer is correct given the documents.
- Inter-annotator agreement reported (Cohen's κ).
- Compare human judgements to LLM-judge TBR to validate LLM-judge calibration. -->

### 6.9 Position Bias Ablation

On a 200-instance sample, compare:
- **Default:** shuffled document order (current implementation)
- **Fixed order:** correct document always first
- **Fixed order:** correct document always last

Measures residual position bias after shuffling and validates the shuffle mechanism.

---

## 7. Execution Plan

### Phase 1 — Data Preparation (Week 1)

| Step | Script | Output | Est. time |
|------|--------|--------|-----------|
| 1.1 Run TCB generation for ConfACT | `build_tcb_confact.sh --force-rebuild` | `tcb_humc_processed_cleaned.json` | 2–3 hrs |
| 1.2 Verify full-corpus RAMDocs TCB | `build_tcb_ramdocs.sh --skip-initial-build` | quality report | 30 min |
| 1.3 Verify full-corpus RAMDocs TCB | `build_and_validate_tcb_full.py` | quality report | 30 min |
| 1.4 Validate all TCB schemas | `validate_tcb_construction.py` | pass/fail per dataset | 10 min |

### Phase 2 — Full Ablation Run (Weeks 2–3)

```bash
bash experiments/run_experiments.sh \
  --dataset-root data/full \
  --judge-suite both \
  --sample-size 0 \          # full corpus, no sampling
  --sample-seed 42 \
  --agent-provider openai \
  --agent-model gpt-5.4-nano \
  --openai-judge-model gpt-5.4-mini \
  --anthropic-judge-model claude-haiku-4-5-20251001 \
  --judge-rounds 3 \
  --max-ablation-rounds 5 \
  --output-root results/full_run_seed42 \
  --parallel-datasets 1
```

Repeat with seeds 7, 123, 256, 999.  
Repeat with `--agent-model gpt-5.4-mini` for agent-family ablation.  
Repeat with `--anthropic-judge-model claude-sonnet-4-6` and `--openai-judge-model gpt-4o` for judge-family expansion.

**Estimated compute:**
- 4,200 instances × 5 datasets × 9 systems × 4 judge models × 5 seeds ≈ 3.78M LLM calls.
- With cross-suite debate cache reuse: debate runs once per (instance × agent-model × seed); judge-only passes add ~4× per debate = effective ~1.2M debate calls + ~3M judge calls.
- At $0.0001/call (haiku-class) → ~$120–300 total. OpenAI judge passes at $0.001/call → ~$400–600.

### Phase 3 — Counterfactual Swaps (Week 3)

```bash
python experiments/build_counterfactual_swaps.py \
  --results-dir results/full_run_seed42 \
  --filter-system madam_rag \
  --output data/full/swap_madam_incorrect.json

python experiments/run_experiment.py \
  --dataset swap_madam_incorrect \
  --judge-suite both \
  --output-dir results/swap_analysis_seed42

python experiments/analyze_counterfactual_swaps.py \
  --original-results results/full_run_seed42 \
  --swapped-results results/swap_analysis_seed42 \
  --json-out results/swap_analysis_seed42/flip_rates.json
```

### Phase 4 — Statistical Analysis (Week 4)

| Step | Script / tool | Output |
|------|---------------|--------|
| 4.1 Bootstrap CIs | `compute_proposal_metrics.py` + scipy bootstrap | TBR ± CI per cell |
| 4.2 Permutation tests | Custom (scipy.stats.permutation_test) | p-values per cell |
| 4.3 Bonferroni correction | Computed on 9 primary cells | Adjusted α, reject/fail-to-reject |
| 4.4 Judge divergence analysis | `analyze_judge_divergence.py` | Agreement rates, confusion matrix |
| 4.5 Counterfactual flip rates | `analyze_counterfactual_swaps.py` | Flip rate ± CI |
| 4.6 Round-ablation plot | `plot_round_ablation_results.py` | Accuracy vs. rounds figure |

### Phase 5 — Human Evaluation (Week 4–5)

- Sample 50 ConfACT-HumC instances with judge disagreement.
- Recruit 2 domain-expert annotators via Prolific/internal.
- Annotation tool: simple web form (existing output JSON schema).
- Compute Cohen's κ, compare to LLM-judge TBR.

### Phase 6 — Paper Writing (Week 5–6)

| Section | Key content |
|---------|-------------|
| Introduction | Source-laundering framing; RQ1–RQ5 |
| Related Work | Debate RAG, LLM-as-judge, tone bias, credibility |
| Methods | §2–§5 of this document |
| Results | Full ablation table; per-benchmark; CI bars; flip rates |
| Discussion | Ablation decomposition; judge-model effects; causal chain |
| Conclusion | Evidence-anchoring as a practical safety intervention |

---

## 8. Hypothesised Ablation Ladder

Based on current partial results and design intent, the hypothesised monotone ordering by TBR (lower is better):

```
eajp < eajp_no_privilege < eajp_scorecard_only ≈ cot_judge < ra_rag < credibility_hint < madam_rag < standard_rag < credibility_vote
```

**Expected breakdown of EAJP's TBR reduction:**
- ~40% from scorecard framing (structured evidence-first prompting; measured by `eajp_no_privilege − madam_rag`)
- ~60% from mechanical override (privilege rule; measured by `eajp − eajp_no_privilege`)

These proportions are testable with the full ablation run in Phase 2.

---

## References

[1] Wang et al., "Retrieval-Augmented Generation with Conflicting Evidence," 2025.  
[2] Tang et al., "Retrieval-Augmented Generation with Estimation of Source Reliability," 2025.  
[3] Gu et al., "A Survey on LLM-as-a-Judge," 2024.  
[4] Ye et al., "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge," 2024.  
[5] Hwang et al., "Can You Trick the Grader? Adversarial Persuasion of LLM Judges," 2025.  
[6] Agarwal and Khanna, "When Persuasion Overrides Truth in Multi-Agent LLM Debates," 2025.  
[7] Tian et al., "Overconfidence in LLM-as-a-Judge," 2025.  
