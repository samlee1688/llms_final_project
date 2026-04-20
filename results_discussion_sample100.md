# Results and Discussion (Sample-100 Run)

## 1. Experimental Methods

### 1.1 Study Objective
This run evaluates two proposal research questions:

1. Do Standard RAG and MADAM-RAG exhibit tone bias under conflicting evidence (favoring assertive/low-credibility sources over better-evidenced neutral/high-credibility sources)?
2. Does EAJP reduce this bias?

### 1.2 Systems Compared

1. Standard RAG (single-pass baseline)
2. MADAM-RAG (multi-agent debate + judge)
3. EAJP (evidence-anchored judge protocol applied to MADAM)

### 1.3 Data and Run Configuration

1. Dataset root: `data/sample100_gemini_preprocess`
2. Per-dataset sample size: 100 (seed=42)
3. Benchmarks included:
1. `ambigdocs`
2. `faitheval`
3. `tcb_ambigdocs`
4. `tcb_faitheval`
4. Judge suites:
1. `openai_gpt-5.4-mini`
2. `anthropic_claude-haiku-4-5-20251001`
5. Agents/modeling setup:
1. Agent model: `gpt-5.4-nano`
2. Standard RAG model: `gpt-5.4-nano`
3. MADAM/EAJP judges: OpenAI and Anthropic suites above

### 1.4 Metrics Used

Proposal-oriented analysis is from `proposal_metrics.json` with:

1. `tone_bias_rate`: fraction of eligible trials where the model selected lower-credibility evidence
2. `substance_alignment_rate`: fraction selecting better-evidenced/correct answer among eligible trials

Per-benchmark judge metrics are additionally summarized via `benchmark_summary.json` and include TBR/SAR/ACC by benchmark, system, and suite.

## 2. Results

### 2.1 Primary RQ Results (Proposal Metrics)

#### RQ1: Do Standard RAG and MADAM-RAG exhibit tone bias?
Yes.

| Suite | System | Eligible trials | Tone bias rate |
|---|---:|---:|---:|
| suite_independent | Standard RAG | 300 | 0.2200 |
| openai_gpt-5.4-mini | MADAM-RAG | 300 | 0.2100 |
| anthropic_claude-haiku-4-5-20251001 | MADAM-RAG | 300 | 0.2067 |

Interpretation: all baseline/debate systems show non-trivial bias (~20-22% of eligible trials), indicating systematic susceptibility to assertive low-credibility evidence.

#### RQ2: Does EAJP reduce this bias?
Yes, but strength depends on judge suite.

| Suite | MADAM-RAG TBR | EAJP TBR | Absolute delta (EAJP - MADAM) | Relative reduction |
|---|---:|---:|---:|---:|
| openai_gpt-5.4-mini | 0.2100 | 0.1500 | -0.0600 | 28.6% |
| anthropic_claude-haiku-4-5-20251001 | 0.2067 | 0.2000 | -0.0067 | 3.2% |

Interpretation:

1. Under OpenAI judging, EAJP substantially reduces tone bias.
2. Under Anthropic judging, the reduction is small.
3. Overall, EAJP helps, but effect size is model/suite dependent.

### 2.2 Benchmark-Level Pattern Check (TBR from benchmark summary)

OpenAI suite:

1. `ambigdocs`: MADAM 0.22 -> EAJP 0.05 (large improvement)
2. `faitheval`: MADAM 0.30 -> EAJP 0.21 (moderate improvement)
3. `tcb_ambigdocs`: MADAM 0.05 -> EAJP 0.02 (small improvement)
4. `tcb_faitheval`: MADAM 0.28 -> EAJP 0.22 (moderate improvement)

Anthropic suite:

1. `ambigdocs`: MADAM 0.17 -> EAJP 0.15 (small improvement)
2. `faitheval`: MADAM 0.31 -> EAJP 0.26 (moderate improvement)
3. `tcb_ambigdocs`: MADAM 0.04 -> EAJP 0.05 (slight regression)
4. `tcb_faitheval`: MADAM 0.27 -> EAJP 0.29 (regression)

This explains the aggregate outcome: robust gains in one suite, mixed behavior in the other.

## 3. Discussion

### 3.1 Answer to Proposal Questions

1. Q1 (bias present?): Supported. Standard RAG and MADAM-RAG both exhibit meaningful tone bias under conflict.
2. Q2 (EAJP reduces bias?): Partially supported overall, strongly supported in OpenAI-judged setting.

### 3.2 Why EAJP Likely Helps
EAJP enforces an evidence-first scorecard and a conflict-resolution rule that can privilege the scorecard outcome over argument rhetoric. This directly targets style-driven override failure modes (source laundering by assertiveness).

### 3.3 Why Gains Are Not Uniform
Possible drivers of heterogeneity across suites:

1. Judge model sensitivity to rhetorical style vs. extracted evidence structure
2. Different reliability in following structured scorecard/final-verdict protocol
3. Dataset-specific interaction effects in FaithEval-style inconsistencies

### 3.4 Validity and Limitations

1. Sample size is moderate (100 per benchmark split), not full-corpus.
2. Single agent model family (`gpt-5.4-nano`) was used for generators/debaters.
3. Results are point estimates; no bootstrap CIs or significance tests are included in this run.
4. Some benchmark/suite cells show regressions, so EAJP should not yet be considered universally superior.

## 4. Practical Applications

### 4.1 Safer RAG in High-Stakes Domains
In domains such as policy, healthcare, finance, and legal analysis, confidence-heavy but weakly supported claims can be harmful. An EAJP-like evidence-first decision protocol can reduce this failure mode, especially where OpenAI-style judge behavior is used.

### 4.2 Production Guardrails
Practical deployment pattern:

1. Keep a baseline pipeline for throughput.
2. Route contention/high-risk queries to EAJP.
3. Log scorecard/final conflicts and low-credibility-win events as monitoring KPIs.

### 4.3 Governance and Auditability
EAJP metadata provides auditable traces (scorecard fields, preliminary/final winner, conflict override). This improves explainability and post-hoc review compared with opaque end-to-end generation.

## 5. Conclusion

From this sample-100 run, tone bias is clearly present in Standard RAG and MADAM-RAG. EAJP reduces this bias overall, with substantial gains in OpenAI-judged conditions and mixed outcomes in Anthropic-judged conditions. The evidence supports adopting EAJP-style evidence anchoring as a practical mitigation, while recommending further multi-seed and full-corpus validation before broad generalization.
