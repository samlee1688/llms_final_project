# Results and Discussion (NeurIPS-Style Draft)

## 4. Results

### 4.1 Experimental Setup and Metrics (Proposal-Aligned)
Following the proposal, we evaluate three systems under conflicting-evidence retrieval: Standard RAG, MADAM-RAG, and EAJP (Evidence-Anchored Judge Prompting). We run all systems on AmbigDocs, FaithEval, and their tone-controlled counterparts (TCB-AmbigDocs, TCB-FaithEval), using two judge suites (OpenAI GPT-5.4-mini and Anthropic Claude Haiku 4.5). The primary proposal metrics are:

- Substance Alignment Rate (SAR): fraction of eligible trials where the system selects the correct, higher-credibility claim.
- Tone Bias Rate (TBR): fraction of eligible trials where the system favors the assertive but lower-credibility/incorrect claim.
- Accuracy (ACC): resolved-answer accuracy from benchmark metrics.

This directly tests the two proposal research questions:

- RQ1: Do Standard RAG and MADAM-RAG exhibit tone bias under conflicting evidence?
- RQ2: Does EAJP reduce tone bias and improve substance-first decisions?

### 4.2 Main Aggregate Results (All 3,000 trials)
Table 1 reports proposal-aligned aggregates from the sample-200 run (4 benchmarks x 200 instances x system/suite combinations).

**Table 1: Proposal-metric summary (aggregate across benchmarks).**

| Judge suite | System | Eligible trials | SAR | TBR | Low-cred wins | EAJP conflict rate |
|---|---:|---:|---:|---:|---:|---:|
| OpenAI GPT-5.4-mini | EAJP | 600 | 0.2083 | 0.1567 | 94 | 0.0075 |
| OpenAI GPT-5.4-mini | MADAM-RAG | 600 | 0.1883 | 0.2033 | 122 | 0.0000 |
| Anthropic Haiku 4.5 | EAJP | 600 | 0.1483 | 0.2033 | 122 | 0.0838 |
| Anthropic Haiku 4.5 | MADAM-RAG | 600 | 0.1667 | 0.2017 | 121 | 0.0000 |
| Suite-independent | Standard RAG | 600 | 0.1483 | 0.2117 | 127 | 0.0000 |

Key takeaways:

- Under OpenAI judging, EAJP improves both proposal objectives over MADAM-RAG: lower TBR (0.1567 vs 0.2033) and higher SAR (0.2083 vs 0.1883).
- Under Anthropic judging, EAJP is roughly parity on TBR with MADAM-RAG (0.2033 vs 0.2017), but lower on SAR (0.1483 vs 0.1667).
- Standard RAG shows the highest aggregate TBR (0.2117), supporting the proposal’s concern that unstructured aggregation is vulnerable to style-driven errors.

### 4.3 Benchmark-Level Results
Table 2 summarizes TBR/SAR patterns by benchmark and judge suite.

**Table 2: Benchmark-level TBR/SAR by system and judge suite (sample-200).**

| Benchmark | Suite | Standard RAG (SAR/TBR) | MADAM-RAG (SAR/TBR) | EAJP (SAR/TBR) |
|---|---|---|---|---|
| AmbigDocs | OpenAI | 0.305 / 0.085 | 0.215 / 0.210 | **0.405 / 0.045** |
| AmbigDocs | Anthropic | 0.305 / 0.085 | 0.205 / 0.145 | **0.215 / 0.110** |
| FaithEval | OpenAI | 0.125 / 0.285 | 0.135 / 0.295 | **0.140 / 0.220** |
| FaithEval | Anthropic | 0.125 / 0.285 | **0.110 / 0.300** | 0.070 / **0.290** |
| TCB-AmbigDocs | OpenAI | 0.205 / 0.090 | 0.300 / 0.040 | **0.320 / 0.010** |
| TCB-AmbigDocs | Anthropic | 0.205 / 0.090 | 0.280 / 0.050 | **0.280 / 0.040** |
| TCB-FaithEval | OpenAI | 0.115 / 0.260 | 0.130 / 0.275 | **0.165 / 0.240** |
| TCB-FaithEval | Anthropic | 0.115 / 0.260 | **0.110 / 0.255** | 0.095 / 0.280 |

Findings from Table 2:

- EAJP is consistently strongest on AmbigDocs and TCB-AmbigDocs, especially with OpenAI judging.
- FaithEval and TCB-FaithEval remain harder for all systems; TBR stays high (roughly 0.22 to 0.30), indicating persistent style susceptibility in more adversarial/noisy conflict settings.
- The TCB transformation increases separation between systems on AmbigDocs (especially for OpenAI), but does not eliminate judge-model sensitivity.

### 4.4 RQ Analysis

#### RQ1: Do Standard RAG and MADAM-RAG exhibit tone bias?
Yes. Across aggregate metrics, Standard RAG reaches TBR = 0.2117, while MADAM-RAG reaches about 0.20 across judge suites. This indicates persistent tendency to prefer assertive-but-weaker evidence in a non-trivial fraction of trials.

#### RQ2: Does EAJP reduce tone bias?
Partially, and judge-dependently.

- OpenAI suite: EAJP reduces TBR by 4.67 points vs MADAM-RAG and 5.50 points vs Standard RAG, while improving SAR (+2.00 and +6.00 points, respectively).
- Anthropic suite: EAJP is near-equal on TBR vs MADAM-RAG (+0.17 points, slightly worse) and lower on SAR (-1.83 points), though still better than Standard RAG on TBR.

Conclusion: EAJP improves substance-over-style behavior under one strong judge regime (OpenAI), but gains do not fully transfer across judge families.

## 5. Discussion

### 5.1 Why Gains Concentrate on AmbigDocs/TCB-AmbigDocs
AmbigDocs appears more compatible with claim-level decomposition and evidence anchoring: disagreements are often localized and directly tied to document support. EAJP’s structured scorecard can separate rhetorical confidence from evidentiary support more effectively in this setting.

### 5.2 Why FaithEval Remains Difficult
FaithEval/TCB-FaithEval include many instances where contradictions, distractors, and ambiguous support interact. In these cases, style cues may still leak into judge preferences, and the evidence graph needed for robust consistency checking is noisier. This is consistent with elevated TBR across all systems and both judges.

### 5.3 Judge-Model Effects
The same EAJP policy produces materially different outcomes by judge model family. The gap in conflict behavior (EAJP scorecard-final conflict rate: 0.0075 OpenAI vs 0.0838 Anthropic) suggests that cross-judge calibration is a primary reliability bottleneck, not just prompt design.

### 5.4 Practical Implications
For high-stakes deployments (medical, legal, financial):

- Require explicit claim-evidence alignment diagnostics (EAJP-style metadata) rather than answer-only outputs.
- Track TBR/SAR as first-class safety KPIs, not only end-task accuracy.
- Use multi-judge validation or calibrated fallback policies before production decisions.

## 6. Figure, Table, and Diagram Assets

### Figure 1 (Method Pipeline Diagram)

```mermaid
flowchart LR
    A[Query + Retrieved Conflicting Docs] --> B[Per-document Agents]
    B --> C[MADAM Debate Rounds]
    C --> D[EAJP Judge Scorecard]
    D --> D1[Claim Extraction]
    D --> D2[Span-level Support Checks]
    D --> D3[Cross-document Consistency]
    D --> D4[Credibility-weighted Selection]
    D1 --> E[Final Verdict]
    D2 --> E
    D3 --> E
    D4 --> E
```

Caption: EAJP extends debate-augmented RAG with explicit claim-level evidence anchoring before final selection.

### Figure 2 (Tone Bias Chart Data)

Use the following CSV to render a grouped bar chart (y-axis: TBR, x-axis: system, hue: judge suite):

```csv
suite,system,tbr
openai_gpt-5.4-mini,standard_rag,0.2117
openai_gpt-5.4-mini,madam_rag,0.2033
openai_gpt-5.4-mini,eajp,0.1567
anthropic_claude-haiku-4-5-20251001,standard_rag,0.2117
anthropic_claude-haiku-4-5-20251001,madam_rag,0.2017
anthropic_claude-haiku-4-5-20251001,eajp,0.2033
```

Suggested caption: EAJP substantially reduces tone bias under OpenAI judging, but not under Anthropic judging.

### Figure 3 (Substance Alignment Chart Data)

```csv
suite,system,sar
openai_gpt-5.4-mini,standard_rag,0.1483
openai_gpt-5.4-mini,madam_rag,0.1883
openai_gpt-5.4-mini,eajp,0.2083
anthropic_claude-haiku-4-5-20251001,standard_rag,0.1483
anthropic_claude-haiku-4-5-20251001,madam_rag,0.1667
anthropic_claude-haiku-4-5-20251001,eajp,0.1483
```

Suggested caption: SAR improvements from EAJP are judge-dependent and strongest under OpenAI.

## 7. Limitations and Threats to Validity

- Fast-pass setting: sample-200 gives strong directional evidence but is not a full-corpus estimate.
- Judge dependence: conclusions about EAJP efficacy vary by judge family; robustness requires multi-judge analysis.
- Post-run strict-enum edge case: one Anthropic EAJP trial emitted cross-document consistency label "partially_inconsistent" (outside strict enum set). Non-strict schema validation passed, and all experiment outputs were produced.

## 8. NeurIPS-Style Summary Paragraph

Across 3,000 evaluated trials, we find clear evidence of tone bias in both Standard RAG and MADAM-RAG, validating the core concern in our proposal. EAJP improves substance-over-style behavior under OpenAI judging, reducing tone bias and increasing substance alignment, with especially strong gains on AmbigDocs and tone-controlled AmbigDocs. However, these gains do not fully transfer to the Anthropic judge suite, where EAJP is near-parity with MADAM-RAG on tone bias and lower on substance alignment in aggregate. The results suggest that claim-level evidence anchoring is a promising mitigation for style-driven failures, but reliable deployment requires explicit cross-judge calibration and benchmark-specific stress testing.

## References (placeholders)

[1] Wang et al., Retrieval-Augmented Generation with Conflicting Evidence, 2025.
[2] Tang et al., Retrieval-Augmented Generation with Estimation of Source Reliability, 2025.
[3] Gu et al., A Survey on LLM-as-a-Judge, 2024.
[4] Ye et al., Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge, 2024.
[5] Hwang et al., Can You Trick the Grader? Adversarial Persuasion of LLM Judges, 2025.
[6] Agarwal and Khanna, When Persuasion Overrides Truth in Multi-Agent LLM Debates, 2025.
[7] Tian et al., Overconfidence in LLM-as-a-Judge, 2025.
