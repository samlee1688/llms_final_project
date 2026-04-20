## 4. Results and Discussion

### 4.1 Experimental Setup
We evaluate three systems under a controlled, conflict-focused RAG benchmark pipeline: Standard RAG, MADAM-RAG, and EAJP. EAJP operationalizes an evidence-anchored judging protocol that requires an evidence-first scorecard and applies a scorecard-privilege rule under verdict conflict [CIT-EAJP-DESIGN], [CIT-DEBATE-OVERSIGHT].

All reported results come from the sample-100 run (`results/fast_pass_sample100`) with fixed seed and matched data preprocessing. We report outcomes across four benchmark slices (`ambigdocs`, `faitheval`, `tcb_ambigdocs`, `tcb_faitheval`) and two judge suites (`openai_gpt-5.4-mini`, `anthropic_claude-haiku-4-5-20251001`).

We focus on proposal-aligned metrics:

1. **Tone Bias Rate (TBR):** proportion of eligible conflicting-evidence trials in which the lower-credibility option is selected.
2. **Substance Alignment Rate (SAR):** proportion of eligible trials in which the better-evidenced option is selected.

These metrics are computed from the proposal-level aggregation and complemented by per-benchmark summaries [CIT-METRIC-DEF], [CIT-RAG-BIAS].

### 4.2 Main Findings

#### RQ1: Do Standard RAG and MADAM-RAG exhibit tone bias under conflicting evidence?
**Yes.**

Proposal-level aggregates indicate substantial bias in both baselines:

- Standard RAG: TBR = 0.2200 (300 eligible trials)
- MADAM-RAG (OpenAI suite): TBR = 0.2100 (300 eligible trials)
- MADAM-RAG (Anthropic suite): TBR = 0.2067 (300 eligible trials)

The observed rates (21-22%) indicate that assertive but weaker-evidence content can systematically influence final selection, consistent with source-laundering concerns in debate-style RAG [CIT-SOURCE-LAUNDERING], [CIT-MULTIAGENT-RAG].

#### RQ2: Does EAJP reduce this bias?
**Partially; effect size is suite-dependent.**

Against MADAM-RAG, EAJP changes TBR as follows:

- OpenAI suite: 0.2100 -> 0.1500 (absolute -0.0600; relative -28.6%)
- Anthropic suite: 0.2067 -> 0.2000 (absolute -0.0067; relative -3.2%)

Thus, EAJP materially reduces bias in the OpenAI-judged condition, while delivering only a marginal reduction in the Anthropic-judged condition.

### 4.3 Benchmark-Level Behavior
Per-benchmark trends help explain the aggregate pattern.

**OpenAI suite:** EAJP consistently lowers TBR relative to MADAM-RAG.

- `ambigdocs`: 0.22 -> 0.05
- `faitheval`: 0.30 -> 0.21
- `tcb_ambigdocs`: 0.05 -> 0.02
- `tcb_faitheval`: 0.28 -> 0.22

**Anthropic suite:** improvements are mixed.

- `ambigdocs`: 0.17 -> 0.15 (improves)
- `faitheval`: 0.31 -> 0.26 (improves)
- `tcb_ambigdocs`: 0.04 -> 0.05 (slight regression)
- `tcb_faitheval`: 0.27 -> 0.29 (regression)

This interaction suggests that evidence-anchoring efficacy is not purely algorithmic; it also depends on judge-model behavior and instruction-following characteristics [CIT-JUDGE-MODEL-VARIANCE], [CIT-LLM-EVAL-ROBUSTNESS].

### 4.4 Interpretation
The results support three claims.

1. **Bias is present in both single-pass and debate-based baselines.** Debate alone does not eliminate style-induced preference shifts.
2. **Evidence anchoring can mitigate bias.** Under one judge family (OpenAI), EAJP achieves a substantial reduction in low-credibility wins.
3. **Mitigation is not universally stable.** Under another judge family (Anthropic), gains are weaker and benchmark-dependent.

Taken together, EAJP appears to be an effective but non-universal intervention: it improves robustness in some operational settings while requiring calibration in others.

### 4.5 Practical Implications
For deployment contexts where evidence reliability is safety-critical (e.g., policy analysis, clinical triage support, legal or compliance workflows), the findings imply:

1. **Adopt evidence-anchored judging on high-risk queries.** Routing contentious cases through EAJP-like protocols can reduce style-driven errors.
2. **Monitor low-credibility win rate as a first-class KPI.** TBR should be tracked over time and stratified by judge model and data regime.
3. **Treat judge-model choice as a governance decision.** The same mitigation logic can yield materially different outcomes across judge families.
4. **Preserve structured audit traces.** Scorecard fields and conflict-resolution metadata improve post-hoc accountability and incident analysis [CIT-AUDITABLE-AI].

### 4.6 Limitations and Threats to Validity
Several constraints temper external validity.

1. **Sample size and seed:** this analysis is based on one sample-100 run and fixed seed.
2. **Model scope:** agents are fixed to one family (`gpt-5.4-nano`), while only two judge suites are tested.
3. **Inference statistics:** point estimates are reported without confidence intervals or formal hypothesis tests.
4. **Data dependence:** benchmark slices differ in conflict construction and may respond differently to protocol constraints.

Future work should include multi-seed evaluation, bootstrap confidence intervals, judge-family ablations, and full-corpus replication [CIT-REPRODUCIBILITY], [CIT-STAT-EVAL].

### 4.7 Summary
This run provides clear evidence that tone bias persists in Standard RAG and MADAM-RAG under conflicting evidence. EAJP reduces this bias overall, with strong improvements in OpenAI-judged settings and weaker, mixed effects in Anthropic-judged settings. The intervention is therefore promising for safety-oriented RAG governance, but its effectiveness should be treated as model- and setting-contingent rather than guaranteed.
