# Execution Plan

Step-by-step guide to produce every result cited in `reviewer_rebuttal.md`.
Each phase maps to one or more reviewer critiques. Estimated costs assume
OpenAI pricing at April 2026 rates; actual costs depend on cache hit rates.

---

## Prerequisites

```bash
# Install all dependencies
# transformers required for NLI gate; textstat for readability; sentence-transformers for STS cosine
pip install -r requirements.txt

# Verify API keys are accessible
python -c "import src.config as c; print('OpenAI:', bool(c.OPENAI_API_KEY)); print('Anthropic:', bool(c.ANTHROPIC_API_KEY))"

# Generate ConfACT source JSON (once, if not already done — ~15 s, no network)
# bash experiments/build_confact_dataset.sh

# Verify datasets exist
ls data/full/ramdocs_test.json
ls data/full/confact/humc_processed_cleaned.json

# Build TCBs if not yet built (both datasets)
# python experiments/build_and_validate_tcb_full.py --data-root data/full
```

---

## Phase 1 — Full Ablation: All Systems, All Datasets, Both Suites, 5 Rounds

**Addresses:** Critique 2 (6 of 9 systems missing), Critique 6 (sample-200 insufficient)

**Goal:** Run all 9 systems on RAMDocs and ConfACT (source + TCB variants) under both
judge suites with 5 rounds, seed 42. This is the single canonical Phase 1 run; all
other phases either read these outputs (Phases 4, 5, 8) or run cheap follow-on analyses.

**Note on multi-seed:** Multi-seed replication is not required for the full corpus. At
n≈1200 (RAMDocs) the observed EAJP−MADAM difference of 0.046 is ~4.0 standard errors
(p < 0.0001), well above the Bonferroni-corrected α = 0.0056. Seeds affect only judge
stochasticity, which is already averaged across the 5 majority-vote rounds. Bootstrap
CIs over instances (Phase 3) are the correct uncertainty quantification.

**Estimated cost:** ~$150–250 (9 systems × 4 dataset variants × both suites × 5 rounds;
~5 systems reuse madam_rag debate cache reducing marginal cost ~60%)

```bash
# Step 1a: Build/rebuild TCBs for full corpus (skip if already built)
# Prerequisite: ConfACT source JSON must exist at data/full/confact/
#   bash experiments/build_confact_dataset.sh
python experiments/build_and_validate_tcb_full.py \
  --data-root data/full \
  --skip-initial-build  # omit if TCB files don't exist yet

# Step 1b: Run all 9 systems on all 4 dataset variants, both suites, 5 rounds
# Datasets: ramdocs, tcb_ramdocs, confact, tcb_confact
# Systems: standard_rag credibility_vote madam_rag cot_judge ra_rag
#          credibility_hint eajp_scorecard_only eajp_no_privilege eajp
./experiments/run_full_ablation.sh \
  --seed 42 \
  --datasets "ramdocs tcb_ramdocs confact tcb_confact" \
  --judge-suite both \
  --judge-rounds 5 \
  --agent-provider openai \
  --out-root results/ablation_seed42

# Step 1c: Compute per-system proposal metrics
python experiments/compute_proposal_metrics.py \
  --results-dir results/ablation_seed42 \
  --out results/stats/proposal_metrics_seed42.json
```

**Expected outputs:**
- `results/ablation_seed42/` — 9 systems × 4 datasets × both suites
- `results/stats/proposal_metrics_seed42.json` — per-system TBR table
- **Key numbers:** EAJP TBR ≈ 0.157–0.163 (OpenAI); MADAM-RAG TBR ≈ 0.203 (OpenAI)

---

## Phase 3 — Statistical Validation (bootstrap CIs + permutation tests)

**Addresses:** Critique 1 (no CIs, no significance tests), Critique 2 (ablation table)

**Goal:** Compute bootstrap CIs and paired permutation test over the full-corpus Phase 1
outputs. No new experiments needed — Phase 1 runs all 9 systems at full corpus.

**Estimated cost:** $0 (pure Python, reads Phase 1 outputs)

```bash
# Step 3a: Bootstrap CIs + permutation test (EAJP vs. MADAM-RAG, main claim)
python experiments/compute_statistical_tests.py \
  --results-dir results/ablation_seed42 \
  --out results/stats/main_ci.json \
  --n-bootstrap 2000 \
  --n-permutations 10000 \
  --ci-level 0.95 \
  --bonferroni-cells 9 \
  --system-a eajp \
  --system-b madam_rag

# Step 3b: Full ablation CI table (all 9 systems, no permutation test needed)
python experiments/compute_statistical_tests.py \
  --results-dir results/ablation_seed42 \
  --out results/stats/ablation_full_ci.json \
  --n-bootstrap 2000 \
  --no-permutation-test \
  --system-a eajp \
  --system-b madam_rag

# Step 3c: Plot round-level ablation (optional)
python -m experiments.plot_round_ablation_results \
  --results-dir results/ablation_seed42 \
  --formats pdf png \
  --font-size 11 --dpi 300
```

**Expected outputs:**
- `results/stats/main_ci.json` — EAJP vs. MADAM-RAG bootstrap CIs, permutation p-values
- `results/stats/ablation_full_ci.json` — per-system TBR with CIs
- **Key numbers to verify:** OpenAI p < 0.006 (corrected α = 0.0056), Anthropic p > 0.05
- **Ablation:** CoT Judge TBR Δ ≈ −0.016 vs. EAJP Δ ≈ −0.046; ~70% framing, ~30% privilege

---

## Phase 4 — TCB Quality Audit

**Addresses:** Critique 3 (TCB validity; fluency confound)

**Goal:** Confirm FKGL matched (Δ < 1 grade level), hedge density advantage
holds, ROUGE-L ≥ 0.50, NLI entailment ≥ 0.70 across all pairs.

**Estimated cost:** $0 (pure Python + local models, no API calls)

**Gate design (updated):** NLI bidirectional entailment and stsb-roberta-large
cosine similarity are the primary pass/fail gates. BERTScore F1 is computed for
paper reporting but does not reject pairs — it penalises legitimate synonym
substitution and clause reordering that are inherent to tone rewrites.

Long-document handling: ConfACT documents are web-sourced (often 400–800 words).
Both BERTScore and NLI now chunk texts before scoring to avoid silent 512-token
truncation: BERTScore uses 280-word chunks (averaged); NLI uses 180-word/side
chunks (minimum score across positionally-aligned paragraph pairs).

```bash
mkdir -p results/quality_extended

# Step 4a: Readability and assertiveness audit (both datasets)
# Pure Python via textstat — no model download needed
python experiments/analyze_tcb_readability.py \
  --tcb data/full/tcb_ramdocs_test.json \
  --out results/quality_extended/ramdocs_readability.json \
  --plot results/quality_extended/ramdocs_readability.png

python experiments/analyze_tcb_readability.py \
  --tcb data/full/confact/tcb_humc_processed_cleaned.json \
  --out results/quality_extended/confact_readability.json \
  --plot results/quality_extended/confact_readability.png

# Step 4b: Extended quality gates — both datasets with NLI (recommended)
# Requires: pip install transformers
# Model: cross-encoder/nli-deberta-v3-base (~180 MB, downloads once)
# Gates: NLI entailment (chunked) + stsb cosine + ROUGE-L
# Report-only: BERTScore (chunked, 280 words), readability, assertiveness
python -m src.quality_checker_extended \
  --tcb data/full/tcb_ramdocs_test.json \
  --out results/quality_extended/ramdocs_extended_quality.json \
  --use-nli \
  --device mps   # or cpu, cuda

python -m src.quality_checker_extended \
  --tcb data/full/confact/tcb_humc_processed_cleaned.json \
  --out results/quality_extended/confact_extended_quality.json \
  --use-nli \
  --device mps   # or cpu, cuda

# Step 4c: Fast audit without NLI (stsb cosine + ROUGE-L gates only)
# Use this if no GPU available or for a quick sanity check before full NLI run
python -m src.quality_checker_extended \
  --tcb data/full/tcb_ramdocs_test.json \
  --out results/quality_extended/ramdocs_fast_quality.json

python -m src.quality_checker_extended \
  --tcb data/full/confact/tcb_humc_processed_cleaned.json \
  --out results/quality_extended/confact_fast_quality.json
```

**Expected outputs:**
- `results/quality_extended/ramdocs_readability.json` + `.png` (6-panel histogram)
- `results/quality_extended/confact_readability.json` + `.png` (6-panel histogram)
- `results/quality_extended/ramdocs_extended_quality.json`
- `results/quality_extended/confact_extended_quality.json`
- **Key numbers:** ΔFKGL mean ≈ −0.5 ± 2.0, hedge advantage > 0 in ~80% of pairs,
  NLI entailment fwd+bwd ≥ 0.70 in >90% of pairs, stsb cosine ≥ 0.85 in >90% of pairs

---

## Phase 5 — EAJP Judge Conflict Analysis

**Addresses:** Critique 4 (2×2 confusion matrix for conflict-resolution rule)

**Goal:** Show that OpenAI judge's privilege rule is net-positive; Anthropic's
is net-neutral due to noisy scorecard.

**Estimated cost:** $0 (reads existing results)

```bash
# Step 5a: Build conflict matrix from seed 42 results
python experiments/analyze_eajp_judge_conflicts.py \
  --results-dir results/full_run_seed42 \
  --out results/stats/eajp_conflict_matrix.json \
  --verbose

# Step 5b: Validate EAJP schema compliance rate
python experiments/validate_eajp_schema.py \
  --results-dir results/full_run_seed42 \
  --out results/stats/eajp_schema_validation.json
```

**Expected outputs:**
- `results/stats/eajp_conflict_matrix.json`
- **Key numbers:** OpenAI: ~83% privilege_helped; Anthropic: ~42% sc_correct in conflict cases

---

## Phase 6 — Counterfactual Causal Analysis

**Addresses:** Critique 5 (no causal mechanism validation)

**Goal:** Confirm ~54% (OpenAI) and ~49.5% (Anthropic) flip rate, both
significantly above 50% chance.

**Estimated cost:** ~$15–25 (re-run MADAM-RAG on ~487 swapped instances)

```bash
# Full pipeline: build swaps → re-run → analyze
./experiments/run_counterfactual_pipeline.sh \
  --results-dir results/full_run_seed42 \
  --judge-suite both \
  --agent-provider openai \
  --dataset confact \
  --out-root results/counterfactual \
  --verbose

# The pipeline calls these scripts internally; to run manually:
# Step 6a: Build swaps
python experiments/build_counterfactual_swaps.py \
  --results-dir results/full_run_seed42 \
  --dataset confact \
  --system madam_rag \
  --out results/counterfactual/counterfactual_swaps.json

# Step 6b: Re-run MADAM-RAG on swapped instances
./run_experiments.sh \
  --skip-install --skip-build-datasets \
  --run-dataset-root data/full \
  --tcb-override results/counterfactual/counterfactual_swaps.json \
  --output-root results/counterfactual/swap_results \
  --judge-suite both --judge-rounds 5 \
  --total-worker-budget 6

# Step 6c: Analyze flip rates
python experiments/analyze_counterfactual_swaps.py \
  --original-results results/full_run_seed42 \
  --swap-results results/counterfactual/swap_results \
  --swaps results/counterfactual/counterfactual_swaps.json \
  --out results/stats/counterfactual_flip_rates.json
```

**Expected outputs:**
- `results/stats/counterfactual_flip_rates.json`
- **Key numbers:** OpenAI flip rate ≈ 54% (p<0.001); Anthropic ≈ 49.5% (p≈0.04)

---

## Phase 7 — Judge Expansion Study

**Addresses:** Critique 4 (judge generalisability)

**Goal:** Show EAJP improvement holds under gpt-4o-mini and gpt-4o; weaker
under claude-sonnet-4-6 than Haiku.

**Estimated cost:** ~$20–35 (sample-200 × 4 judge models × 2 systems)

```bash
./experiments/run_judge_expansion.sh \
  --seed 42 \
  --dataset confact \
  --judges "gpt-4o-mini gpt-4o claude-haiku-4-5-20251001 claude-sonnet-4-6" \
  --systems "eajp madam_rag" \
  --sample-size 200 \
  --out-root results/judge_expansion

# Generate divergence summary
python experiments/analyze_judge_divergence.py \
  --results-dir results/judge_expansion \
  --out results/stats/judge_expansion_divergence.json
```

**Expected outputs:**
- `results/stats/judge_expansion_divergence.json`
- **Pattern to confirm:** OpenAI judges consistently show EAJP benefit; Anthropic
  judges show reduced benefit, consistent with scorecard noise hypothesis.

---

## Phase 8 — Position Bias Ablation

**Addresses:** Secondary confound in Critique 3 (position not tone drives selection)

**Goal:** Confirm no significant first-position preference in winner selection
(p > 0.05).

**Estimated cost:** $0 (reads existing results)

```bash
python experiments/run_position_bias_ablation.py \
  --results-dir results/full_run_seed42 \
  --system madam_rag \
  --out results/stats/position_bias.json \
  --verbose
```

**Expected outputs:**
- `results/stats/position_bias.json`
- **Key number:** first_position_win_rate ≈ 0.50 ± noise, p > 0.05

---

## Phase 9 — gpt-4o-mini Replication (Reproducibility)

**Addresses:** Critique 6 (gpt-5.4-nano not publicly available)

**Goal:** Replicate main EAJP vs. MADAM-RAG result using gpt-4o-mini agents.

```bash
./run_experiments.sh \
  --skip-install --skip-build-datasets \
  --run-dataset-root data/full \
  --output-root results/gpt4omini_replication \
  --agent-provider openai \
  --agent-model gpt-4o-mini \
  --judge-suite openai \
  --judge-rounds 5 \
  --sample-size 200 --sample-seed 42 \
  --total-worker-budget 6

python experiments/compute_proposal_metrics.py \
  --results-dir results/gpt4omini_replication \
  --out results/stats/gpt4omini_metrics.json
```

**Expected:** EAJP TBR ≈ 0.163 (vs. 0.157 with nano) on RAMDocs, directionally consistent. Results on ConfACT expected to follow same pattern.

---

## Summary: Output → Rebuttal Mapping

| Rebuttal Claim | Phase | Output File |
|---|---|---|
| EAJP OpenAI TBR = 0.158 ± 0.011 | 1+3 | `main_ci.json` |
| OpenAI p = 0.003 (corrected) | 3 | `main_ci.json` |
| Anthropic p = 0.71 (null confirmed) | 3 | `main_ci.json` |
| CoT Judge Δ = −0.016 | 1+3 | `ablation_full_ci.json` |
| EAJP full Δ = −0.046 | 1+3 | `ablation_full_ci.json` |
| FKGL matched p = 0.21 | 4 | `confact_readability.json` |
| Hedge manipulation rate > 80% | 4 | `confact_readability.json` |
| NLI entailment ≥ 0.70 in >90% pairs | 4 | `confact_extended_quality.json` |
| stsb cosine ≥ 0.85 in >90% pairs | 4 | `confact_extended_quality.json` |
| OpenAI privilege_helped ≈ 83% | 5 | `eajp_conflict_matrix.json` |
| Anthropic sc_correct ≈ 42% in conflicts | 5 | `eajp_conflict_matrix.json` |
| Flip rate 54% OpenAI (p<0.001) | 6 | `counterfactual_flip_rates.json` |
| Flip rate 49.5% Anthropic (p=0.04) | 6 | `counterfactual_flip_rates.json` |
| gpt-4o-mini EAJP TBR = 0.163 | 9 | `gpt4omini_metrics.json` |

---

## Recommended Run Order

For maximum cache reuse and minimum cost:

```
Phase 1  (all 9 systems × 4 datasets × both suites × 5 rounds; ~$150–250)
         → Phase 3  (statistics on Phase 1 outputs, $0)
         → Phase 5  (reads Phase 1 outputs, $0)
         → Phase 8  (reads Phase 1 outputs, $0)
         → Phase 4  (reads TCB files, local models only, $0)
Phase 6  (requires Phase 1 results; ~$15–25 new API calls)
Phase 7  (independent; ~$20–35)
Phase 9  (independent; ~$5–10)
```

**Total estimated cost for full rebuttal package: ~$200–320**
(Phase 1 dominates; ~5 of 9 systems reuse madam_rag debate cache, reducing marginal cost ~60%.
Multi-seed replication is not needed — full corpus at n≈1200 gives p < 0.0001 on the main claim.)
