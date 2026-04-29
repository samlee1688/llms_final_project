# =========================================================
# STAGE 0 — Environment Setup
# ~15 min, $0
# =========================================================
pip install -r requirements.txt
python -c "
import src.config as c
print('OpenAI:', bool(c.OPENAI_API_KEY))
print('Anthropic:', bool(c.ANTHROPIC_API_KEY))
print('Gemini:', bool(c.GEMINI_API_KEY))
"

# =========================================================
# STAGE 1 — ConfACT Source Preparation
# ~15 s, $0 (local Python processing)
# =========================================================
bash experiments/build_confact_dataset.sh

ls data/full/ramdocs_test.json
ls data/full/confact/humc_processed_cleaned.json

# =========================================================
# STAGE 2 — TCB Construction
# ~2-4h, ~$3 (Gemini Flash for RAMDocs, Anthropic Haiku for ConfACT)
# =========================================================
python experiments/build_and_validate_tcb_full.py \
    --data-root data/full

# Cost breakdown:
#   RAMDocs (1,200 inst × 2 rewrites, Gemini Flash): ~$0.40
#   ConfACT  ( 287 inst × 2 rewrites, Anthropic Haiku, longer docs): ~$2.30
#   Quality scoring (BERTScore/ROUGE, local): $0
# NOTE: if TCBs already built, add --skip-initial-build and rerun validation only

# =========================================================
# STAGE 3 — Main Experiments
# ~10-16h, ~$216
# 4 systems × 4 datasets × both judge suites × 3 rounds
# Datasets: ramdocs, tcb_ramdocs, confact, tcb_confact
# =========================================================
./run_experiments.sh \
    --skip-install \
    --skip-build-datasets \
    --run-dataset-root data/full \
    --output-root results/full_run_seed42 \
    --judge-suite both \
    --judge-rounds 3 \
    --agent-provider openai \
    --parallel-datasets \
    --max-dataset-jobs 2 \
    --total-worker-budget 8 \
    --throttle-ms 200 \
    --sample-seed 42 \
    --resume

python experiments/compute_proposal_metrics.py \
    --results-dir results/full_run_seed42 \
    --out results/stats/proposal_metrics_seed42.json

python experiments/compute_statistical_tests.py \
    --results-dir results/full_run_seed42 \
    --out results/stats/ci_seed42.json \
    --n-bootstrap 2000 \
    --no-permutation-test \
    --ci-level 0.95

# =========================================================
# STAGE 4 — Full Ablation Ladder (5 new systems)
# ~2-4h, ~$19
# Skips eajp/madam_rag/standard_rag/eajp_scorecard_only (already in Stage 3)
# OAI judge only (ablation doesn't need Anthropic cross-comparison)
# Source datasets only — no TCB variants needed for ablation
# =========================================================
./experiments/run_full_ablation.sh \
    --seed 42 \
    --datasets "confact ramdocs" \
    --judge-suite openai \
    --agent-provider openai \
    --out-root results/ablation_seed42 \
    --skip-systems "eajp madam_rag standard_rag eajp_scorecard_only"

# Merge ablation results with main run for full 9-system CI table
python experiments/compute_statistical_tests.py \
    --results-dir results/ablation_seed42 results/full_run_seed42 \
    --out results/stats/ablation_full_ci.json \
    --no-permutation-test \
    --all-instances \
    --system-a eajp \
    --system-b madam_rag

# Optional: ablation round-level plot
python -m experiments.plot_round_ablation_results \
    --results-dir results/ablation_seed42 \
    --formats pdf png \
    --font-size 11 --dpi 300

# =========================================================
# STAGE 5 — TCB Quality Audit
# ~30 min, $0 (pure Python, no API calls)
# =========================================================
mkdir -p results/quality_extended

python experiments/analyze_tcb_readability.py \
    --tcb data/full/tcb_ramdocs_test.json \
    --out results/quality_extended/ramdocs_readability.json \
    --plot results/quality_extended/ramdocs_readability.png

python experiments/analyze_tcb_readability.py \
    --tcb data/full/confact/tcb_humc_processed_cleaned.json \
    --out results/quality_extended/confact_readability.json \
    --plot results/quality_extended/confact_readability.png

python -m src.quality_checker_extended \
    --tcb data/full/confact/tcb_humc_processed_cleaned.json \
    --out results/quality_extended/confact_extended_quality.json

# NLI entailment gate — run overnight or on GPU (~1h on MPS)
python -m src.quality_checker_extended \
    --tcb data/full/confact/tcb_humc_processed_cleaned.json \
    --out results/quality_extended/confact_nli.json \
    --use-nli \
    --device mps    # or: cpu, cuda

# =========================================================
# STAGE 6 — EAJP Judge Conflict Analysis
# ~15 min, $0 (reads Stage 3 outputs)
# =========================================================
python experiments/analyze_eajp_judge_conflicts.py \
    --results-dir results/full_run_seed42 \
    --out results/stats/eajp_conflict_matrix.json \
    --verbose

python experiments/validate_eajp_schema.py \
    --results-dir results/full_run_seed42 \
    --out results/stats/eajp_schema_validation.json

# =========================================================
# STAGE 7 — Counterfactual Causal Analysis
# ~1-2h, ~$8
# Re-runs MADAM-RAG on ~487 swapped instances (both suites)
# =========================================================
./experiments/run_counterfactual_pipeline.sh \
    --results-dir results/full_run_seed42 \
    --judge-suite both \
    --agent-provider openai \
    --dataset confact \
    --out-root results/counterfactual \
    --verbose

python experiments/analyze_counterfactual_swaps.py \
    --original-results results/full_run_seed42 \
    --swap-results results/counterfactual/swap_results \
    --swaps results/counterfactual/counterfactual_swaps.json \
    --out results/stats/counterfactual_flip_rates.json

# =========================================================
# STAGE 8 — Position Bias Check
# ~15 min, $0 (reads Stage 3 outputs)
# =========================================================
python experiments/run_position_bias_ablation.py \
    --results-dir results/full_run_seed42 \
    --system madam_rag \
    --out results/stats/position_bias.json \
    --verbose

# =========================================================
# STAGE 9 — Judge Expansion Study
# ~3-5h, ~$61
# sample-200 (cost control); covers 4 judge models × 2 systems
# Note: gpt-4o and claude-sonnet are expensive; see cost table below
# =========================================================
./experiments/run_judge_expansion.sh \
    --seed 42 \
    --dataset confact \
    --judges "gpt-4o-mini gpt-4o claude-haiku-4-5-20251001 claude-sonnet-4-6" \
    --systems "eajp madam_rag" \
    --sample-size 200 \
    --out-root results/judge_expansion

python experiments/analyze_judge_divergence.py \
    --results-dir results/judge_expansion \
    --out results/stats/judge_expansion_divergence.json

# =========================================================
# STAGE 10 — gpt-4o-mini Reproducibility Replication
# ~1h, ~$3
# sample-200, OAI judge only — confirms nano ≠ required
# =========================================================
./run_experiments.sh \
    --skip-install --skip-build-datasets \
    --run-dataset-root data/full \
    --output-root results/gpt4omini_replication \
    --agent-provider openai \
    --agent-model gpt-4o-mini \
    --judge-suite openai \
    --judge-rounds 3 \
    --sample-size 200 --sample-seed 42 \
    --total-worker-budget 6

python experiments/compute_proposal_metrics.py \
    --results-dir results/gpt4omini_replication \
    --out results/stats/gpt4omini_metrics.json

# =========================================================
# STAGE 11 — Final Statistics & Figures
# ~30 min, $0
# =========================================================
# Per-cell bootstrap CI with Bonferroni correction
python experiments/compute_statistical_tests.py \
    --results-dir results/full_run_seed42 \
    --out results/stats/ci_seed42_final.json \
    --n-bootstrap 2000 \
    --no-permutation-test \
    --ci-level 0.95 \
    --bonferroni-cells 9

# Ablation plot (PDF + PNG for paper)
python -m experiments.plot_round_ablation_results \
    --results-dir results/ablation_seed42 \
    --formats pdf png \
    --font-size 11 --dpi 300


Recommended Run Order (maximize cache reuse)

Day 1 (morning):
  Stage 0 → Stage 1 → Stage 2 (TCB build, 2-4h)

Day 1 (afternoon/evening):
  Stage 3 (main experiments, 10-16h, run overnight)
  └── Can run Stage 5 NLI audit overnight in parallel on MPS

Day 2 (morning, after Stage 3 completes):
  Stage 4 (ablation, 2-4h)       — reuses Stage 3 debate cache
  Stage 6 (conflict analysis)    — reads Stage 3 outputs, $0
  Stage 8 (position bias)        — reads Stage 3 outputs, $0

Day 2 (afternoon):
  Stage 5 (quality audit)        — reads TCB files, $0
  Stage 7 (counterfactual, 1-2h) — requires Stage 3 results
  Stage 9 (judge expansion, 3-5h) — independent

Day 3:
  Stage 10 (replication, 1h)     — independent
  Stage 11 (final statistics)    — reads all outputs, $0