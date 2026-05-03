#!/usr/bin/env bash
set -euo pipefail

# Quality benchmark using either the sampled (default) or full validated datasets.
#
# Uses larger debate-context budgets for higher fidelity, strict two-pass EAJP,
# and the cross-suite MADAM debate-reuse optimization.
#
# Mode selection (env var):
#   DATASET_MODE=sample  (default) — data/sample200_gemini_preprocess
#   DATASET_MODE=full               — data/full  (requires build_and_validate_tcb_full.py)
#
# Prerequisites:
#   sample mode:  ./run_preprocess_sample200_gemini.sh  (generates sample200 artifacts)
#   full mode:    python experiments/build_and_validate_tcb_full.py  (downloads + validates)
#
# Usage examples:
#   ./run_quality_fast_pass_preprocessed_sample200.sh
#   DATASET_MODE=full ./run_quality_fast_pass_preprocessed_sample200.sh
#   DATASET_MODE=full OUTPUT_ROOT=results/full_quality ./run_quality_fast_pass_preprocessed_sample200.sh
#   DATASET_MODE=sample SAMPLE_SIZE=50 ./run_quality_fast_pass_preprocessed_sample200.sh --judge-rounds 1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Mode-dependent defaults
# ---------------------------------------------------------------------------
DATASET_MODE="${DATASET_MODE:-sample}"

if [[ "$DATASET_MODE" == "full" ]]; then
  DEFAULT_DATASET_ROOT="data/full"
  DEFAULT_OUTPUT_ROOT="results/quality_full"
  # Full mode: no fast-mode sample cap (run all instances)
  USE_FAST_MODE=0
else
  DEFAULT_DATASET_ROOT="data/sample200_gemini_preprocess"
  DEFAULT_OUTPUT_ROOT="results/quality_fast_pass_preprocessed_sample200"
  USE_FAST_MODE=1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

# Verify dataset root exists and has expected files
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: Dataset root '$DATASET_ROOT' does not exist." >&2
  if [[ "$DATASET_MODE" == "full" ]]; then
    echo "Run: python experiments/build_and_validate_tcb_full.py" >&2
  else
    echo "Run: ./run_preprocess_sample200_gemini.sh" >&2
  fi
  exit 1
fi

echo "[info] DATASET_MODE=$DATASET_MODE"
echo "[info] DATASET_ROOT=$DATASET_ROOT"
echo "[info] OUTPUT_ROOT=$OUTPUT_ROOT"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

# Common flags
COMMON_FLAGS=(
  --skip-install
  --skip-build-datasets
  --run-dataset-root "$DATASET_ROOT"
  --ramdocs-split test
  --output-root "$OUTPUT_ROOT"
  --judge-suite both
  --judge-rounds 3
  --sample-seed "$SAMPLE_SEED"
  --parallel-datasets
  --max-dataset-jobs 3
  --total-worker-budget 8
  --throttle-ms 200
  --resume
  --madam-max-parallel-agents 4
  --madam-compact-doc-chars 10000
  --madam-argument-summary-chars 1000
  --madam-early-stop-patience 1
  --no-eajp-short-circuit
  --judge-only-on-contention
  --validate-eajp-schema
  --eajp-schema-strict-enums
  --compute-proposal-metrics
  --agent-provider openai
  --agent-model gpt-5.4-nano
  --standard-rag-provider openai
  --standard-rag-model gpt-5.4-nano
  --madam-agent-provider openai
  --madam-agent-model gpt-5.4-nano
  --eajp-agent-provider openai
  --eajp-agent-model gpt-5.4-nano
  --openai-judge-model gpt-5.4-mini
  --anthropic-judge-model claude-haiku-4-5-20251001
)

# Sample-mode-only flags
if [[ "$USE_FAST_MODE" -eq 1 ]]; then
  COMMON_FLAGS+=(--fast-mode --sample-size "$SAMPLE_SIZE")
fi

LLM_CACHE_ENABLED="${LLM_CACHE_ENABLED:-1}" \
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-6}" \
LLM_BACKOFF_BASE_SECONDS="${LLM_BACKOFF_BASE_SECONDS:-1.5}" \
./run_experiments.sh "${COMMON_FLAGS[@]}" "$@"
