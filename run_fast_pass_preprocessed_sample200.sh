#!/usr/bin/env bash
set -euo pipefail

# Fast-pass benchmark using preprocessed sampled datasets.
# Default settings prioritize resilience and throughput while preserving determinism.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/fast_pass_preprocessed_sample200}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

LLM_CACHE_ENABLED="${LLM_CACHE_ENABLED:-1}" \
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-6}" \
LLM_BACKOFF_BASE_SECONDS="${LLM_BACKOFF_BASE_SECONDS:-1.5}" \
./run_experiments.sh \
  --skip-install \
  --skip-build-datasets \
  --run-dataset-root data/sample200_gemini_preprocess \
  --ambigdocs-split validation \
  --faitheval-split test \
  --output-root "$OUTPUT_ROOT" \
  --judge-suite both \
  --judge-rounds 3 \
  --fast-mode \
  --sample-size "$SAMPLE_SIZE" \
  --sample-seed "$SAMPLE_SEED" \
  --parallel-datasets \
  --max-dataset-jobs 3 \
  --total-worker-budget 8 \
  --throttle-ms 200 \
  --resume \
  --madam-max-parallel-agents 4 \
  --madam-compact-doc-chars 1200 \
  --madam-argument-summary-chars 240 \
  --madam-early-stop-patience 1 \
  --judge-only-on-contention \
  --validate-eajp-schema \
  --eajp-schema-strict-enums \
  --compute-proposal-metrics \
  --agent-provider openai \
  --agent-model gpt-5.4-nano \
  --standard-rag-provider openai \
  --standard-rag-model gpt-5.4-nano \
  --madam-agent-provider openai \
  --madam-agent-model gpt-5.4-nano \
  --eajp-agent-provider openai \
  --eajp-agent-model gpt-5.4-nano \
  --openai-judge-model gpt-5.4-mini \
  --anthropic-judge-model claude-haiku-4-5-20251001 \
  "$@"
