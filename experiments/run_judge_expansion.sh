#!/usr/bin/env bash
# run_judge_expansion.sh — Run EAJP and MADAM-RAG under expanded judge models.
#
# Addresses reviewer Critique 4: judge-model dependency.
# Extends beyond {OpenAI Haiku, Anthropic Haiku} to include:
#   - gpt-4o-mini  (OpenAI, publicly available, cheap)
#   - gpt-4o       (OpenAI, most capable)
#   - claude-haiku-4-5-20251001  (Anthropic, already in baseline)
#   - claude-sonnet-4-6          (Anthropic, stronger)
#   - gemini-2.0-flash           (Google, for cross-provider diversity)
#
# Usage:
#   ./experiments/run_judge_expansion.sh [--seed 42] [--dataset confact] \
#       [--judges "gpt-4o-mini gpt-4o claude-haiku-4-5-20251001 claude-sonnet-4-6"] \
#       [--systems "eajp madam_rag"] [--out-root results/judge_expansion] \
#       [--sample-size 200] [--dry-run]

set -euo pipefail

SEED=42
DATASET="confact"
JUDGES="gpt-4o-mini gpt-4o claude-haiku-4-5-20251001 claude-sonnet-4-6"
SYSTEMS="eajp madam_rag"
OUT_ROOT="results/judge_expansion"
SAMPLE_SIZE=200       # smaller sample for cost control
DRY_RUN=0
VENV_DIR=".venv"
VERBOSE=""

usage() {
  grep '^#' "$0" | sed 's/^# \{0,2\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)           SEED="$2";           shift 2 ;;
    --dataset)        DATASET="$2";        shift 2 ;;
    --judges)         JUDGES="$2";         shift 2 ;;
    --systems)        SYSTEMS="$2";        shift 2 ;;
    --out-root)       OUT_ROOT="$2";       shift 2 ;;
    --sample-size)    SAMPLE_SIZE="$2";    shift 2 ;;
    --dry-run)        DRY_RUN=1;           shift ;;
    --venv)           VENV_DIR="$2";       shift 2 ;;
    --verbose|-v)     VERBOSE="--verbose"; shift ;;
    --help|-h)        usage ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "JUDGE EXPANSION RUNNER"
echo "Seed:        $SEED"
echo "Dataset:     $DATASET"
echo "Judges:      $JUDGES"
echo "Systems:     $SYSTEMS"
echo "Sample size: $SAMPLE_SIZE"
echo "Out root:    $OUT_ROOT"
echo "============================================================"

SAMPLE_FLAG=""
if [[ $SAMPLE_SIZE -gt 0 ]]; then
  SAMPLE_FLAG="--sample-size $SAMPLE_SIZE --sample-seed $SEED"
fi

_judge_provider() {
  # Return openai|anthropic|google based on judge model name
  case "$1" in
    gpt-*)    echo "openai" ;;
    claude-*) echo "anthropic" ;;
    gemini-*) echo "google" ;;
    *)        echo "openai" ;;
  esac
}

TOTAL=0
FAILED=0

for JUDGE_MODEL in $JUDGES; do
  JUDGE_PROVIDER=$(_judge_provider "$JUDGE_MODEL")

  for SYSTEM in $SYSTEMS; do
    JUDGE_SHORT="${JUDGE_MODEL//claude-haiku-4-5-20251001/haiku45}"
    JUDGE_SHORT="${JUDGE_SHORT//claude-sonnet-4-6/sonnet46}"
    JUDGE_SHORT="${JUDGE_SHORT//gpt-4o-mini/gpt4omini}"
    JUDGE_SHORT="${JUDGE_SHORT//gpt-4o/gpt4o}"
    JUDGE_SHORT="${JUDGE_SHORT//gemini-2.0-flash/gemini20}"
    CELL_OUT="$OUT_ROOT/${DATASET}/${JUDGE_SHORT}/${SYSTEM}"

    echo ""
    echo ">>> $SYSTEM  judge=$JUDGE_MODEL ($JUDGE_PROVIDER)  →  $CELL_OUT"

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    [DRY RUN]"
      TOTAL=$((TOTAL + 1))
      continue
    fi

    case "$SYSTEM" in
      standard_rag)  SYS_FLAG="--run-standard-rag" ;;
      madam_rag)     SYS_FLAG="--run-madam-rag" ;;
      eajp)          SYS_FLAG="--run-eajp" ;;
      *) echo "Unknown system: $SYSTEM"; continue ;;
    esac

    $PYTHON experiments/run_experiment.py \
      --dataset "$DATASET" \
      --judge-provider "$JUDGE_PROVIDER" \
      --judge-model "$JUDGE_MODEL" \
      --seed "$SEED" \
      --output-dir "$CELL_OUT" \
      $SYS_FLAG \
      $SAMPLE_FLAG \
      --resume \
      ${VERBOSE:+--verbose} && TOTAL=$((TOTAL + 1)) || {
        echo "WARNING: $SYSTEM judge=$JUDGE_MODEL failed; continuing."
        FAILED=$((FAILED + 1))
      }
  done
done

echo ""
echo "============================================================"
echo "JUDGE EXPANSION COMPLETE"
echo "  Ran:    $TOTAL cells"
echo "  Failed: $FAILED cells"
echo "============================================================"

if [[ $DRY_RUN -eq 0 ]] && [[ -d "$OUT_ROOT" ]]; then
  mkdir -p results/stats
  $PYTHON experiments/analyze_judge_divergence.py \
    --results-dir "$OUT_ROOT" \
    --out "results/stats/judge_expansion_divergence.json" \
    ${VERBOSE:+--verbose} 2>&1 | tee results/stats/judge_expansion_summary.txt || \
    echo "WARNING: judge divergence analysis failed (check analyze_judge_divergence.py output)"
fi
