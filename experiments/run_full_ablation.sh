#!/usr/bin/env bash
# run_full_ablation.sh — Run all 9 systems in the ablation ladder on all datasets.
#
# Addresses reviewer Critique 2: only 3/9 ablation systems reported.
#
# Ablation ladder (ordered by complexity):
#   1. standard_rag
#   2. credibility_vote
#   3. madam_rag            ← MADAM-RAG baseline
#   4. cot_judge
#   5. ra_rag
#   6. credibility_hint
#   7. eajp_scorecard_only
#   8. eajp_no_privilege
#   9. eajp                 ← full EAJP
#
# Datasets: confact, ramdocs (configurable via --datasets)
#
# Usage:
#   ./experiments/run_full_ablation.sh [--seed 42] [--datasets "confact ramdocs"] \
#       [--judge-suite both] [--judge-rounds 5] [--agent-provider openai] \
#       [--out-root results/ablation_seed42] [--sample-size 0] \
#       [--skip-systems "credibility_vote cot_judge"] [--dry-run]

set -euo pipefail

SEED=42
DATASETS="confact ramdocs"
JUDGE_SUITE="both"
JUDGE_ROUNDS=5
AGENT_PROVIDER="openai"
OUT_ROOT="results/ablation_seed${SEED}"
SAMPLE_SIZE=0
SKIP_SYSTEMS=""
DRY_RUN=0
VENV_DIR=".venv"
VERBOSE=""
RESUME=1

ALL_SYSTEMS="standard_rag credibility_vote madam_rag cot_judge ra_rag credibility_hint eajp_scorecard_only eajp_no_privilege eajp"

usage() {
  grep '^#' "$0" | sed 's/^# \{0,2\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)           SEED="$2";           shift 2 ;;
    --datasets)       DATASETS="$2";       shift 2 ;;
    --judge-suite)    JUDGE_SUITE="$2";    shift 2 ;;
    --judge-rounds)   JUDGE_ROUNDS="$2";   shift 2 ;;
    --agent-provider) AGENT_PROVIDER="$2"; shift 2 ;;
    --out-root)       OUT_ROOT="$2";       shift 2 ;;
    --sample-size)    SAMPLE_SIZE="$2";    shift 2 ;;
    --skip-systems)   SKIP_SYSTEMS="$2";  shift 2 ;;
    --no-resume)      RESUME=0;            shift ;;
    --dry-run)        DRY_RUN=1;           shift ;;
    --venv)           VENV_DIR="$2";       shift 2 ;;
    --verbose|-v)     VERBOSE="--verbose"; shift ;;
    --help|-h)        usage ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

OUT_ROOT="results/ablation_seed${SEED}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "FULL ABLATION LADDER RUNNER"
echo "Seed:        $SEED"
echo "Datasets:    $DATASETS"
echo "Systems:     $ALL_SYSTEMS"
echo "Skip:        ${SKIP_SYSTEMS:-none}"
echo "Judge suite: $JUDGE_SUITE"
echo "Judge rounds: $JUDGE_ROUNDS"
echo "Out root:    $OUT_ROOT"
echo "Dry run:     $DRY_RUN"
echo "============================================================"

# Build the set of systems to actually run
declare -A SKIP_SET
for SYS in $SKIP_SYSTEMS; do
  SKIP_SET[$SYS]=1
done

SAMPLE_FLAG=""
if [[ $SAMPLE_SIZE -gt 0 ]]; then
  SAMPLE_FLAG="--sample-size $SAMPLE_SIZE --sample-seed $SEED"
fi

RESUME_FLAG=""
if [[ $RESUME -eq 1 ]]; then
  RESUME_FLAG="--resume"
fi

TOTAL=0
SKIPPED=0
FAILED=0

for DATASET in $DATASETS; do
  for SYSTEM in $ALL_SYSTEMS; do
    if [[ -n "${SKIP_SET[$SYSTEM]+x}" ]]; then
      echo "  SKIP $SYSTEM on $DATASET (--skip-systems)"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    SYSTEM_OUT="$OUT_ROOT/${DATASET}/${SYSTEM}"
    echo ""
    echo ">>> $SYSTEM on $DATASET  →  $SYSTEM_OUT"

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    [DRY RUN] would run: $SYSTEM on $DATASET"
      TOTAL=$((TOTAL + 1))
      continue
    fi

    # Map system name to flag(s)
    case "$SYSTEM" in
      standard_rag)       SYS_FLAG="--run-standard-rag" ;;
      credibility_vote)   SYS_FLAG="--run-credibility-vote" ;;
      madam_rag)          SYS_FLAG="--run-madam-rag" ;;
      cot_judge)          SYS_FLAG="--run-cot-judge" ;;
      ra_rag)             SYS_FLAG="--run-ra-rag" ;;
      credibility_hint)   SYS_FLAG="--run-credibility-hint" ;;
      eajp_scorecard_only) SYS_FLAG="--run-eajp-scorecard-only" ;;
      eajp_no_privilege)  SYS_FLAG="--run-eajp-no-privilege" ;;
      eajp)               SYS_FLAG="--run-eajp" ;;
      *) echo "Unknown system: $SYSTEM"; exit 1 ;;
    esac

    $PYTHON experiments/run_experiment.py \
      --dataset "$DATASET" \
      --judge-suite "$JUDGE_SUITE" \
      --judge-rounds "$JUDGE_ROUNDS" \
      --agent-provider "$AGENT_PROVIDER" \
      --seed "$SEED" \
      --output-dir "$SYSTEM_OUT" \
      $SYS_FLAG \
      $SAMPLE_FLAG \
      ${RESUME_FLAG} \
      ${VERBOSE:+--verbose} && TOTAL=$((TOTAL + 1)) || {
        echo "WARNING: $SYSTEM on $DATASET failed; continuing."
        FAILED=$((FAILED + 1))
      }
  done
done

echo ""
echo "============================================================"
echo "ABLATION COMPLETE"
echo "  Ran:     $TOTAL cells"
echo "  Skipped: $SKIPPED cells"
echo "  Failed:  $FAILED cells"
echo ""
echo "Computing per-cell metrics..."
echo "============================================================"

# Compute bootstrap CIs across all runs in the ablation
if [[ $DRY_RUN -eq 0 ]] && [[ -d "$OUT_ROOT" ]]; then
  mkdir -p results/stats

  ALL_SYSTEM_NAMES="standard_rag credibility_vote madam_rag cot_judge ra_rag credibility_hint eajp_scorecard_only eajp_no_privilege eajp"
  $PYTHON experiments/compute_statistical_tests.py \
    --results-dir "$OUT_ROOT" \
    --out "results/stats/ablation_seed${SEED}_ci.json" \
    --n-bootstrap 2000 \
    --no-permutation-test \
    --system-a eajp \
    --system-b madam_rag \
    ${VERBOSE:+--verbose} 2>&1 | tee "results/stats/ablation_seed${SEED}_summary.txt"

  echo ""
  echo "  CI JSON    : results/stats/ablation_seed${SEED}_ci.json"
  echo "  Summary    : results/stats/ablation_seed${SEED}_summary.txt"
fi

echo "============================================================"
