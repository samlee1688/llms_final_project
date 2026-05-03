#!/usr/bin/env bash
# run_multiseed.sh — Run experiments under multiple random seeds for statistical validity.
#
# Addresses reviewer Critique 1: single-seed point estimates; no CIs or significance tests.
# Runs the full pipeline under seeds 42, 7, 123, 256, 999 then computes bootstrap CIs
# and permutation tests across all seeds.
#
# Usage:
#   ./experiments/run_multiseed.sh [--seeds "42 7 123 256 999"] \
#       [--dataset confact|ramdocs|tcb_confact|tcb_ramdocs|<path>] \
#       [--data-root data/full] \
#       [--judge-suite openai] [--agent-provider openai] \
#       [--systems "eajp madam_rag"] [--out-root results/multiseed] \
#       [--skip-runs] [--n-bootstrap 2000] [--n-permutations 10000]
#
# Dataset shortcuts (resolved against --data-root):
#   confact      → <data-root>/confact/humc_processed_cleaned.json  (benchmark=confact)
#   ramdocs      → <data-root>/ramdocs_test.json                           (benchmark=ramdocs)
#   tcb_confact  → <data-root>/confact/tcb_humc_processed_cleaned.json (benchmark=tcb)
#   tcb_ramdocs  → <data-root>/tcb_ramdocs_test.json                       (benchmark=tcb)
#   <path>       → used as-is                                               (benchmark=custom)
#
# Outputs:
#   results/multiseed/seed_<N>/   — per-seed run results
#   results/stats/ci_multiseed.json — bootstrap CIs + permutation tests
#   results/stats/multiseed_summary.txt — human-readable table

set -euo pipefail

SEEDS="42 7 123 256 999"
DATASET="confact"      # short name or path; resolved below
DATA_ROOT="data/full"
BENCHMARK=""           # auto-set from DATASET short name; override with --benchmark
JUDGE_SUITE="openai"
AGENT_PROVIDER="openai"
SYSTEMS="eajp madam_rag"
OUT_ROOT="results/multiseed"
SKIP_RUNS=0
N_BOOTSTRAP=2000
N_PERMUTATIONS=10000
CI_LEVEL=0.95
BONFERRONI_CELLS=9
SAMPLE_SIZE=0          # 0 = full corpus
VENV_DIR=".venv"
VERBOSE=""

usage() {
  grep '^#' "$0" | sed 's/^# \{0,2\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)        SEEDS="$2";         shift 2 ;;
    --dataset)      DATASET="$2";       shift 2 ;;
    --data-root)    DATA_ROOT="$2";     shift 2 ;;
    --benchmark)    BENCHMARK="$2";     shift 2 ;;
    --judge-suite)  JUDGE_SUITE="$2";   shift 2 ;;
    --agent-provider) AGENT_PROVIDER="$2"; shift 2 ;;
    --systems)      SYSTEMS="$2";       shift 2 ;;
    --out-root)     OUT_ROOT="$2";      shift 2 ;;
    --skip-runs)    SKIP_RUNS=1;        shift ;;
    --n-bootstrap)  N_BOOTSTRAP="$2";   shift 2 ;;
    --n-permutations) N_PERMUTATIONS="$2"; shift 2 ;;
    --ci-level)     CI_LEVEL="$2";      shift 2 ;;
    --sample-size)  SAMPLE_SIZE="$2";   shift 2 ;;
    --venv)         VENV_DIR="$2";      shift 2 ;;
    --verbose|-v)   VERBOSE="--verbose"; shift ;;
    --help|-h)      usage ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Activate venv if present
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Resolve dataset short name → file path + benchmark type
case "$DATASET" in
  confact)
    DATASET_PATH="$DATA_ROOT/confact/humc_processed_cleaned.json"
    [[ -z "$BENCHMARK" ]] && BENCHMARK="confact"
    ;;
  ramdocs)
    DATASET_PATH="$DATA_ROOT/ramdocs_test.json"
    [[ -z "$BENCHMARK" ]] && BENCHMARK="ramdocs"
    ;;
  tcb_confact)
    DATASET_PATH="$DATA_ROOT/confact/tcb_humc_processed_cleaned.json"
    [[ -z "$BENCHMARK" ]] && BENCHMARK="tcb"
    ;;
  tcb_ramdocs)
    DATASET_PATH="$DATA_ROOT/tcb_ramdocs_test.json"
    [[ -z "$BENCHMARK" ]] && BENCHMARK="tcb"
    ;;
  *)
    DATASET_PATH="$DATASET"
    [[ -z "$BENCHMARK" ]] && BENCHMARK="custom"
    ;;
esac

echo "============================================================"
echo "MULTI-SEED EXPERIMENT RUNNER"
echo "Seeds:       $SEEDS"
echo "Dataset:     $DATASET  →  $DATASET_PATH"
echo "Benchmark:   $BENCHMARK"
echo "Judge suite: $JUDGE_SUITE"
echo "Systems:     $SYSTEMS"
echo "Out root:    $OUT_ROOT"
echo "============================================================"

SEED_DIRS=()

if [[ $SKIP_RUNS -eq 0 ]]; then
  for SEED in $SEEDS; do
    SEED_OUT="$OUT_ROOT/seed_${SEED}"
    echo ""
    echo ">>> Running seed=$SEED  →  $SEED_OUT"

    SAMPLE_FLAG=""
    if [[ $SAMPLE_SIZE -gt 0 ]]; then
      SAMPLE_FLAG="--sample-size $SAMPLE_SIZE --sample-seed $SEED"
    fi

    # Build system flags from SYSTEMS variable
    SYSTEM_FLAGS=""
    for SYS in $SYSTEMS; do
      case "$SYS" in
        standard_rag) SYSTEM_FLAGS="$SYSTEM_FLAGS --run-standard-rag" ;;
        madam_rag)    SYSTEM_FLAGS="$SYSTEM_FLAGS --run-madam-rag" ;;
        eajp)         SYSTEM_FLAGS="$SYSTEM_FLAGS --run-eajp" ;;
        eajp_scorecard_only) SYSTEM_FLAGS="$SYSTEM_FLAGS --run-eajp-scorecard-only" ;;
        eajp_no_privilege)   SYSTEM_FLAGS="$SYSTEM_FLAGS --run-eajp-no-privilege" ;;
        cot_judge)    SYSTEM_FLAGS="$SYSTEM_FLAGS --run-cot-judge" ;;
        ra_rag)       SYSTEM_FLAGS="$SYSTEM_FLAGS --run-ra-rag" ;;
        credibility_hint)    SYSTEM_FLAGS="$SYSTEM_FLAGS --run-credibility-hint" ;;
        credibility_vote)    SYSTEM_FLAGS="$SYSTEM_FLAGS --run-credibility-vote" ;;
      esac
    done

    $PYTHON experiments/run_experiment.py \
      --dataset "$DATASET_PATH" \
      --benchmark "$BENCHMARK" \
      --judge-suite "$JUDGE_SUITE" \
      --agent-provider "$AGENT_PROVIDER" \
      --seed "$SEED" \
      --output-dir "$SEED_OUT" \
      $SYSTEM_FLAGS \
      $SAMPLE_FLAG \
      ${VERBOSE:+--verbose} || {
        echo "WARNING: seed=$SEED failed; continuing with remaining seeds."
      }

    SEED_DIRS+=("$SEED_OUT")
  done
else
  echo "Skipping experiment runs (--skip-runs)."
  for SEED in $SEEDS; do
    SEED_DIRS+=("$OUT_ROOT/seed_${SEED}")
  done
fi

# Collect existing seed directories
EXISTING_DIRS=()
for D in "${SEED_DIRS[@]}"; do
  if [[ -d "$D" ]]; then
    EXISTING_DIRS+=("$D")
  else
    echo "WARNING: $D does not exist; skipping in stats."
  fi
done

if [[ ${#EXISTING_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No seed result directories found."
  exit 1
fi

echo ""
echo ">>> Computing bootstrap CIs and permutation tests over ${#EXISTING_DIRS[@]} seeds..."

mkdir -p results/stats

STATS_OUT="results/stats/ci_multiseed.json"
$PYTHON experiments/compute_statistical_tests.py \
  --results-dir "${EXISTING_DIRS[@]}" \
  --out "$STATS_OUT" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --n-permutations "$N_PERMUTATIONS" \
  --ci-level "$CI_LEVEL" \
  --bonferroni-cells "$BONFERRONI_CELLS" \
  --system-a eajp \
  --system-b madam_rag \
  ${VERBOSE:+--verbose} 2>&1 | tee results/stats/multiseed_summary.txt

echo ""
echo "============================================================"
echo "DONE"
echo "  Stats JSON : $STATS_OUT"
echo "  Summary    : results/stats/multiseed_summary.txt"
echo "============================================================"
