#!/usr/bin/env bash
# run_counterfactual_pipeline.sh — Counterfactual tone-swap causal analysis pipeline.
#
# Addresses reviewer Critique 5: no causal mechanism validation.
#
# Pipeline:
#   Step 1: Collect incorrect MADAM-RAG trials from existing results
#   Step 2: Build counterfactual swaps (swap neutral↔assertive for each trial)
#   Step 3: Re-run only the swapped instances through MADAM-RAG
#   Step 4: Compute flip rate (incorrect → correct after swap)
#
# Expected result (from rebuttal): ~54% flip rate (OpenAI judge), ~49.5% (Anthropic)
# Rates significantly above 50% chance confirm tone is a causal contributor.
#
# Usage:
#   ./experiments/run_counterfactual_pipeline.sh \
#       --results-dir results/full_run_seed42 \
#       [--judge-suite openai] [--dataset confact] \
#       [--out-root results/counterfactual] [--dry-run]

set -euo pipefail

RESULTS_DIR="results/full_run_seed42"
JUDGE_SUITE="both"
DATASET="confact"
OUT_ROOT="results/counterfactual"
DRY_RUN=0
VENV_DIR=".venv"
VERBOSE=""
AGENT_PROVIDER="openai"
SEED=42

usage() {
  grep '^#' "$0" | sed 's/^# \{0,2\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)    RESULTS_DIR="$2";    shift 2 ;;
    --judge-suite)    JUDGE_SUITE="$2";    shift 2 ;;
    --dataset)        DATASET="$2";        shift 2 ;;
    --out-root)       OUT_ROOT="$2";       shift 2 ;;
    --agent-provider) AGENT_PROVIDER="$2"; shift 2 ;;
    --seed)           SEED="$2";           shift 2 ;;
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

SWAPS_JSON="$OUT_ROOT/counterfactual_swaps.json"
SWAP_RESULTS_DIR="$OUT_ROOT/swap_results"
ANALYSIS_JSON="results/stats/counterfactual_flip_rates.json"

echo "============================================================"
echo "COUNTERFACTUAL CAUSAL ANALYSIS PIPELINE"
echo "Results dir: $RESULTS_DIR"
echo "Judge suite: $JUDGE_SUITE"
echo "Dataset:     $DATASET"
echo "Out root:    $OUT_ROOT"
echo "============================================================"

mkdir -p "$OUT_ROOT"
mkdir -p results/stats

# ---------------------------------------------------------------------------
# Step 1: Build counterfactual swaps
# ---------------------------------------------------------------------------
echo ""
echo "Step 1: Building counterfactual swaps from incorrect MADAM-RAG trials..."

if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [DRY RUN] would run: build_counterfactual_swaps.py"
else
  $PYTHON experiments/build_counterfactual_swaps.py \
    --results-dir "$RESULTS_DIR" \
    --dataset "$DATASET" \
    --system madam_rag \
    --out "$SWAPS_JSON" \
    ${VERBOSE:+--verbose}

  N_SWAPS=$(python3 -c "import json; d=json.load(open('$SWAPS_JSON')); print(len(d.get('swaps', d) if isinstance(d, dict) else d))" 2>/dev/null || echo "?")
  echo "  Built $N_SWAPS counterfactual swaps → $SWAPS_JSON"
fi

# ---------------------------------------------------------------------------
# Step 2: Re-run MADAM-RAG on swapped instances
# ---------------------------------------------------------------------------
echo ""
echo "Step 2: Re-running MADAM-RAG on swapped instances..."

if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [DRY RUN] would run MADAM-RAG on swapped TCB"
else
  $PYTHON experiments/run_experiment.py \
    --dataset "$DATASET" \
    --judge-suite "$JUDGE_SUITE" \
    --agent-provider "$AGENT_PROVIDER" \
    --seed "$SEED" \
    --output-dir "$SWAP_RESULTS_DIR" \
    --tcb-override "$SWAPS_JSON" \
    --run-madam-rag \
    --resume \
    ${VERBOSE:+--verbose} || {
      echo "WARNING: MADAM-RAG re-run failed. Trying analyze with existing swap results..."
    }
fi

# ---------------------------------------------------------------------------
# Step 3: Analyze flip rates
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Analyzing flip rates..."

if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [DRY RUN] would run: analyze_counterfactual_swaps.py"
else
  $PYTHON experiments/analyze_counterfactual_swaps.py \
    --original-results "$RESULTS_DIR" \
    --swap-results "$SWAP_RESULTS_DIR" \
    --swaps "$SWAPS_JSON" \
    --dataset "$DATASET" \
    --out "$ANALYSIS_JSON" \
    ${VERBOSE:+--verbose}

  echo ""
  echo "============================================================"
  echo "COUNTERFACTUAL PIPELINE COMPLETE"
  echo "  Swaps JSON     : $SWAPS_JSON"
  echo "  Swap results   : $SWAP_RESULTS_DIR"
  echo "  Flip rates JSON: $ANALYSIS_JSON"
  echo "============================================================"

  # Print summary
  $PYTHON - <<'PYEOF'
import json, sys
try:
    d = json.load(open("results/stats/counterfactual_flip_rates.json"))
    print("\nFLIP RATE SUMMARY")
    print("-" * 50)
    for suite, stats in d.get("by_suite", {}).items():
        flipped = stats.get("flipped", 0)
        total = stats.get("total", 0)
        rate = stats.get("flip_rate", flipped/total if total else 0)
        p = stats.get("p_value", float("nan"))
        print(f"  {suite:<30} {flipped}/{total} = {rate:.1%}  p={p:.4f}")
except Exception as e:
    print(f"Could not print summary: {e}", file=sys.stderr)
PYEOF
fi
