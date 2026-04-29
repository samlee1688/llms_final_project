#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_tcb_confact.sh
#
# Wrapper for experiments/build_tcb_confact.py.
# Builds a Tone-Controlled Benchmark dataset from CONFACT HumC using an LLM
# rewriter (default: Anthropic claude-haiku-4-5-20251001) and validates
# meaning preservation with BERTScore F1 (rescaled roberta-large) and sentence
# cosine similarity (all-mpnet-base-v2).
#
# Document selection (CONFACT-specific):
#   yes-doc: supports_answer="yes", highest credibility, shortest content
#   no-doc:  supports_answer="no",  lowest credibility ≤ yes-doc, shortest content
#
# Usage:
#   ./experiments/build_tcb_confact.sh [options]
#
# Options:
#   --source <path>            Input CONFACT HumC JSON
#                              (default: data/full/confact/humc_processed_cleaned.json)
#   --dest-dir <path>          Output directory for TCB JSON, quality reports,
#                              and plots (default: data/full/confact)
#   --rewriter-provider <p>    Rewrite provider: anthropic, gemini, local
#                              (default: anthropic)
#   --rewriter-model <model>   Model name for the rewrite provider
#                              (default: claude-haiku-4-5-20251001)
#   --bs-target <float>        Minimum rescaled BERTScore F1 per pair (default: 0.30,
#                              lower than RAMDocs due to long fact-check documents)
#   --cosine-target <float>    Minimum cosine similarity per pair (default: 0.90)
#   --rewrite-workers <n>      Parallel API workers for initial TCB build (default: 6)
#   --checkpoint-every <n>     Checkpoint every N instances during initial build (default: 50)
#   --batch-size <n>           Scoring batch size (default: auto — 128 for GPU/MPS, 64 for CPU)
#   --device <device>          Torch device: mps, cuda, cpu (default: auto-detect mps>cuda>cpu)
#   --n-low-end <n>            Low-scoring pairs printed for inspection (default: 15)
#   --python <path>            Python executable (default: auto-detect)
#   --skip-initial-build       Skip TCB build (TCB JSON must already exist in --dest-dir)
#   --force-rebuild            Delete and re-build TCB JSON even if it exists
#   --smoke-test               Run end-to-end on --smoke-n instances only
#   --smoke-n <n>              Number of instances for --smoke-test (default: 5)
#   -v, --verbose              Verbose logging
#   -h, --help                 Show this help message
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SOURCE="data/full/confact/humc_processed_cleaned.json"
DEST_DIR="data/full/confact"
REWRITER_PROVIDER="anthropic"
REWRITER_MODEL="claude-haiku-4-5-20251001"
BS_TARGET="0.30"
COSINE_TARGET="0.90"
REWRITE_WORKERS="6"
CHECKPOINT_EVERY="50"
BATCH_SIZE=""
DEVICE=""
N_LOW_END="15"
PYTHON_EXE=""
SKIP_INITIAL_BUILD=0
FORCE_REBUILD=0
SMOKE_TEST=0
SMOKE_N="5"
VERBOSE=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
print_help() {
  sed -n '3,/^set -euo/p' "$0" | grep "^#" | sed 's/^# \{0,2\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)             SOURCE="${2:?'--source requires a value'}"; shift 2 ;;
    --dest-dir)           DEST_DIR="${2:?'--dest-dir requires a value'}"; shift 2 ;;
    --rewriter-provider)  REWRITER_PROVIDER="${2:?'--rewriter-provider requires a value'}"; shift 2 ;;
    --rewriter-model)     REWRITER_MODEL="${2:?'--rewriter-model requires a value'}"; shift 2 ;;
    --bs-target)          BS_TARGET="${2:?'--bs-target requires a value'}"; shift 2 ;;
    --cosine-target)      COSINE_TARGET="${2:?'--cosine-target requires a value'}"; shift 2 ;;
    --rewrite-workers)    REWRITE_WORKERS="${2:?'--rewrite-workers requires a value'}"; shift 2 ;;
    --checkpoint-every)   CHECKPOINT_EVERY="${2:?'--checkpoint-every requires a value'}"; shift 2 ;;
    --batch-size)         BATCH_SIZE="${2:?'--batch-size requires a value'}"; shift 2 ;;
    --device)             DEVICE="${2:?'--device requires a value'}"; shift 2 ;;
    --n-low-end)          N_LOW_END="${2:?'--n-low-end requires a value'}"; shift 2 ;;
    --python)             PYTHON_EXE="${2:?'--python requires a value'}"; shift 2 ;;
    --skip-initial-build) SKIP_INITIAL_BUILD=1; shift ;;
    --force-rebuild)      FORCE_REBUILD=1; shift ;;
    --smoke-test)         SMOKE_TEST=1; shift ;;
    --smoke-n)            SMOKE_N="${2:?'--smoke-n requires a value'}"; shift 2 ;;
    -v|--verbose)         VERBOSE=1; shift ;;
    -h|--help)            print_help; exit 0 ;;
    *) echo "ERROR: Unknown option: $1" >&2; print_help; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve paths and Python
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [[ -z "$PYTHON_EXE" ]]; then
  for candidate in .venv/bin/python python3 python; do
    if command -v "$candidate" &>/dev/null 2>&1; then
      PYTHON_EXE="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_EXE" ]]; then
  echo "ERROR: Could not find a Python executable. Use --python." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detect device and tune batch size
# ---------------------------------------------------------------------------
_detect_device() {
  "$PYTHON_EXE" - <<'PYEOF'
import sys
try:
    import torch
    if torch.backends.mps.is_available():
        print("mps"); sys.exit(0)
    if torch.cuda.is_available():
        print("cuda"); sys.exit(0)
except ImportError:
    pass
print("cpu")
PYEOF
}

if [[ -z "$DEVICE" ]]; then
  DEVICE="$(_detect_device)"
  echo "[info] Auto-detected device: $DEVICE"
fi

if [[ -z "$BATCH_SIZE" ]]; then
  if [[ "$DEVICE" == "mps" || "$DEVICE" == cuda* ]]; then
    BATCH_SIZE="128"
  else
    BATCH_SIZE="64"
  fi
  echo "[info] Auto-set batch-size=$BATCH_SIZE for device=$DEVICE"
fi

_ts()   { date '+%H:%M:%S'; }
_step() { echo; echo "=== [$(_ts)] $* ==="; }

# ---------------------------------------------------------------------------
# Validate numeric arguments
# ---------------------------------------------------------------------------
_is_float() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
_is_int()   { [[ "$1" =~ ^[0-9]+$ ]]; }

_check_float() {
  local flag="$1" val="$2"
  if ! _is_float "$val"; then
    echo "ERROR: $flag must be a number between 0 and 1, got: $val" >&2; exit 1
  fi
  if awk "BEGIN{exit !($val > 1.0)}"; then
    echo "ERROR: $flag must be ≤ 1.0, got: $val" >&2; exit 1
  fi
}

_check_posint() {
  local flag="$1" val="$2"
  if ! _is_int "$val" || (( val < 1 )); then
    echo "ERROR: $flag must be a positive integer, got: $val" >&2; exit 1
  fi
}

_check_float   "--bs-target"        "$BS_TARGET"
_check_float   "--cosine-target"    "$COSINE_TARGET"
_check_posint  "--rewrite-workers"  "$REWRITE_WORKERS"
_check_posint  "--checkpoint-every" "$CHECKPOINT_EVERY"
_check_posint  "--batch-size"       "${BATCH_SIZE:-1}"
_check_posint  "--n-low-end"        "$N_LOW_END"
_check_posint  "--smoke-n"          "$SMOKE_N"

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
_step "Configuration"
echo "  Python          : $PYTHON_EXE ($("$PYTHON_EXE" --version 2>&1))"
echo "  Source          : $SOURCE"
echo "  Dest dir        : $DEST_DIR"
echo "  Rewriter        : $REWRITER_PROVIDER / $REWRITER_MODEL"
echo "  BERTScore target: $BS_TARGET  (drop threshold)"
echo "  Cosine target   : $COSINE_TARGET"
echo "  Rewrite workers : $REWRITE_WORKERS  (initial build only)"
echo "  Checkpoint every: $CHECKPOINT_EVERY"
echo "  Batch size      : $BATCH_SIZE"
echo "  Device          : ${DEVICE:-auto}"
echo "  N low-end shown : $N_LOW_END"
[[ $SMOKE_TEST -eq 1 ]]         && echo "  Smoke test      : YES (n=$SMOKE_N)"
[[ $SKIP_INITIAL_BUILD -eq 1 ]] && echo "  Skip init build : YES"
[[ $FORCE_REBUILD -eq 1 ]]      && echo "  Force rebuild   : YES"

if [[ ! -f "$SOURCE" ]]; then
  echo "ERROR: Source file not found: $SOURCE" >&2
  echo "       Pass --source <path> pointing to the CONFACT HumC JSON." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build argument array
# ---------------------------------------------------------------------------
PYARGS=(
  experiments/build_tcb_confact.py
  --source            "$SOURCE"
  --dest-dir          "$DEST_DIR"
  --rewriter-provider "$REWRITER_PROVIDER"
  --rewriter-model    "$REWRITER_MODEL"
  --bs-target         "$BS_TARGET"
  --cosine-target     "$COSINE_TARGET"
  --rewrite-workers   "$REWRITE_WORKERS"
  --checkpoint-every  "$CHECKPOINT_EVERY"
  --batch-size        "$BATCH_SIZE"
  --n-low-end         "$N_LOW_END"
)

[[ -n "$DEVICE" ]]              && PYARGS+=(--device "$DEVICE")
[[ $SKIP_INITIAL_BUILD -eq 1 ]] && PYARGS+=(--skip-initial-build)
[[ $FORCE_REBUILD -eq 1 ]]      && PYARGS+=(--force-rebuild)
[[ $SMOKE_TEST -eq 1 ]]         && PYARGS+=(--smoke-test --smoke-n "$SMOKE_N")
[[ $VERBOSE -eq 1 ]]            && PYARGS+=(--verbose)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
_step "Running build_tcb_confact.py"
echo "  Command: $PYTHON_EXE ${PYARGS[*]}"
echo

START_TS=$(date +%s)
"$PYTHON_EXE" "${PYARGS[@]}"
END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_step "Done  (elapsed: ${ELAPSED}s)"

IN_STEM="$(basename "${SOURCE%.json}")"
TCB_OUT="$DEST_DIR/tcb_${IN_STEM}.json"
REPORT_OUT="$DEST_DIR/tcb_${IN_STEM}.report.json"
[[ $SMOKE_TEST -eq 1 ]] && TCB_OUT="$DEST_DIR/smoke/tcb_smoke.json"
[[ $SMOKE_TEST -eq 1 ]] && REPORT_OUT="$DEST_DIR/smoke/tcb_smoke.report.json"

if [[ -f "$TCB_OUT" ]]; then
  COUNT=$("$PYTHON_EXE" -c "import json; print(len(json.load(open('$TCB_OUT'))))" 2>/dev/null || echo "?")
  echo "  TCB dataset : $TCB_OUT  ($COUNT instances)"
fi
if [[ -f "$REPORT_OUT" ]]; then
  echo "  Report      : $REPORT_OUT"
  _REPORT_PATH="$REPORT_OUT" "$PYTHON_EXE" - <<'PYEOF'
import json, sys, os
p = os.environ.get("_REPORT_PATH")
if not p:
    sys.exit(0)
try:
    r = json.load(open(p))
    print(f"  BS  F1      : mean={r['final_mean_bs']:.4f}  min={r['final_min_bs']:.4f}  pass={r['final_pass_rate_bs']:.1%}")
    print(f"  Cosine sim  : mean={r['final_mean_cos']:.4f}  min={r['final_min_cos']:.4f}  pass={r['final_pass_rate_cos']:.1%}")
    print(f"  Dropped     : {r.get('dropped_instances', '?')}")
except Exception as e:
    print(f"  (could not parse report: {e})")
PYEOF
fi
echo
