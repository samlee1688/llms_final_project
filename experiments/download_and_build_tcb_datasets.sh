#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_and_build_tcb_datasets.sh
#
# Downloads configured dataset splits from Hugging Face, converts them to the
# project JSON schema, and builds Tone-Controlled Benchmark (TCB) datasets.
#
# Output layout under DATASET_ROOT (default: datasets/full):
#   ramdocs_test.json                          – converted RAMDocs test split
#   tcb_ramdocs_test.json                      – TCB-rewritten RAMDocs
#   confact/humc_processed_cleaned.json  – ConfACT source (pre-req; see below)
#   confact/tcb_humc_processed_cleaned.json – TCB-ConfACT (Anthropic rewriter)
#
# NOTE: ConfACT has no Hugging Face source — the source JSON must be generated
# separately before this script runs:
#   bash experiments/build_confact_dataset.sh
#     (outputs to data/full/confact/ by default; pass --out-dir for a custom location)
#
# Usage:
#   ./experiments/download_and_build_tcb_datasets.sh [options]
#
# Options:
#   --dataset-root <path>      Output root (default: datasets/full)
#   --python <path>            Python executable (default: auto-detect)
#   --rewriter-provider <p>    gemini|local for RAMDocs rewriter (default: gemini)
#   --ramdocs-split <split>    RAMDocs HF split (default: test)
#   --skip-ramdocs             Skip RAMDocs download + TCB build
#   --skip-confact             Skip ConfACT TCB build
#   --confact-source <path>    Override path to preprocessed ConfACT JSON
#   --skip-download            Skip HF download steps (use existing JSON files)
#   --no-resume                Disable checkpoint resume (rebuild from scratch)
#   --checkpoint-every <n>     Save checkpoint every N instances (default: 100)
#   --max-rewrite-retries <n>  Rewrite retries per doc (default: 6)
#   --backoff-base <s>         Exponential backoff base seconds (default: 1.5)
#   --fail-on-error            Abort on first instance build failure
#   -h, --help                 Show this help message
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATASET_ROOT="datasets/full"
PYTHON_EXE=""
REWRITER_PROVIDER="gemini"
RAMDOCS_SPLIT="test"
SKIP_RAMDOCS=0
SKIP_CONFACT=0
CONFACT_SOURCE=""
SKIP_DOWNLOAD=0
RESUME_FLAG="--resume"
CHECKPOINT_EVERY=100
MAX_REWRITE_RETRIES=6
BACKOFF_BASE=1.5
FAIL_ON_ERROR_FLAG=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
print_help() {
  sed -n '3,/^set -euo/p' "$0" | grep "^#" | sed 's/^# \{0,2\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)   DATASET_ROOT="${2:?}"; shift 2 ;;
    --python)         PYTHON_EXE="${2:?}"; shift 2 ;;
    --rewriter-provider) REWRITER_PROVIDER="${2:?}"; shift 2 ;;
    --ramdocs-split)  RAMDOCS_SPLIT="${2:?}"; shift 2 ;;
    --skip-ramdocs)   SKIP_RAMDOCS=1; shift ;;
    --skip-confact)   SKIP_CONFACT=1; shift ;;
    --confact-source) CONFACT_SOURCE="${2:?}"; shift 2 ;;
    --skip-download)  SKIP_DOWNLOAD=1; shift ;;
    --no-resume)      RESUME_FLAG="--no-resume"; shift ;;
    --checkpoint-every) CHECKPOINT_EVERY="${2:?}"; shift 2 ;;
    --max-rewrite-retries) MAX_REWRITE_RETRIES="${2:?}"; shift 2 ;;
    --backoff-base)   BACKOFF_BASE="${2:?}"; shift 2 ;;
    --fail-on-error)  FAIL_ON_ERROR_FLAG="--fail-on-instance-error"; shift ;;
    -h|--help)        print_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Resolve Python
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
echo "[info] Using Python: $PYTHON_EXE ($("$PYTHON_EXE" --version 2>&1))"

# Create output directory
mkdir -p "$DATASET_ROOT"
echo "[info] Dataset root: $PROJECT_ROOT/$DATASET_ROOT"

# ---------------------------------------------------------------------------
# Helper: log timing
# ---------------------------------------------------------------------------
_ts() { date '+%H:%M:%S'; }
_step() { echo; echo "=== [$(_ts)] $* ==="; }

# ---------------------------------------------------------------------------
# 1. RAMDocs
# ---------------------------------------------------------------------------
RAMDOCS_JSON="$DATASET_ROOT/ramdocs_${RAMDOCS_SPLIT}.json"
TCB_RAMDOCS_JSON="$DATASET_ROOT/tcb_ramdocs_${RAMDOCS_SPLIT}.json"

if [[ $SKIP_RAMDOCS -eq 0 ]]; then

  if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    _step "Downloading RAMDocs (HanNight/RAMDocs, split=$RAMDOCS_SPLIT)"
    "$PYTHON_EXE" experiments/convert_ramdocs.py \
      --split "$RAMDOCS_SPLIT" \
      --output "$RAMDOCS_JSON" \
      --verbose
    echo "[info] Saved: $RAMDOCS_JSON"
  else
    echo "[info] Skipping download; expecting: $RAMDOCS_JSON"
    if [[ ! -f "$RAMDOCS_JSON" ]]; then
      echo "ERROR: $RAMDOCS_JSON not found. Remove --skip-download to fetch it." >&2
      exit 1
    fi
  fi

  _step "Building TCB for RAMDocs"
  "$PYTHON_EXE" experiments/build_tcb.py \
    --input "$RAMDOCS_JSON" \
    --benchmark ramdocs \
    --output "$TCB_RAMDOCS_JSON" \
    --rewriter-provider "$REWRITER_PROVIDER" \
    $RESUME_FLAG \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --max-rewrite-retries "$MAX_REWRITE_RETRIES" \
    --rewrite-backoff-base-seconds "$BACKOFF_BASE" \
    ${FAIL_ON_ERROR_FLAG:+"$FAIL_ON_ERROR_FLAG"} \
    --verbose
  echo "[info] Saved: $TCB_RAMDOCS_JSON"

else
  echo "[info] Skipping RAMDocs (--skip-ramdocs)"
fi

# ---------------------------------------------------------------------------
# 2. ConfACT
# ---------------------------------------------------------------------------
# ConfACT has no Hugging Face source — the preprocessed JSON must already
# exist.  Generate it first with:
#   bash experiments/build_confact_dataset.sh
#
CONFACT_OUT_DIR="$DATASET_ROOT/confact"
if [[ -z "$CONFACT_SOURCE" ]]; then
  CONFACT_SOURCE="$CONFACT_OUT_DIR/humc_processed_cleaned.json"
fi
TCB_CONFACT_JSON="$CONFACT_OUT_DIR/tcb_humc_processed_cleaned.json"

if [[ $SKIP_CONFACT -eq 0 ]]; then
  if [[ ! -f "$CONFACT_SOURCE" ]]; then
    echo "[warn] ConfACT source not found: $CONFACT_SOURCE"
    echo "[warn] Generate it with: bash experiments/build_confact_dataset.sh"
    echo "[warn] Skipping ConfACT TCB build."
  else
    _step "Building TCB for ConfACT (Anthropic rewriter)"
    mkdir -p "$CONFACT_OUT_DIR"
    bash "$SCRIPT_DIR/build_tcb_confact.sh" \
      --source "$CONFACT_SOURCE" \
      --dest-dir "$CONFACT_OUT_DIR" \
      --rewriter-provider anthropic
    echo "[info] Saved: $TCB_CONFACT_JSON"
  fi
else
  echo "[info] Skipping ConfACT (--skip-confact)"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
_step "All done"
echo
echo "Outputs written to $DATASET_ROOT/:"
for f in "$RAMDOCS_JSON" "$TCB_RAMDOCS_JSON" "$CONFACT_SOURCE" "$TCB_CONFACT_JSON"; do
  if [[ -f "$f" ]]; then
    count=$("$PYTHON_EXE" -c "import json; d=json.load(open('$f')); print(len(d))" 2>/dev/null || echo "?")
    echo "  $f  ($count instances)"
  fi
done
echo
echo "Next steps:"
echo "  1. Run BERTScore quality analysis (RAMDocs):"
echo "       python experiments/analyze_tcb_bertscore.py \\"
echo "         --tcb-file $TCB_RAMDOCS_JSON --orig-file $RAMDOCS_JSON \\"
echo "         --output-dir results/quality_analysis/ramdocs"
echo "  2. Run BERTScore quality analysis (ConfACT):"
echo "       python experiments/analyze_tcb_bertscore.py \\"
echo "         --tcb-file $TCB_CONFACT_JSON --orig-file $CONFACT_SOURCE \\"
echo "         --output-dir results/quality_analysis/confact"
echo "  3. Run cosine similarity sanity check (RAMDocs):"
echo "       python experiments/analyze_tcb_sentence_similarity.py \\"
echo "         --tcb-file $TCB_RAMDOCS_JSON --orig-file $RAMDOCS_JSON \\"
echo "         --output-dir results/quality_analysis/ramdocs"
