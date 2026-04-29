#!/usr/bin/env bash
# build_confact_dataset.sh — Build the full ConfACT HumC project-schema dataset.
#
# Pipeline
# --------
#   1. convert_confact.py               HumC.pkl.gz  →  humc.json
#      Loads the raw CONFACT pickle, converts to project schema, and enriches
#      source credibility from all_media_data.pkl.
#
#   2. confact_populate_supports_answers.py   humc.json + modC.json  →  humc_processed.json
#      Populates supports_answer ("yes"/"no"/"neither") on every evidence document
#      using the per-evidence majority_vote labels from the ModC sibling split.
#
#   3. clean_confact_v2.py              humc_processed.json  →  humc_processed_cleaned.json
#      Strips nav menus, cookie banners, and boilerplate from raw scraped content
#      using a 4-tier dispatch (zero-out / error-page / custom extractor / generic).
#
# All output JSON files land in data/full/confact/.
#
# Usage
# -----
#   ./experiments/build_confact_dataset.sh
#   ./experiments/build_confact_dataset.sh --dry-run
#   ./experiments/build_confact_dataset.sh --verbose
#   ./experiments/build_confact_dataset.sh \
#       --pkl-path   data/raw/confact/HumC.pkl.gz \
#       --modc       data/old/confact/modC.json \
#       --media-data data/raw/confact/all_media_data.pkl \
#       --out-dir    data/full/confact

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PKL_PATH="data/raw/confact/HumC.pkl.gz"
MEDIA_DATA="data/raw/confact/all_media_data.pkl"
MODC_PATH="data/old/confact/modC.json"
OUT_DIR="data/full/confact"
DRY_RUN=0
VERBOSE=""
VENV_DIR=".venv"

usage() {
  grep '^#' "$0" | sed 's/^# \{0,2\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pkl-path)   PKL_PATH="$2";   shift 2 ;;
    --modc)       MODC_PATH="$2";  shift 2 ;;
    --media-data) MEDIA_DATA="$2"; shift 2 ;;
    --out-dir)    OUT_DIR="$2";    shift 2 ;;
    --venv)       VENV_DIR="$2";   shift 2 ;;
    --dry-run)    DRY_RUN=1;       shift ;;
    --verbose|-v) VERBOSE="--verbose"; shift ;;
    --help|-h)    usage ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Activate venv if present
# ---------------------------------------------------------------------------

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Derived output paths
# ---------------------------------------------------------------------------

HUMC_JSON="$OUT_DIR/humc.json"
HUMC_PROCESSED="$OUT_DIR/humc_processed.json"
HUMC_CLEANED="$OUT_DIR/humc_processed_cleaned.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [[ ! -f "$PKL_PATH" ]]; then
  echo "ERROR: HumC pickle not found at $PKL_PATH"
  echo "       Place HumC.pkl.gz there or pass --pkl-path <path>"
  exit 1
fi

if [[ ! -f "$MODC_PATH" ]]; then
  echo "ERROR: modC.json not found at $MODC_PATH"
  echo "       Pass --modc <path> to specify its location"
  exit 1
fi

echo "============================================================"
echo "ConfACT HumC Dataset Builder"
echo "  pkl        : $PKL_PATH"
echo "  modC       : $MODC_PATH"
echo "  media-data : $MEDIA_DATA"
echo "  out-dir    : $OUT_DIR"
echo "  dry-run    : $DRY_RUN"
echo "============================================================"

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "[DRY RUN] Would execute:"
  echo "  Step 1: convert_confact.py         → $HUMC_JSON"
  echo "  Step 2: confact_populate_...py     → $HUMC_PROCESSED"
  echo "  Step 3: clean_confact_v2.py        → $HUMC_CLEANED"
  exit 0
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Step 1: Convert HumC.pkl.gz → humc.json (project schema)
# ---------------------------------------------------------------------------

echo ""
echo ">>> Step 1/3: convert_confact.py  →  $HUMC_JSON"

MEDIA_FLAG=""
if [[ -f "$MEDIA_DATA" ]]; then
  MEDIA_FLAG="--media-data $MEDIA_DATA --skip-media-download"
else
  echo "  (media-data not found at $MEDIA_DATA — running with --no-media)"
  MEDIA_FLAG="--no-media"
fi

$PYTHON experiments/convert_confact.py \
  --skip-download \
  --pkl-path  "$PKL_PATH" \
  --output    "$HUMC_JSON" \
  ${MEDIA_FLAG} \
  ${VERBOSE:+--verbose}

echo "  OK → $HUMC_JSON"

# ---------------------------------------------------------------------------
# Step 2: Enrich supports_answer using modC majority_vote
# ---------------------------------------------------------------------------

echo ""
echo ">>> Step 2/3: confact_populate_supports_answers.py  →  $HUMC_PROCESSED"

$PYTHON experiments/confact_populate_supports_answers.py \
  --humc   "$HUMC_JSON" \
  --modc   "$MODC_PATH" \
  --output "$HUMC_PROCESSED" \
  ${VERBOSE:+--verbose}

echo "  OK → $HUMC_PROCESSED"

# ---------------------------------------------------------------------------
# Step 3: Clean scraped content
# ---------------------------------------------------------------------------

echo ""
echo ">>> Step 3/3: clean_confact_v2.py  →  $HUMC_CLEANED"

$PYTHON experiments/clean_confact_v2.py \
  --input  "$HUMC_PROCESSED" \
  --output "$HUMC_CLEANED" \
  ${VERBOSE:+--verbose}

echo "  OK → $HUMC_CLEANED"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "DONE"
echo "  $HUMC_JSON"
echo "  $HUMC_PROCESSED"
echo "  $HUMC_CLEANED"
echo "============================================================"
