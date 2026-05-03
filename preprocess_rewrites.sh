#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE=""
REWRITER_PROVIDER="gemini"
RAMDOCS_SPLITS="test"
DATASET_ROOT="data"
PULL_MODE="full"
SAMPLE_SIZE=200
SAMPLE_SEED=42
TCB_RESUME=1
TCB_CHECKPOINT_EVERY=100
TCB_MAX_REWRITE_RETRIES=6
TCB_REWRITE_BACKOFF_BASE_SECONDS=1.5
TCB_FAIL_ON_INSTANCE_ERROR=0
SKIP_RAMDOCS=0
SKIP_CONFACT=0
CONFACT_SOURCE=""

print_help() {
  cat <<'EOF'
Usage: ./preprocess_rewrites.sh [options]

Download/convert source corpora, then build TCB datasets using the selected
rewrite provider (Gemini by default).

Options:
  --python <path>                Python executable to use
  --rewriter-provider <name>     gemini|local (default: gemini)
  --dataset-root <path>          Root folder for original + TCB outputs (default: data)
  --output-dir <path>            Alias for --dataset-root (backward compatible)
  --ramdocs-splits <csv>         Comma-separated RAMDocs splits (default: test)
  --pull-mode <mode>             full|sample (default: full)
  --sample-size <n>              Target matched sample size when --pull-mode sample
  --sample-seed <n>              RNG seed for sampling (default: 42)
  --tcb-resume <0|1>             Resume TCB build outputs (default: 1)
  --tcb-checkpoint-every <n>     Save TCB checkpoint every N instances
  --tcb-max-rewrite-retries <n>  Retry attempts for provider TCB rewrites
  --tcb-rewrite-backoff-base-seconds <s>
                                 Backoff base seconds for TCB rewrite retries
  --tcb-fail-on-instance-error <0|1>
                                 Abort TCB build on first instance error
  --skip-ramdocs                 Skip RAMDocs conversion + TCB build
  --skip-confact                 Skip ConfACT TCB build
  --confact-source <path>        Path to preprocessed ConfACT JSON
                                  (generate with: bash experiments/build_confact_dataset.sh)
  -h, --help                     Show this help message

Sample mode behavior:
  In sample mode, preprocessing converts the full split, then builds matched
  source/TCB pairs by trying instances in seeded random order and keeping only
  successful rewrites until target size is reached.
  It only fails if the full converted split is exhausted before reaching the
  requested target size.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_EXE="${2:-}"
      shift 2
      ;;
    --rewriter-provider)
      REWRITER_PROVIDER="${2:-}"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --output-dir)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --ramdocs-splits)
      RAMDOCS_SPLITS="${2:-}"
      shift 2
      ;;
    --pull-mode)
      PULL_MODE="${2:-}"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="${2:-}"
      shift 2
      ;;
    --sample-seed)
      SAMPLE_SEED="${2:-}"
      shift 2
      ;;
    --tcb-resume)
      TCB_RESUME="${2:-}"
      shift 2
      ;;
    --tcb-checkpoint-every)
      TCB_CHECKPOINT_EVERY="${2:-}"
      shift 2
      ;;
    --tcb-max-rewrite-retries)
      TCB_MAX_REWRITE_RETRIES="${2:-}"
      shift 2
      ;;
    --tcb-rewrite-backoff-base-seconds)
      TCB_REWRITE_BACKOFF_BASE_SECONDS="${2:-}"
      shift 2
      ;;
    --tcb-fail-on-instance-error)
      TCB_FAIL_ON_INSTANCE_ERROR="${2:-}"
      shift 2
      ;;
    --skip-ramdocs)
      SKIP_RAMDOCS=1
      shift
      ;;
    --skip-confact)
      SKIP_CONFACT=1
      shift
      ;;
    --confact-source)
      CONFACT_SOURCE="${2:-}"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [[ "$REWRITER_PROVIDER" != "gemini" && "$REWRITER_PROVIDER" != "local" ]]; then
  echo "Invalid --rewriter-provider: $REWRITER_PROVIDER" >&2
  exit 1
fi

if [[ "$PULL_MODE" != "full" && "$PULL_MODE" != "sample" ]]; then
  echo "Invalid --pull-mode: $PULL_MODE (expected full|sample)" >&2
  exit 1
fi

if ! [[ "$SAMPLE_SIZE" =~ ^[0-9]+$ ]] || (( SAMPLE_SIZE < 1 )); then
  echo "Invalid --sample-size: $SAMPLE_SIZE" >&2
  exit 1
fi

if ! [[ "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid --sample-seed: $SAMPLE_SEED" >&2
  exit 1
fi

if [[ "$TCB_RESUME" != "0" && "$TCB_RESUME" != "1" ]]; then
  echo "Invalid --tcb-resume: $TCB_RESUME (expected 0|1)" >&2
  exit 1
fi

if ! [[ "$TCB_CHECKPOINT_EVERY" =~ ^[0-9]+$ ]] || (( TCB_CHECKPOINT_EVERY < 1 )); then
  echo "Invalid --tcb-checkpoint-every: $TCB_CHECKPOINT_EVERY" >&2
  exit 1
fi

if ! [[ "$TCB_MAX_REWRITE_RETRIES" =~ ^[0-9]+$ ]] || (( TCB_MAX_REWRITE_RETRIES < 1 )); then
  echo "Invalid --tcb-max-rewrite-retries: $TCB_MAX_REWRITE_RETRIES" >&2
  exit 1
fi

if ! [[ "$TCB_REWRITE_BACKOFF_BASE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Invalid --tcb-rewrite-backoff-base-seconds: $TCB_REWRITE_BACKOFF_BASE_SECONDS" >&2
  exit 1
fi

if [[ "$TCB_FAIL_ON_INSTANCE_ERROR" != "0" && "$TCB_FAIL_ON_INSTANCE_ERROR" != "1" ]]; then
  echo "Invalid --tcb-fail-on-instance-error: $TCB_FAIL_ON_INSTANCE_ERROR (expected 0|1)" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "$PYTHON_EXE" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_EXE=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
  else
    PYTHON_EXE="$(command -v python)"
  fi
fi

mkdir -p "$DATASET_ROOT"

TCB_COMMON_ARGS=(
  --checkpoint-every "$TCB_CHECKPOINT_EVERY"
  --max-rewrite-retries "$TCB_MAX_REWRITE_RETRIES"
  --rewrite-backoff-base-seconds "$TCB_REWRITE_BACKOFF_BASE_SECONDS"
)
if [[ "$TCB_RESUME" == "1" ]]; then
  TCB_COMMON_ARGS+=(--resume)
else
  TCB_COMMON_ARGS+=(--no-resume)
fi
if [[ "$TCB_FAIL_ON_INSTANCE_ERROR" == "1" ]]; then
  TCB_COMMON_ARGS+=(--fail-on-instance-error)
fi

build_matched_sample_pair() {
  local source_path="$1"
  local benchmark_name="$2"
  local tcb_out_path="$3"

  "$PYTHON_EXE" -m experiments.build_tcb_matched_sample \
    --input "$source_path" \
    --benchmark "$benchmark_name" \
    --output-source "$source_path" \
    --output-tcb "$tcb_out_path" \
    --target-size "$SAMPLE_SIZE" \
    --sample-seed "$SAMPLE_SEED" \
    --rewriter-provider "$REWRITER_PROVIDER" \
    --max-rewrite-retries "$TCB_MAX_REWRITE_RETRIES" \
    --rewrite-backoff-base-seconds "$TCB_REWRITE_BACKOFF_BASE_SECONDS" \
    --require-full-target
}

if [[ "$SKIP_RAMDOCS" -eq 0 ]]; then
  IFS=',' read -r -a SPLITS <<< "$RAMDOCS_SPLITS"
  for split in "${SPLITS[@]}"; do
    if [[ -z "$split" ]]; then
      echo "Invalid RAMDocs split in --ramdocs-splits: $split" >&2
      exit 1
    fi

    ramdocs_out="$DATASET_ROOT/ramdocs_${split}.json"
    ramdocs_tcb_out="$DATASET_ROOT/tcb_ramdocs_${split}.json"

    echo "[preprocess] Converting RAMDocs split=$split -> $ramdocs_out"
    "$PYTHON_EXE" -m experiments.convert_ramdocs \
      --split "$split" \
      --output "$ramdocs_out"

    if [[ "$PULL_MODE" == "sample" ]]; then
      echo "[preprocess] Building matched sampled pair for RAMDocs split=$split (target=$SAMPLE_SIZE)"
      build_matched_sample_pair "$ramdocs_out" "ramdocs" "$ramdocs_tcb_out"
    else
      echo "[preprocess] Building TCB for RAMDocs split=$split with provider=$REWRITER_PROVIDER -> $ramdocs_tcb_out"
      "$PYTHON_EXE" -m experiments.build_tcb \
        --input "$ramdocs_out" \
        --benchmark ramdocs \
        --rewriter-provider "$REWRITER_PROVIDER" \
        --output "$ramdocs_tcb_out" \
        "${TCB_COMMON_ARGS[@]}"
    fi
  done
fi

# ---------------------------------------------------------------------------
# ConfACT — TCB build (no download step; source must be pre-generated)
# ---------------------------------------------------------------------------
# Generate the source file first:
#   bash experiments/build_confact_dataset.sh

if [[ "$SKIP_CONFACT" -eq 0 ]]; then
  CONFACT_OUT_DIR="$DATASET_ROOT/confact"
  if [[ -z "$CONFACT_SOURCE" ]]; then
    CONFACT_SOURCE="$CONFACT_OUT_DIR/humc_processed_cleaned.json"
  fi
  TCB_CONFACT_OUT="$CONFACT_OUT_DIR/tcb_humc_processed_cleaned.json"

  if [[ ! -f "$CONFACT_SOURCE" ]]; then
    echo "[preprocess] WARNING: ConfACT source not found: $CONFACT_SOURCE"
    echo "[preprocess] Run: bash experiments/build_confact_dataset.sh"
    echo "[preprocess] Skipping ConfACT TCB build."
  else
    echo "[preprocess] Building TCB for ConfACT with provider=anthropic -> $TCB_CONFACT_OUT"
    mkdir -p "$CONFACT_OUT_DIR"
    bash "$(dirname "${BASH_SOURCE[0]}")/experiments/build_tcb_confact.sh" \
      --source "$CONFACT_SOURCE" \
      --dest-dir "$CONFACT_OUT_DIR" \
      --rewriter-provider anthropic
  fi
fi

echo "[preprocess] Done."
