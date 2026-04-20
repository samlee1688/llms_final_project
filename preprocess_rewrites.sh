#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE=""
REWRITER_PROVIDER="gemini"
AMBIGDOCS_SPLITS="train,validation,test"
FAITHEVAL_SPLITS="train,validation,test"
DATASET_ROOT="data"
PULL_MODE="full"
SAMPLE_SIZE=200
SAMPLE_SEED=42
TCB_RESUME=1
TCB_CHECKPOINT_EVERY=100
TCB_MAX_REWRITE_RETRIES=6
TCB_REWRITE_BACKOFF_BASE_SECONDS=1.5
TCB_FAIL_ON_INSTANCE_ERROR=0
SKIP_AMBIGDOCS=0
SKIP_FAITHEVAL=0

print_help() {
  cat <<'EOF'
Usage: ./preprocess_rewrites.sh [options]

Download/convert full AmbigDocs and FaithEval corpora, then build
TCB datasets using the selected rewrite provider (Gemini by default).

Options:
  --python <path>                Python executable to use
  --rewriter-provider <name>     gemini|local (default: gemini)
  --dataset-root <path>          Root folder for original + TCB outputs (default: data)
  --output-dir <path>            Alias for --dataset-root (backward compatible)
  --ambigdocs-splits <csv>       Comma-separated splits (default: train,validation,test)
  --faitheval-splits <csv>       Comma-separated splits (default: train,validation,test)
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
  --skip-ambigdocs               Skip AmbigDocs conversion + TCB build
  --skip-faitheval               Skip FaithEval conversion + TCB build
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
    --ambigdocs-splits)
      AMBIGDOCS_SPLITS="${2:-}"
      shift 2
      ;;
    --faitheval-splits)
      FAITHEVAL_SPLITS="${2:-}"
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
    --skip-ambigdocs)
      SKIP_AMBIGDOCS=1
      shift
      ;;
    --skip-faitheval)
      SKIP_FAITHEVAL=1
      shift
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

if [[ "$SKIP_AMBIGDOCS" -eq 0 ]]; then
  IFS=',' read -r -a SPLITS <<< "$AMBIGDOCS_SPLITS"
  for split in "${SPLITS[@]}"; do
    case "$split" in
      train|validation|test)
        ;;
      *)
        echo "Invalid AmbigDocs split in --ambigdocs-splits: $split" >&2
        exit 1
        ;;
    esac

    ambig_out="$DATASET_ROOT/ambigdocs_${split}.json"
    ambig_tcb_out="$DATASET_ROOT/tcb_ambigdocs_${split}.json"

    echo "[preprocess] Converting AmbigDocs split=$split -> $ambig_out"
    "$PYTHON_EXE" -m experiments.convert_ambigdocs \
      --split "$split" \
      --output "$ambig_out"

    if [[ "$PULL_MODE" == "sample" ]]; then
      echo "[preprocess] Building matched sampled pair for AmbigDocs split=$split (target=$SAMPLE_SIZE)"
      build_matched_sample_pair "$ambig_out" "ambigdocs" "$ambig_tcb_out"
    else
      echo "[preprocess] Building TCB for AmbigDocs split=$split with provider=$REWRITER_PROVIDER -> $ambig_tcb_out"
      "$PYTHON_EXE" -m experiments.build_tcb \
        --input "$ambig_out" \
        --benchmark ambigdocs \
        --rewriter-provider "$REWRITER_PROVIDER" \
        --output "$ambig_tcb_out" \
        "${TCB_COMMON_ARGS[@]}"
    fi
  done
fi

if [[ "$SKIP_FAITHEVAL" -eq 0 ]]; then
  IFS=',' read -r -a FAITH_SPLITS <<< "$FAITHEVAL_SPLITS"
  for split in "${FAITH_SPLITS[@]}"; do
    case "$split" in
      train|validation|test)
        ;;
      *)
        echo "Invalid FaithEval split in --faitheval-splits: $split" >&2
        exit 1
        ;;
    esac
  done

  AVAILABLE_FAITH_SPLITS=()
  while IFS= read -r split_name; do
    if [[ -n "$split_name" ]]; then
      AVAILABLE_FAITH_SPLITS+=("$split_name")
    fi
  done < <(
    "$PYTHON_EXE" - <<'PY'
from datasets import get_dataset_split_names

datasets = [
    "Salesforce/FaithEval-counterfactual-v1.0",
    "Salesforce/FaithEval-inconsistent-v1.0",
]
split_sets = []
for name in datasets:
    split_sets.append(set(get_dataset_split_names(name)))

common = sorted(set.intersection(*split_sets)) if split_sets else []
for s in common:
    print(s)
PY
  )

if [[ "${#AVAILABLE_FAITH_SPLITS[@]}" -eq 0 ]]; then
  echo "No available FaithEval splits were discovered from Hugging Face." >&2
  exit 1
fi

processed_any=0
for split in "${FAITH_SPLITS[@]}"; do
  available=0
  for s in "${AVAILABLE_FAITH_SPLITS[@]}"; do
    if [[ "$s" == "$split" ]]; then
      available=1
      break
    fi
  done
  if [[ "$available" -ne 1 ]]; then
    echo "[preprocess] Skipping FaithEval split=$split (not available on HF; available: ${AVAILABLE_FAITH_SPLITS[*]})"
    continue
  fi

  processed_any=1
  if [[ "$split" == "test" ]]; then
    faith_out="$DATASET_ROOT/faitheval_combined.json"
    faith_tcb_out="$DATASET_ROOT/tcb_faitheval.json"
  else
    faith_out="$DATASET_ROOT/faitheval_combined_${split}.json"
    faith_tcb_out="$DATASET_ROOT/tcb_faitheval_${split}.json"
  fi

  echo "[preprocess] Converting FaithEval split=$split -> $faith_out"
  "$PYTHON_EXE" -m experiments.convert_faitheval \
    --dataset-types counterfactual inconsistent \
    --split "$split" \
    --output "$faith_out"

  if [[ "$PULL_MODE" == "sample" ]]; then
    echo "[preprocess] Building matched sampled pair for FaithEval split=$split (target=$SAMPLE_SIZE)"
    build_matched_sample_pair "$faith_out" "faitheval" "$faith_tcb_out"
  else
    echo "[preprocess] Building TCB for FaithEval split=$split with provider=$REWRITER_PROVIDER -> $faith_tcb_out"
    "$PYTHON_EXE" -m experiments.build_tcb \
      --input "$faith_out" \
      --benchmark faitheval \
      --rewriter-provider "$REWRITER_PROVIDER" \
      --output "$faith_tcb_out" \
      "${TCB_COMMON_ARGS[@]}"
  fi
  done

  if [[ "$processed_any" -eq 0 ]]; then
    echo "No requested FaithEval splits were available. Requested: $FAITHEVAL_SPLITS; available: ${AVAILABLE_FAITH_SPLITS[*]}" >&2
    exit 1
  fi
fi

echo "[preprocess] Done."
