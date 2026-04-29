#!/usr/bin/env bash
set -euo pipefail

INSTALL_DEPENDENCIES=1
BUILD_DATASETS=1
RUN_EXPERIMENTS=1
CREATE_VENV=1
RAMDOCS_SPLIT="test"
JUDGE_SUITE="both"
AGENT_PROVIDER="openai"
AGENT_MODEL=""
STANDARD_RAG_PROVIDER=""
STANDARD_RAG_MODEL=""
MADAM_AGENT_PROVIDER=""
MADAM_AGENT_MODEL=""
EAJP_AGENT_PROVIDER=""
EAJP_AGENT_MODEL=""
OPENAI_JUDGE_MODEL=""
ANTHROPIC_JUDGE_MODEL="claude-haiku-4-5-20251001"
OUTPUT_ROOT="results/full_run"
RUN_DATASET_ROOT="data"
VENV_DIR=".venv"
PYTHON_EXE=""
REWRITER_PROVIDER="gemini"
PREPROCESS_REWRITES=0
REWRITE_DATASET_PROVIDER="gemini"
PREPROCESS_DATASET_ROOT="data"
PREPROCESS_PULL_MODE="full"
PREPROCESS_SAMPLE_SIZE=200
PREPROCESS_SAMPLE_SEED=42
TCB_RESUME=1
TCB_CHECKPOINT_EVERY=100
TCB_MAX_REWRITE_RETRIES=6
TCB_REWRITE_BACKOFF_BASE_SECONDS=1.5
TCB_FAIL_ON_INSTANCE_ERROR=0

PARALLEL_DATASETS=0
MAX_DATASET_JOBS=2
TOTAL_WORKER_BUDGET=6
THROTTLE_MS=0
RESUME_RUNS=1
FAST_MODE=0
SAMPLE_SIZE=120
SAMPLE_SEED=42
JUDGE_ROUNDS=""
MADAM_MAX_PARALLEL_AGENTS=4
MADAM_COMPACT_DOC_CHARS=1200
MADAM_ARGUMENT_SUMMARY_CHARS=240
MADAM_EARLY_STOP_PATIENCE=1
MADAM_INCLUDE_FULL_DOCUMENT_IN_DEBATE=0
EAJP_SHORT_CIRCUIT=0
JUDGE_ONLY_ON_CONTENTION=0
LLM_PROVIDER_PROMPT_CACHE=1
VALIDATE_EAJP_SCHEMA=1
EAJP_SCHEMA_STRICT_ENUMS=1
COMPUTE_PROPOSAL_METRICS=1
MAX_ABLATION_ROUNDS=""

print_help() {
  cat <<'EOF'
Usage: ./run_experiments.sh [options]

End-to-end workflow:
  1) create/update Python environment
  2) install dependencies
  3) convert source datasets
  4) build TCB datasets
    5) run standard_rag, madam_rag, eajp_scorecard_only, eajp
      and cache-driven round ablations from 0..max rounds
  6) compute benchmark summary table from produced metrics

Options:
  --skip-install                Skip pip install -r requirements.txt
  --skip-build-datasets         Skip dataset conversion and TCB building
  --skip-run                    Skip experiment execution
  --skip-venv                   Use existing Python without creating venv
  --ramdocs-split <split>       RAMDocs split (default: test)
  --judge-suite <suite>         openai|anthropic|both (default: both)
  --agent-provider <provider>   openai|anthropic (default: openai)
  --agent-model <model>         Override agent model
  --standard-rag-provider <p>   Provider for standard_rag only
  --standard-rag-model <m>      Model for standard_rag only
  --madam-agent-provider <p>    Provider for madam_rag agents only
  --madam-agent-model <m>       Model for madam_rag agents only
  --eajp-agent-provider <p>     Provider for eajp agents only
  --eajp-agent-model <m>        Model for eajp agents only
  --openai-judge-model <m>      Model for OpenAI judge suite
  --anthropic-judge-model <m>   Model for Anthropic judge suite
  --output-root <path>          Output root (default: results/full_run)
  --run-dataset-root <path>     Dataset root used for conversion/build/run inputs (default: data)
  --venv-dir <path>             Virtualenv directory (default: .venv)
  --python <path>               Base python executable
  --rewriter-provider <name>    gemini|local (default: gemini)
  --preprocess-rewrites         Download full corpora and prebuild TCB artifacts
  --rewrite-dataset-provider <name>
                                gemini|local provider for preprocessing TCB rewrites
  --preprocess-dataset-root <path>
                                Root folder for preprocessing dataset/TCB outputs (default: data)
  --preprocess-pull-mode <mode> full|sample (default: full)
  --preprocess-sample-size <n>  Sample size when preprocess pull mode is sample
  --preprocess-sample-seed <n>  Sample RNG seed for preprocess sampling (default: 42)
  --tcb-resume                  Resume TCB builds from existing output (default)
  --no-tcb-resume               Rebuild TCB outputs from scratch
  --tcb-checkpoint-every <n>    Save TCB build checkpoint every N instances
  --tcb-max-rewrite-retries <n>
                                Retry attempts for provider TCB rewrites
  --tcb-rewrite-backoff-base-seconds <s>
                                Backoff base seconds for TCB rewrite retries
  --tcb-fail-on-instance-error  Abort TCB build on first instance-level error

  --parallel-datasets           Run dataset experiment groups in parallel
  --max-dataset-jobs <n>        Max dataset jobs in parallel (default: 2)
  --total-worker-budget <n>     Total per-process worker budget (default: 6)
  --throttle-ms <n>             Min provider call interval per process in ms

  --fast-mode                   Enable sampled fast benchmark mode
  --sample-size <n>             Sample size per dataset when fast mode is on
  --sample-seed <n>             Random seed for sampled subsets
  --judge-rounds <n>            MADAM-RAG/EAJP debate rounds override
  --max-ablation-rounds <n>     Maximum round for cache-driven ablations (default: judge-rounds)
  --madam-max-parallel-agents <n>
                                 Max concurrent per-round agent calls
  --madam-compact-doc-chars <n> Character cap for debate document excerpts
  --madam-argument-summary-chars <n>
                                 Character cap for each opposing argument
  --madam-early-stop-patience <n>
                                 Stable consensus rounds required to stop early
  --madam-include-full-document-in-debate
                                 Include full doc each debate round
  --eajp-short-circuit          Allow EAJP to skip final pass on confident scorecard
  --no-eajp-short-circuit       Disable EAJP short-circuit (strict EAJP flow; default)
  --judge-only-on-contention    Skip MADAM/EAJP on uncontested instances
  --no-provider-prompt-cache    Disable provider-side prompt cache hints
  --validate-eajp-schema        Validate EAJP outputs after run (default)
  --no-validate-eajp-schema     Skip EAJP schema validation post-run
  --eajp-schema-strict-enums    Enforce strict enum labels in validator (default)
  --no-eajp-schema-strict-enums Relax enum-label checks in validator
  --compute-proposal-metrics    Compute proposal metrics from details outputs after run (default)
  --no-compute-proposal-metrics Skip proposal-metric computation post-run

  --no-resume                   Disable resume from partial checkpoints
  --resume                      Enable resume from partial checkpoints (default)

  -h, --help                    Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      INSTALL_DEPENDENCIES=0
      shift
      ;;
    --skip-build-datasets)
      BUILD_DATASETS=0
      shift
      ;;
    --skip-run)
      RUN_EXPERIMENTS=0
      shift
      ;;
    --skip-venv)
      CREATE_VENV=0
      shift
      ;;
    --ramdocs-split)
      RAMDOCS_SPLIT="${2:-}"
      shift 2
      ;;
    --judge-suite)
      JUDGE_SUITE="${2:-}"
      shift 2
      ;;
    --agent-provider)
      AGENT_PROVIDER="${2:-}"
      shift 2
      ;;
    --agent-model)
      AGENT_MODEL="${2:-}"
      shift 2
      ;;
    --standard-rag-provider)
      STANDARD_RAG_PROVIDER="${2:-}"
      shift 2
      ;;
    --standard-rag-model)
      STANDARD_RAG_MODEL="${2:-}"
      shift 2
      ;;
    --madam-agent-provider)
      MADAM_AGENT_PROVIDER="${2:-}"
      shift 2
      ;;
    --madam-agent-model)
      MADAM_AGENT_MODEL="${2:-}"
      shift 2
      ;;
    --eajp-agent-provider)
      EAJP_AGENT_PROVIDER="${2:-}"
      shift 2
      ;;
    --eajp-agent-model)
      EAJP_AGENT_MODEL="${2:-}"
      shift 2
      ;;
    --openai-judge-model)
      OPENAI_JUDGE_MODEL="${2:-}"
      shift 2
      ;;
    --anthropic-judge-model)
      ANTHROPIC_JUDGE_MODEL="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --run-dataset-root)
      RUN_DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_EXE="${2:-}"
      shift 2
      ;;
    --rewriter-provider)
      REWRITER_PROVIDER="${2:-}"
      shift 2
      ;;
    --preprocess-rewrites)
      PREPROCESS_REWRITES=1
      shift
      ;;
    --rewrite-dataset-provider)
      REWRITE_DATASET_PROVIDER="${2:-}"
      shift 2
      ;;
    --preprocess-dataset-root)
      PREPROCESS_DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --preprocess-pull-mode)
      PREPROCESS_PULL_MODE="${2:-}"
      shift 2
      ;;
    --preprocess-sample-size)
      PREPROCESS_SAMPLE_SIZE="${2:-}"
      shift 2
      ;;
    --preprocess-sample-seed)
      PREPROCESS_SAMPLE_SEED="${2:-}"
      shift 2
      ;;
    --tcb-resume)
      TCB_RESUME=1
      shift
      ;;
    --no-tcb-resume)
      TCB_RESUME=0
      shift
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
      TCB_FAIL_ON_INSTANCE_ERROR=1
      shift
      ;;
    --parallel-datasets)
      PARALLEL_DATASETS=1
      shift
      ;;
    --max-dataset-jobs)
      MAX_DATASET_JOBS="${2:-}"
      shift 2
      ;;
    --total-worker-budget)
      TOTAL_WORKER_BUDGET="${2:-}"
      shift 2
      ;;
    --throttle-ms)
      THROTTLE_MS="${2:-}"
      shift 2
      ;;
    --fast-mode)
      FAST_MODE=1
      shift
      ;;
    --sample-size)
      SAMPLE_SIZE="${2:-}"
      shift 2
      ;;
    --sample-seed)
      SAMPLE_SEED="${2:-}"
      shift 2
      ;;
    --judge-rounds)
      JUDGE_ROUNDS="${2:-}"
      shift 2
      ;;
    --max-ablation-rounds)
      MAX_ABLATION_ROUNDS="${2:-}"
      shift 2
      ;;
    --madam-max-parallel-agents)
      MADAM_MAX_PARALLEL_AGENTS="${2:-}"
      shift 2
      ;;
    --madam-compact-doc-chars)
      MADAM_COMPACT_DOC_CHARS="${2:-}"
      shift 2
      ;;
    --madam-argument-summary-chars)
      MADAM_ARGUMENT_SUMMARY_CHARS="${2:-}"
      shift 2
      ;;
    --madam-early-stop-patience)
      MADAM_EARLY_STOP_PATIENCE="${2:-}"
      shift 2
      ;;
    --madam-include-full-document-in-debate)
      MADAM_INCLUDE_FULL_DOCUMENT_IN_DEBATE=1
      shift
      ;;
    --no-eajp-short-circuit)
      EAJP_SHORT_CIRCUIT=0
      shift
      ;;
    --eajp-short-circuit)
      EAJP_SHORT_CIRCUIT=1
      shift
      ;;
    --judge-only-on-contention)
      JUDGE_ONLY_ON_CONTENTION=1
      shift
      ;;
    --no-provider-prompt-cache)
      LLM_PROVIDER_PROMPT_CACHE=0
      shift
      ;;
    --validate-eajp-schema)
      VALIDATE_EAJP_SCHEMA=1
      shift
      ;;
    --no-validate-eajp-schema)
      VALIDATE_EAJP_SCHEMA=0
      shift
      ;;
    --eajp-schema-strict-enums)
      EAJP_SCHEMA_STRICT_ENUMS=1
      shift
      ;;
    --no-eajp-schema-strict-enums)
      EAJP_SCHEMA_STRICT_ENUMS=0
      shift
      ;;
    --compute-proposal-metrics)
      COMPUTE_PROPOSAL_METRICS=1
      shift
      ;;
    --no-compute-proposal-metrics)
      COMPUTE_PROPOSAL_METRICS=0
      shift
      ;;
    --no-resume)
      RESUME_RUNS=0
      shift
      ;;
    --resume)
      RESUME_RUNS=1
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

if [[ "$JUDGE_SUITE" != "openai" && "$JUDGE_SUITE" != "anthropic" && "$JUDGE_SUITE" != "both" ]]; then
  echo "Invalid --judge-suite: $JUDGE_SUITE" >&2
  exit 1
fi

if [[ "$AGENT_PROVIDER" != "openai" && "$AGENT_PROVIDER" != "anthropic" ]]; then
  echo "Invalid --agent-provider: $AGENT_PROVIDER" >&2
  exit 1
fi

if [[ -z "$RAMDOCS_SPLIT" ]]; then
  echo "Invalid --ramdocs-split: $RAMDOCS_SPLIT" >&2
  exit 1
fi

if [[ "$REWRITE_DATASET_PROVIDER" != "gemini" && "$REWRITE_DATASET_PROVIDER" != "local" ]]; then
  echo "Invalid --rewrite-dataset-provider: $REWRITE_DATASET_PROVIDER" >&2
  exit 1
fi

if [[ "$PREPROCESS_PULL_MODE" != "full" && "$PREPROCESS_PULL_MODE" != "sample" ]]; then
  echo "Invalid --preprocess-pull-mode: $PREPROCESS_PULL_MODE" >&2
  exit 1
fi

if ! [[ "$PREPROCESS_SAMPLE_SIZE" =~ ^[0-9]+$ ]] || (( PREPROCESS_SAMPLE_SIZE < 1 )); then
  echo "Invalid --preprocess-sample-size: $PREPROCESS_SAMPLE_SIZE" >&2
  exit 1
fi

if ! [[ "$PREPROCESS_SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid --preprocess-sample-seed: $PREPROCESS_SAMPLE_SEED" >&2
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

if ! [[ "$MAX_DATASET_JOBS" =~ ^[0-9]+$ ]] || (( MAX_DATASET_JOBS < 1 )); then
  echo "Invalid --max-dataset-jobs: $MAX_DATASET_JOBS" >&2
  exit 1
fi

if ! [[ "$TOTAL_WORKER_BUDGET" =~ ^[0-9]+$ ]] || (( TOTAL_WORKER_BUDGET < 1 )); then
  echo "Invalid --total-worker-budget: $TOTAL_WORKER_BUDGET" >&2
  exit 1
fi

if ! [[ "$THROTTLE_MS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --throttle-ms: $THROTTLE_MS" >&2
  exit 1
fi

if ! [[ "$SAMPLE_SIZE" =~ ^[0-9]+$ ]] || (( SAMPLE_SIZE < 1 )); then
  echo "Invalid --sample-size: $SAMPLE_SIZE" >&2
  exit 1
fi

if [[ -n "$JUDGE_ROUNDS" ]]; then
  if ! [[ "$JUDGE_ROUNDS" =~ ^[0-9]+$ ]] || (( JUDGE_ROUNDS < 0 )); then
    echo "Invalid --judge-rounds: $JUDGE_ROUNDS" >&2
    exit 1
  fi
fi

if [[ -n "$MAX_ABLATION_ROUNDS" ]]; then
  if ! [[ "$MAX_ABLATION_ROUNDS" =~ ^[0-9]+$ ]]; then
    echo "Invalid --max-ablation-rounds: $MAX_ABLATION_ROUNDS" >&2
    exit 1
  fi
fi

if ! [[ "$MADAM_MAX_PARALLEL_AGENTS" =~ ^[0-9]+$ ]] || (( MADAM_MAX_PARALLEL_AGENTS < 1 )); then
  echo "Invalid --madam-max-parallel-agents: $MADAM_MAX_PARALLEL_AGENTS" >&2
  exit 1
fi
if ! [[ "$MADAM_COMPACT_DOC_CHARS" =~ ^[0-9]+$ ]] || (( MADAM_COMPACT_DOC_CHARS < 200 )); then
  echo "Invalid --madam-compact-doc-chars: $MADAM_COMPACT_DOC_CHARS" >&2
  exit 1
fi
if ! [[ "$MADAM_ARGUMENT_SUMMARY_CHARS" =~ ^[0-9]+$ ]] || (( MADAM_ARGUMENT_SUMMARY_CHARS < 50 )); then
  echo "Invalid --madam-argument-summary-chars: $MADAM_ARGUMENT_SUMMARY_CHARS" >&2
  exit 1
fi
if ! [[ "$MADAM_EARLY_STOP_PATIENCE" =~ ^[0-9]+$ ]]; then
  echo "Invalid --madam-early-stop-patience: $MADAM_EARLY_STOP_PATIENCE" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "$PYTHON_EXE" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python)"
  else
    echo "No python executable found in PATH" >&2
    exit 1
  fi
fi

if [[ "$CREATE_VENV" -eq 1 ]]; then
  echo "[setup] Creating/updating virtualenv at $VENV_DIR"
  "$PYTHON_EXE" -m venv "$VENV_DIR"
  PYTHON_RUN="$PROJECT_ROOT/$VENV_DIR/bin/python"
else
  echo "[setup] Using base Python without creating virtualenv"
  PYTHON_RUN="$PYTHON_EXE"
fi

echo "[setup] Python: $PYTHON_RUN"

if [[ "$INSTALL_DEPENDENCIES" -eq 1 ]]; then
  echo "[setup] Installing dependencies"
  "$PYTHON_RUN" -m pip install --upgrade pip
  "$PYTHON_RUN" -m pip install -r requirements.txt
fi

mkdir -p data
mkdir -p "$RUN_DATASET_ROOT"
mkdir -p "$OUTPUT_ROOT"

TCB_COMMON_ARGS=(
  --checkpoint-every "$TCB_CHECKPOINT_EVERY"
  --max-rewrite-retries "$TCB_MAX_REWRITE_RETRIES"
  --rewrite-backoff-base-seconds "$TCB_REWRITE_BACKOFF_BASE_SECONDS"
)
if [[ "$TCB_RESUME" -eq 1 ]]; then
  TCB_COMMON_ARGS+=(--resume)
else
  TCB_COMMON_ARGS+=(--no-resume)
fi
if [[ "$TCB_FAIL_ON_INSTANCE_ERROR" -eq 1 ]]; then
  TCB_COMMON_ARGS+=(--fail-on-instance-error)
fi

RAMDOCS_OUT="$RUN_DATASET_ROOT/ramdocs_${RAMDOCS_SPLIT}.json"

TCB_RAMDOCS_OUT="$RUN_DATASET_ROOT/tcb_ramdocs.json"
if [[ -f "$RUN_DATASET_ROOT/tcb_ramdocs_${RAMDOCS_SPLIT}.json" ]]; then
  TCB_RAMDOCS_OUT="$RUN_DATASET_ROOT/tcb_ramdocs_${RAMDOCS_SPLIT}.json"
fi

# ConfACT uses a subdirectory and its own dedicated builder
CONFACT_SOURCE="${CONFACT_SOURCE:-$RUN_DATASET_ROOT/confact/humc_processed_cleaned.json}"
TCB_CONFACT_OUT="${TCB_CONFACT_OUT:-$RUN_DATASET_ROOT/confact/tcb_humc_processed_cleaned.json}"

if [[ "$BUILD_DATASETS" -eq 1 ]]; then
  echo "[data] Converting RAMDocs"
  "$PYTHON_RUN" -m experiments.convert_ramdocs \
    --split "$RAMDOCS_SPLIT" \
    --output "$RAMDOCS_OUT"

  echo "[data] Building TCB from RAMDocs"
  "$PYTHON_RUN" -m experiments.build_tcb \
    --input "$RAMDOCS_OUT" \
    --benchmark ramdocs \
    --rewriter-provider "$REWRITER_PROVIDER" \
    --output "$TCB_RAMDOCS_OUT" \
    "${TCB_COMMON_ARGS[@]}"

  echo "[data] Building ConfACT TCB (uses dedicated builder with Anthropic rewriter)"
  if [[ -f "$CONFACT_SOURCE" ]]; then
    bash "$PROJECT_ROOT/experiments/build_tcb_confact.sh" \
      --source "$CONFACT_SOURCE" \
      --dest-dir "$(dirname "$TCB_CONFACT_OUT")" \
      --rewriter-provider "anthropic" \
      ${VERBOSE:+--verbose}
  else
    echo "[data] WARNING: ConfACT source not found at $CONFACT_SOURCE — skipping TCB build"
    echo "[data]   Run experiments/clean_confact_v2.py first to generate the source file."
  fi

  if [[ "$PREPROCESS_REWRITES" -eq 1 ]]; then
    echo "[data] Preprocessing full corpora and prebuilding TCB artifacts"
    bash "$PROJECT_ROOT/preprocess_rewrites.sh" \
      --python "$PYTHON_RUN" \
      --rewriter-provider "$REWRITE_DATASET_PROVIDER" \
      --dataset-root "$PREPROCESS_DATASET_ROOT" \
      --pull-mode "$PREPROCESS_PULL_MODE" \
      --sample-size "$PREPROCESS_SAMPLE_SIZE" \
      --sample-seed "$PREPROCESS_SAMPLE_SEED" \
      --tcb-resume "$TCB_RESUME" \
      --tcb-checkpoint-every "$TCB_CHECKPOINT_EVERY" \
      --tcb-max-rewrite-retries "$TCB_MAX_REWRITE_RETRIES" \
      --tcb-rewrite-backoff-base-seconds "$TCB_REWRITE_BACKOFF_BASE_SECONDS" \
      --tcb-fail-on-instance-error "$TCB_FAIL_ON_INSTANCE_ERROR"
  fi
fi

run_one() {
  local name="$1"
  local dataset_path="$2"
  local benchmark="$3"
  local workers="$4"
  local throttle="$5"

  local out_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$out_dir"

  local configured_rounds="$JUDGE_ROUNDS"
  if [[ -z "$configured_rounds" ]]; then
    configured_rounds="${DEBATE_ROUNDS:-3}"
  fi
  local ablation_max="$MAX_ABLATION_ROUNDS"
  if [[ -z "$ablation_max" ]]; then
    ablation_max="$configured_rounds"
  fi
  local ablation_round_args=()
  for ((r=0; r<=ablation_max; r++)); do
    ablation_round_args+=("$r")
  done

  echo "[run] $name -> benchmark=$benchmark, dataset=$dataset_path, workers=$workers, throttle_ms=$throttle"

  local cmd=(
    "$PYTHON_RUN" -m experiments.run_experiment
    --dataset "$dataset_path"
    --benchmark "$benchmark"
    --systems standard_rag madam_rag eajp_scorecard_only eajp
    --judge-suite "$JUDGE_SUITE"
    --agent-provider "$AGENT_PROVIDER"
    --max-workers "$workers"
    --output "$out_dir"
    --reuse-madam-debate-for-eajp
    --ablate-rounds-from-cache "${ablation_round_args[@]}"
  )

  if [[ -n "$AGENT_MODEL" ]]; then
    cmd+=(--agent-model "$AGENT_MODEL")
  fi

  if [[ -n "$STANDARD_RAG_PROVIDER" ]]; then
    cmd+=(--standard-rag-provider "$STANDARD_RAG_PROVIDER")
  fi
  if [[ -n "$STANDARD_RAG_MODEL" ]]; then
    cmd+=(--standard-rag-model "$STANDARD_RAG_MODEL")
  fi
  if [[ -n "$MADAM_AGENT_PROVIDER" ]]; then
    cmd+=(--madam-agent-provider "$MADAM_AGENT_PROVIDER")
  fi
  if [[ -n "$MADAM_AGENT_MODEL" ]]; then
    cmd+=(--madam-agent-model "$MADAM_AGENT_MODEL")
  fi
  if [[ -n "$EAJP_AGENT_PROVIDER" ]]; then
    cmd+=(--eajp-agent-provider "$EAJP_AGENT_PROVIDER")
  fi
  if [[ -n "$EAJP_AGENT_MODEL" ]]; then
    cmd+=(--eajp-agent-model "$EAJP_AGENT_MODEL")
  fi
  if [[ -n "$OPENAI_JUDGE_MODEL" ]]; then
    cmd+=(--openai-judge-model "$OPENAI_JUDGE_MODEL")
  fi
  if [[ -n "$ANTHROPIC_JUDGE_MODEL" ]]; then
    cmd+=(--anthropic-judge-model "$ANTHROPIC_JUDGE_MODEL")
  fi

  if [[ "$FAST_MODE" -eq 1 ]]; then
    cmd+=(--fast-mode --fast-mode-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
  fi

  if [[ -n "$JUDGE_ROUNDS" ]]; then
    cmd+=(--judge-rounds "$JUDGE_ROUNDS")
  fi

  cmd+=(
    --madam-max-parallel-agents "$MADAM_MAX_PARALLEL_AGENTS"
    --madam-compact-doc-chars "$MADAM_COMPACT_DOC_CHARS"
    --madam-argument-summary-chars "$MADAM_ARGUMENT_SUMMARY_CHARS"
    --madam-early-stop-patience "$MADAM_EARLY_STOP_PATIENCE"
  )
  if [[ "$MADAM_INCLUDE_FULL_DOCUMENT_IN_DEBATE" -eq 1 ]]; then
    cmd+=(--madam-include-full-document-in-debate)
  fi
  if [[ "$EAJP_SHORT_CIRCUIT" -eq 1 ]]; then
    cmd+=(--eajp-short-circuit)
  else
    cmd+=(--no-eajp-short-circuit)
  fi
  if [[ "$JUDGE_ONLY_ON_CONTENTION" -eq 1 ]]; then
    cmd+=(--judge-only-on-contention)
  fi

  if [[ "$RESUME_RUNS" -eq 1 ]]; then
    cmd+=(--resume)
  else
    cmd+=(--no-resume)
  fi

  LLM_MIN_INTERVAL_MS="$throttle" LLM_PROVIDER_PROMPT_CACHE="$LLM_PROVIDER_PROMPT_CACHE" "${cmd[@]}"
}

if [[ "$RUN_EXPERIMENTS" -eq 1 ]]; then
  runs=(
    "ramdocs|$RAMDOCS_OUT|ramdocs"
    "tcb_ramdocs|$TCB_RAMDOCS_OUT|tcb"
    "confact|$CONFACT_SOURCE|confact"
    "tcb_confact|$TCB_CONFACT_OUT|tcb"
  )

  if [[ "$PARALLEL_DATASETS" -eq 1 ]]; then
    jobs=$MAX_DATASET_JOBS
    if (( jobs > ${#runs[@]} )); then
      jobs=${#runs[@]}
    fi
    if (( jobs < 1 )); then
      jobs=1
    fi

    workers_per_job=$(( TOTAL_WORKER_BUDGET / jobs ))
    if (( workers_per_job < 1 )); then
      workers_per_job=1
    fi

    throttle_ms=$THROTTLE_MS
    if (( throttle_ms == 0 && jobs > 1 )); then
      throttle_ms=$(( 120 * jobs ))
    fi

    echo "[run] Parallel dataset mode enabled: jobs=$jobs, workers_per_job=$workers_per_job, throttle_ms=$throttle_ms"

    pids=()
    failed=0
    for spec in "${runs[@]}"; do
      IFS='|' read -r name dataset_path benchmark <<< "$spec"
      run_one "$name" "$dataset_path" "$benchmark" "$workers_per_job" "$throttle_ms" &
      pids+=("$!")

      if (( ${#pids[@]} >= jobs )); then
        first_pid="${pids[0]}"
        if ! wait "$first_pid"; then
          failed=1
        fi
        pids=("${pids[@]:1}")
      fi
    done

    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done

    if (( failed != 0 )); then
      echo "[run] One or more dataset jobs failed." >&2
      exit 1
    fi
  else
    workers=$TOTAL_WORKER_BUDGET
    throttle_ms=$THROTTLE_MS
    echo "[run] Sequential dataset mode: workers=$workers, throttle_ms=$throttle_ms"
    for spec in "${runs[@]}"; do
      IFS='|' read -r name dataset_path benchmark <<< "$spec"
      run_one "$name" "$dataset_path" "$benchmark" "$workers" "$throttle_ms"
    done
  fi
fi

if [[ "$RUN_EXPERIMENTS" -eq 1 && "$VALIDATE_EAJP_SCHEMA" -eq 1 ]]; then
  echo "[validate] Running EAJP schema validator on $OUTPUT_ROOT"
  validate_cmd=(
    "$PYTHON_RUN" -m experiments.validate_eajp_schema
    --path "$OUTPUT_ROOT"
  )
  if [[ "$EAJP_SCHEMA_STRICT_ENUMS" -eq 1 ]]; then
    validate_cmd+=(--strict-enums)
  else
    validate_cmd+=(--no-strict-enums)
  fi
  "${validate_cmd[@]}"
fi

if [[ "$RUN_EXPERIMENTS" -eq 1 && "$COMPUTE_PROPOSAL_METRICS" -eq 1 ]]; then
  proposal_metrics_json="$OUTPUT_ROOT/proposal_metrics.json"
  echo "[metrics] Computing proposal metrics from details outputs under $OUTPUT_ROOT"
  "$PYTHON_RUN" -m experiments.compute_proposal_metrics \
    --path "$OUTPUT_ROOT" \
    --json-out "$proposal_metrics_json"
fi

echo "[summary] Aggregating benchmark metrics under $OUTPUT_ROOT"
OUTPUT_ROOT_OVERRIDE="$OUTPUT_ROOT" "$PYTHON_RUN" - <<'PY'
import json
from pathlib import Path

output_root = Path("results/full_run")

env_root = Path(__import__("os").environ.get("OUTPUT_ROOT_OVERRIDE", ""))
if str(env_root) and str(env_root) != ".":
    output_root = env_root

if not output_root.exists():
    print(f"No results directory found: {output_root}")
    raise SystemExit(0)

metric_files = sorted(output_root.rglob("*_metrics.json"))
if not metric_files:
    print(f"No metrics files found under {output_root}")
    raise SystemExit(0)

rows = []
skipped_non_dict = []
for path in metric_files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(data, dict):
        skipped_non_dict.append(str(path.relative_to(output_root)))
        continue
    rel = path.relative_to(output_root)
    parts = rel.parts
    benchmark = parts[0] if len(parts) > 0 else "unknown"
    suite = parts[1] if len(parts) > 2 else "suite_independent"
    rows.append(
        {
            "benchmark": benchmark,
            "suite": suite,
            "system": data.get("system", ""),
            "model": data.get("model", ""),
            "total": data.get("total", 0),
            "sar": data.get("substance_alignment_rate", 0.0),
            "tbr": data.get("tone_bias_rate", 0.0),
            "acc": data.get("accuracy", 0.0),
            "path": str(rel),
        }
    )

rows.sort(key=lambda r: (r["benchmark"], r["suite"], r["system"]))

if not rows:
    print(f"No dict-shaped metrics files found under {output_root}")
    if skipped_non_dict:
        print("Skipped non-dict metric files:")
        for item in skipped_non_dict:
            print(f"- {item}")
    raise SystemExit(0)

header = (
    f"{'benchmark':<16} {'suite':<24} {'system':<14} {'total':>7} "
    f"{'SAR':>8} {'TBR':>8} {'ACC':>8}"
)
print(header)
print("-" * len(header))
for r in rows:
    print(
        f"{r['benchmark']:<16} {r['suite']:<24} {r['system']:<14} {r['total']:>7} "
        f"{r['sar']:>8.4f} {r['tbr']:>8.4f} {r['acc']:>8.4f}"
    )

summary_path = output_root / "benchmark_summary.json"
summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nWrote summary: {summary_path}")
if skipped_non_dict:
    print("Skipped non-dict metric files:")
    for item in skipped_non_dict:
        print(f"- {item}")
PY

echo "Done."
