#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

./preprocess_rewrites.sh \
  --rewriter-provider gemini \
  --dataset-root data/sample200_gemini_preprocess \
  --pull-mode sample \
  --sample-size 200 \
  --sample-seed 42 \
  --ambigdocs-splits validation \
  --faitheval-splits test \
  --tcb-resume 1 \
  --tcb-checkpoint-every 100 \
  --tcb-max-rewrite-retries 8 \
  --tcb-rewrite-backoff-base-seconds 2.0 \
  --tcb-fail-on-instance-error 0
