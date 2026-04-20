#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

./preprocess_rewrites.sh \
  --rewriter-provider gemini \
  --dataset-root data/full \
  --ambigdocs-splits train,validation,test \
  --faitheval-splits train,validation,test \
  --pull-mode full \
  --tcb-resume 1 \
  --tcb-checkpoint-every 100 \
  --tcb-max-rewrite-retries 6 \
  --tcb-rewrite-backoff-base-seconds 1.5 \
  --tcb-fail-on-instance-error 0
