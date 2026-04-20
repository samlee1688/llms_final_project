#!/usr/bin/env python3
"""Construct a Tone-Controlled Benchmark (TCB) dataset.

The script loads an existing benchmark-style JSON file, selects instances
with at least one correct and one incorrect document, rewrites the chosen
documents into neutral/assertive registers, and saves the resulting TCB
dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import QueryInstance, load_benchmark, save_dataset
from src.tcb_builder import ToneControlledBenchmarkBuilder, ToneRewriter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a tone-controlled benchmark.")
    p.add_argument("--input", required=True, help="Source benchmark JSON file.")
    p.add_argument(
        "--benchmark",
        choices=["custom", "ambigdocs", "faitheval"],
        default="custom",
        help="Source benchmark family used for filtering.",
    )
    p.add_argument("--output", required=True, help="Output TCB JSON file.")
    p.add_argument(
        "--rewriter-provider",
        default="gemini",
        choices=["gemini", "local"],
        help="Use Gemini if available, otherwise a deterministic local fallback.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output file when available.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save intermediate output every N newly built instances.",
    )
    p.add_argument(
        "--max-rewrite-retries",
        type=int,
        default=6,
        help="Max retries for each provider rewrite call before local fallback.",
    )
    p.add_argument(
        "--rewrite-backoff-base-seconds",
        type=float,
        default=1.5,
        help="Exponential backoff base seconds for provider rewrite retries.",
    )
    p.add_argument(
        "--fail-on-instance-error",
        action="store_true",
        help="Abort immediately if a single instance fails to build.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    return p.parse_args()


def _source_instance_id(instance: QueryInstance) -> str:
    constructed_from = str(instance.metadata.get("constructed_from", "")).strip()
    if constructed_from:
        return constructed_from
    if instance.id.startswith("tcb_"):
        return instance.id[len("tcb_") :]
    return instance.id


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )
    for noisy_logger in ("google", "google.genai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    instances = load_benchmark(args.input, args.benchmark)
    total_instances = len(instances)

    checkpoint_every = max(1, args.checkpoint_every)

    built: list[QueryInstance] = []
    existing_by_source_id: dict[str, QueryInstance] = {}
    if args.resume and os.path.exists(args.output):
        try:
            existing = load_benchmark(args.output, "tcb")
            existing_by_source_id = {_source_instance_id(item): item for item in existing}
            built = list(existing_by_source_id.values())
            logger.info("Resume enabled: loaded %d existing TCB instances", len(built))
        except Exception as exc:
            logger.warning("Failed loading existing output for resume (%s). Rebuilding.", exc)

    builder = ToneControlledBenchmarkBuilder(
        rewriter=ToneRewriter(
            provider=args.rewriter_provider,
            max_retries=max(1, args.max_rewrite_retries),
            backoff_base_seconds=max(0.0, args.rewrite_backoff_base_seconds),
        )
    )

    created = 0
    skipped_existing = 0
    failed = 0
    for idx, instance in enumerate(instances, start=1):
        if instance.id in existing_by_source_id:
            skipped_existing += 1
        else:
            try:
                tcb_instance = builder.build_instance(instance, source_benchmark=args.benchmark)
                if tcb_instance is not None:
                    built.append(tcb_instance)
                    created += 1
                else:
                    failed += 1
                    reason = getattr(builder, "last_failure_reason", "unknown") or "unknown"
                    logger.warning(
                        "Document %d/%d with id %s failed meaning preservation check: due to %s",
                        idx,
                        total_instances,
                        instance.id,
                        reason,
                    )
            except Exception as exc:
                failed += 1
                logger.warning("Skipping instance %s due to build error: %s", instance.id, exc)
                if args.fail_on_instance_error:
                    raise

        if created > 0 and created % checkpoint_every == 0:
            save_dataset(built, args.output)
            logger.warning(
                "Checkpoint saved at %s (processed=%d/%d, created=%d, resumed=%d, failed=%d)",
                args.output,
                idx,
                total_instances,
                created,
                skipped_existing,
                failed,
            )

    save_dataset(built, args.output)
    logger.warning(
        "Saved %d TCB instances to %s (created=%d, resumed=%d, failed=%d)",
        len(built),
        args.output,
        created,
        skipped_existing,
        failed,
    )


if __name__ == "__main__":
    main()