#!/usr/bin/env python3
"""Build a matched source/TCB sample of target size.

This utility iterates source instances in deterministic random order,
keeps only instances whose TCB rewrite succeeds, and stops once the target
size is reached. It writes:

- source subset (matched originals)
- TCB subset (same instance count, aligned by constructed_from)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import QueryInstance, load_benchmark, save_dataset
from src.tcb_builder import ToneControlledBenchmarkBuilder, ToneRewriter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build matched source/TCB sample with target size.")
    p.add_argument("--input", required=True, help="Source benchmark JSON file.")
    p.add_argument(
        "--benchmark",
        choices=["custom", "ramdocs", "confact"],
        default="custom",
        help="Source benchmark family used for filtering.",
    )
    p.add_argument("--output-source", required=True, help="Output matched source JSON file.")
    p.add_argument("--output-tcb", required=True, help="Output matched TCB JSON file.")
    p.add_argument("--target-size", type=int, required=True, help="Required matched sample size.")
    p.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling order.")
    p.add_argument(
        "--rewriter-provider",
        default="gemini",
        choices=["gemini", "local"],
        help="Use Gemini if available, otherwise a deterministic local fallback.",
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
        help="Abort immediately if a single instance raises an exception.",
    )
    p.add_argument(
        "--require-full-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if fewer than target-size matched samples can be built (default: true).",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )
    for noisy_logger in ("google", "google.genai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    if args.target_size < 1:
        raise SystemExit("--target-size must be >= 1")

    instances = load_benchmark(args.input, args.benchmark)
    total_instances = len(instances)

    rng = random.Random(args.sample_seed)
    ordered = list(instances)
    rng.shuffle(ordered)

    builder = ToneControlledBenchmarkBuilder(
        rewriter=ToneRewriter(
            provider=args.rewriter_provider,
            max_retries=max(1, args.max_rewrite_retries),
            backoff_base_seconds=max(0.0, args.rewrite_backoff_base_seconds),
        )
    )

    matched_source: list[QueryInstance] = []
    matched_tcb: list[QueryInstance] = []
    attempted = 0
    failed = 0

    for inst in ordered:
        if len(matched_tcb) >= args.target_size:
            break
        attempted += 1
        try:
            tcb_inst = builder.build_instance(inst, source_benchmark=args.benchmark)
            if tcb_inst is None:
                failed += 1
                reason = getattr(builder, "last_failure_reason", "unknown") or "unknown"
                logger.warning(
                    "Document %d/%d with id %s failed meaning preservation check: due to %s",
                    attempted,
                    total_instances,
                    inst.id,
                    reason,
                )
                continue
            matched_source.append(inst)
            matched_tcb.append(tcb_inst)
        except Exception as exc:
            failed += 1
            logger.warning("Skipping instance %s due to build error: %s", inst.id, exc)
            if args.fail_on_instance_error:
                raise

    if len(matched_tcb) < args.target_size:
        msg = (
            f"Could not reach target-size={args.target_size}; "
            f"built={len(matched_tcb)} after attempted={attempted}, failed={failed}."
        )
        if args.require_full_target:
            raise SystemExit(msg)
        logger.warning(msg)

    save_dataset(matched_source, args.output_source)
    save_dataset(matched_tcb, args.output_tcb)
    logger.warning(
        "Saved matched datasets: source=%s (%d), tcb=%s (%d), attempted=%d, failed=%d",
        args.output_source,
        len(matched_source),
        args.output_tcb,
        len(matched_tcb),
        attempted,
        failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
