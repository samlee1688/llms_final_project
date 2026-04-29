#!/usr/bin/env python3
"""Rewrite every document in a dataset and store original + rewritten views.

This preprocessing stage is useful when you want to pay rewrite costs once,
persist the rewritten variants, and reuse them across many experiment runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import Document, QueryInstance, load_benchmark, save_dataset
from src.tcb_builder import ToneRewriter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rewrite all documents and persist original/neutral/assertive variants."
    )
    p.add_argument("--input", required=True, help="Source dataset JSON.")
    p.add_argument(
        "--benchmark",
        choices=["custom", "ramdocs", "confact", "tcb"],
        default="custom",
        help="Dataset family for loader behavior.",
    )
    p.add_argument("--output", required=True, help="Output JSON with rewrite variants.")
    p.add_argument(
        "--rewriter-provider",
        choices=["gemini", "local"],
        default="gemini",
        help="Rewrite provider. Gemini falls back to local if unavailable.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse already-rewritten instances from output file when available.",
    )
    p.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Optional cap for debugging (0 = all instances).",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def _rewrite_doc(doc: Document, rewriter: ToneRewriter) -> Document:
    original = doc.content
    neutral = rewriter.rewrite(original, "neutral")
    assertive = rewriter.rewrite(original, "assertive")
    metadata = {
        **dict(doc.metadata or {}),
        "rewrite_variants": {
            "original": original,
            "neutral": neutral,
            "assertive": assertive,
        },
        "rewrite_provider": rewriter.provider,
        "rewrite_model": rewriter.model,
        "rewrite_complete": True,
    }
    return Document(
        id=doc.id,
        content=original,
        source=doc.source,
        credibility=doc.credibility,
        supports_answer=doc.supports_answer,
        tone=doc.tone,
        metadata=metadata,
    )


def _rewrite_instance(instance: QueryInstance, rewriter: ToneRewriter) -> QueryInstance:
    docs = [_rewrite_doc(doc, rewriter) for doc in instance.documents]
    metadata = {
        **dict(instance.metadata or {}),
        "full_document_rewrites": True,
        "rewrite_provider": rewriter.provider,
        "rewrite_model": rewriter.model,
    }
    return QueryInstance(
        id=instance.id,
        question=instance.question,
        documents=docs,
        correct_answer=instance.correct_answer,
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    instances = load_benchmark(args.input, args.benchmark)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    logger.info("Loaded %d source instances from %s", len(instances), args.input)

    existing_by_id: Dict[str, QueryInstance] = {}
    if args.resume and os.path.exists(args.output):
        try:
            existing = load_benchmark(args.output, "custom")
            existing_by_id = {inst.id: inst for inst in existing}
            logger.info("Resume enabled: found %d existing rewritten instances", len(existing_by_id))
        except Exception as exc:
            logger.warning("Failed to read existing output for resume: %s", exc)

    rewriter = ToneRewriter(provider=args.rewriter_provider)

    built: List[QueryInstance] = []
    rewritten = 0
    reused = 0
    for i, inst in enumerate(instances, start=1):
        if inst.id in existing_by_id:
            built.append(existing_by_id[inst.id])
            reused += 1
        else:
            built.append(_rewrite_instance(inst, rewriter))
            rewritten += 1
        if i % 10 == 0 or i == len(instances):
            logger.info(
                "Progress %d/%d (rewritten=%d, reused=%d)",
                i,
                len(instances),
                rewritten,
                reused,
            )

    save_dataset(built, args.output)
    logger.info(
        "Saved rewritten dataset to %s (total=%d, rewritten=%d, reused=%d)",
        args.output,
        len(built),
        rewritten,
        reused,
    )


if __name__ == "__main__":
    main()
