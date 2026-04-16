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

from src.data_loader import load_benchmark
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
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    instances = load_benchmark(args.input, args.benchmark)
    builder = ToneControlledBenchmarkBuilder(
        rewriter=ToneRewriter(provider=args.rewriter_provider)
    )
    builder.build_and_save(instances, args.output, source_benchmark=args.benchmark)


if __name__ == "__main__":
    main()