#!/usr/bin/env python3
"""Filter a source dataset to the subset that has TCB counterparts.

Use this to build apples-to-apples matched sets when TCB creation skips
instances (for example due to rewrite validation failures).
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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter source dataset to TCB-overlap IDs.")
    p.add_argument("--source", required=True, help="Path to original/source dataset JSON.")
    p.add_argument("--tcb", required=True, help="Path to TCB dataset JSON.")
    p.add_argument("--output", required=True, help="Output path for filtered source JSON.")
    p.add_argument(
        "--source-benchmark",
        choices=["custom", "ramdocs", "confact", "tcb"],
        default="custom",
        help="Loader family for --source.",
    )
    p.add_argument(
        "--tcb-benchmark",
        choices=["custom", "ramdocs", "confact", "tcb"],
        default="tcb",
        help="Loader family for --tcb.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def _source_id_from_tcb_instance(instance: QueryInstance) -> str:
    constructed_from = str(instance.metadata.get("constructed_from", "")).strip()
    if constructed_from:
        return constructed_from
    if instance.id.startswith("tcb_"):
        return instance.id[len("tcb_") :]
    return instance.id


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    source = load_benchmark(args.source, args.source_benchmark)
    tcb = load_benchmark(args.tcb, args.tcb_benchmark)

    keep_ids = {_source_id_from_tcb_instance(inst) for inst in tcb}
    filtered = [inst for inst in source if inst.id in keep_ids]

    missing = sorted(i for i in keep_ids if i not in {s.id for s in source})
    if missing:
        logger.warning(
            "%d IDs found in TCB set were not present in source dataset (showing up to 10): %s",
            len(missing),
            ", ".join(missing[:10]),
        )

    save_dataset(filtered, args.output)
    logger.info(
        "Filtered source dataset: source=%d, tcb=%d, overlap=%d, output=%s",
        len(source),
        len(tcb),
        len(filtered),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
