#!/usr/bin/env python3
"""Populate supports_answer on humc.json documents using ModC majority_vote.

Background
----------
The raw CONFACT HumC dataset has no per-evidence support/refute annotation.
ModC is a sibling CONFACT split (611 records) that carries per-evidence
LLM-judged verdicts in a ``majority_vote`` field.  HumC (287 records) is a
strict subset of ModC — every HumC record_id appears in ModC — so ModC
majority_vote can be used as a ground-truth per-evidence label for HumC.

Mapping
-------
  majority_vote                    →  supports_answer
  ──────────────────────────────────────────────────
  "support"                        →  "yes"
  "reject"                         →  "no"
  "not enough evidence"            →  "neither"
  "error occurred in generation"   →  "neither"
  (not found in ModC)              →  "" (or value of --unknown-as)

ID alignment
------------
convert_confact.py creates project-schema Document IDs as::

    humc_{record_id}_{evidence_id}

where ``evidence_id`` comes verbatim from the raw evidence item and is
itself formatted as ``{record_id}_{seq_num}`` (e.g. "1_3").  This means
the record_id appears twice in the full document ID (e.g. ``humc_1_1_3``).

This script looks up the ModC annotation via the evidence_id stored in
``doc["metadata"]["evidence_id"]``, which is the unambiguous key that
convert_confact.py saves directly from the raw evidence payload.

Usage
-----
  # Default paths (humc.json → humc_processed.json, same directory):
  python experiments/confact_populate_supports_answers.py

  # Explicit paths:
  python experiments/confact_populate_supports_answers.py \\
      --humc  data/full/confact/humc.json \\
      --modc  data/full/confact/modC.json \\
      --output data/full/confact/humc_processed.json

  # Preview without writing:
  python experiments/confact_populate_supports_answers.py --dry-run

  # Show 10 before/after samples:
  python experiments/confact_populate_supports_answers.py --sample 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HUMC  = "data/full/confact/humc.json"
_DEFAULT_MODC  = "data/full/confact/modC.json"
_DEFAULT_OUT   = None  # derived from --humc stem: <stem>_processed.json

# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

VOTE_MAP: dict[str, str] = {
    "support":                      "yes",
    "reject":                       "no",
    "not enough evidence":          "neither",
    "error occurred in generation": "neither",
}


# ---------------------------------------------------------------------------
# ModC lookup builder
# ---------------------------------------------------------------------------

def build_modc_lookup(modc_path: str) -> dict[int, dict[str, str]]:
    """Return {record_id (int): {evidence_id (str): majority_vote (str)}}.

    ModC evidence_id values are formatted as "{record_id}_{seq_num}",
    e.g. "1_3".  The full dict key is the evidence_id string as-is.
    """
    logger.info("Loading ModC: %s", modc_path)
    with open(modc_path, encoding="utf-8") as fh:
        modc: list[dict[str, Any]] = json.load(fh)

    lookup: dict[int, dict[str, str]] = {}
    for rec in modc:
        rid = int(rec["id"])
        lookup[rid] = {}
        for ev in rec.get("evidence_url", []):
            ev_id    = ev.get("evidence_id", "")
            mv       = (ev.get("majority_vote") or "").strip()
            if ev_id:
                lookup[rid][ev_id] = mv

    n_records  = len(lookup)
    n_evidence = sum(len(v) for v in lookup.values())
    logger.info("ModC: %d records, %d evidence items", n_records, n_evidence)
    return lookup


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def enrich(
    humc_data:   list[dict[str, Any]],
    modc_lookup: dict[int, dict[str, str]],
    unknown_as:  str = "",
    overwrite:   bool = True,
) -> tuple[list[dict[str, Any]], Counter]:
    """Populate supports_answer on every document in *humc_data* in-place.

    Parameters
    ----------
    humc_data:
        List of QueryInstance dicts (project schema) loaded from humc.json.
    modc_lookup:
        {record_id → {evidence_id → majority_vote}} from build_modc_lookup().
    unknown_as:
        Value to write when the evidence is absent from ModC (default "").
    overwrite:
        When False, skip documents that already have a non-empty
        supports_answer (useful for incremental re-runs).

    Returns
    -------
    (enriched_data, stats_counter)
        stats_counter keys: "yes", "no", "neither", "unknown",
        "skipped_existing", "ev_id_missing".
    """
    stats: Counter = Counter()

    for qi in humc_data:
        # Record ID is the integer embedded in the instance id "humc_{n}".
        try:
            record_id = int(qi["id"].split("_")[1])
        except (IndexError, ValueError):
            logger.warning("Cannot parse record_id from instance id %r; skipping.", qi["id"])
            continue

        ev_by_id = modc_lookup.get(record_id, {})

        for doc in qi.get("documents", []):
            existing = (doc.get("supports_answer") or "").strip()

            if existing and not overwrite:
                stats["skipped_existing"] += 1
                continue

            # Primary key: the evidence_id stored in metadata by convert_confact.py.
            # This is the full ModC evidence_id string (e.g. "1_3") — the unambiguous
            # lookup key that avoids the doubled record_id in the document ID.
            ev_id = (doc.get("metadata", {}).get("evidence_id") or "").strip()

            if not ev_id:
                # Fallback: derive ev_id from the document ID.
                # Document IDs have the form  humc_{record_id}_{evidence_id}
                # e.g. humc_1_1_3  →  evidence_id = "1_3"
                # The evidence_id itself starts with the record_id, producing the
                # doubled-record_id appearance in the full document ID.
                # Stripping the "humc_{record_id}_" prefix recovers the evidence_id.
                prefix = f"humc_{record_id}_"
                doc_id = doc.get("id", "")
                if doc_id.startswith(prefix):
                    ev_id = doc_id[len(prefix):]
                else:
                    logger.debug(
                        "Cannot derive ev_id from doc id %r (prefix %r not found)",
                        doc_id, prefix,
                    )
                    stats["ev_id_missing"] += 1
                    doc["supports_answer"] = unknown_as
                    continue

            majority_vote = ev_by_id.get(ev_id, "")

            if not majority_vote:
                logger.debug(
                    "ev_id %r for record %d not found in ModC lookup",
                    ev_id, record_id,
                )
                doc["supports_answer"] = unknown_as
                stats["unknown"] += 1
            else:
                mapped = VOTE_MAP.get(majority_vote)
                if mapped is None:
                    logger.warning(
                        "Unexpected majority_vote %r for ev_id %r (record %d); "
                        "treating as unknown.",
                        majority_vote, ev_id, record_id,
                    )
                    doc["supports_answer"] = unknown_as
                    stats["unknown"] += 1
                else:
                    doc["supports_answer"] = mapped
                    stats[mapped] += 1

    return humc_data, stats


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _default_output(humc_path: str) -> str:
    base, ext = os.path.splitext(humc_path)
    return base + "_processed" + ext


def _print_stats(stats: Counter, total: int) -> None:
    print()
    print("Enrichment results")
    print("  {:<18s}  {:>6s}  {:>7s}".format("supports_answer", "count", "pct"))
    print("  " + "-" * 36)
    for key in ("yes", "no", "neither", "unknown", "ev_id_missing", "skipped_existing"):
        n = stats[key]
        if n or key in ("yes", "no", "neither"):
            pct = 100.0 * n / total if total else 0.0
            print("  {:<18s}  {:>6d}  {:>6.1f}%".format(key, n, pct))
    print("  " + "-" * 36)
    print("  {:<18s}  {:>6d}".format("total documents", total))
    print()


def _print_samples(
    samples: list[tuple[str, str, str, str]],
    n: int,
) -> None:
    """Print N random (doc_id, ev_id, majority_vote, supports_answer) samples."""
    chosen = random.sample(samples, min(n, len(samples)))
    print(f"\n{'─'*60}")
    print(f"Sample ({len(chosen)} docs shown)")
    print(f"{'─'*60}")
    for doc_id, ev_id, mv, sa in chosen:
        print(f"  doc          : {doc_id}")
        print(f"  ev_id        : {ev_id}")
        print(f"  majority_vote: {mv!r}")
        print(f"  supports_ans : {sa!r}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    out_path = args.output or _default_output(args.humc)

    # ── Load ─────────────────────────────────────────────────────────────────
    modc_lookup = build_modc_lookup(args.modc)

    logger.info("Loading humc: %s", args.humc)
    with open(args.humc, encoding="utf-8") as fh:
        humc_data: list[dict[str, Any]] = json.load(fh)
    logger.info("humc: %d instances", len(humc_data))

    total_docs = sum(len(qi.get("documents", [])) for qi in humc_data)
    logger.info("Total documents to enrich: %d", total_docs)

    # ── Enrich ───────────────────────────────────────────────────────────────
    # Collect sample data before enrichment for the --sample display.
    samples: list[tuple[str, str, str, str]] = []

    if args.sample > 0:
        for qi in humc_data:
            try:
                record_id = int(qi["id"].split("_")[1])
            except (IndexError, ValueError):
                continue
            ev_by_id = modc_lookup.get(record_id, {})
            for doc in qi.get("documents", []):
                ev_id = (doc.get("metadata", {}).get("evidence_id") or "").strip()
                if not ev_id:
                    prefix = f"humc_{record_id}_"
                    doc_id = doc.get("id", "")
                    ev_id = doc_id[len(prefix):] if doc_id.startswith(prefix) else ""
                mv = ev_by_id.get(ev_id, "")
                if mv:
                    samples.append((doc.get("id", ""), ev_id, mv, ""))

    _, stats = enrich(
        humc_data,
        modc_lookup,
        unknown_as=args.unknown_as,
        overwrite=not args.no_overwrite,
    )

    # Back-fill the resolved supports_answer into the sample list.
    if samples:
        sa_map = {
            doc["id"]: doc.get("supports_answer", "")
            for qi in humc_data
            for doc in qi.get("documents", [])
        }
        samples = [(d, e, m, sa_map.get(d, "")) for d, e, m, _ in samples]

    # ── Print stats ──────────────────────────────────────────────────────────
    _print_stats(stats, total_docs)

    if args.sample > 0 and samples:
        _print_samples(samples, args.sample)

    # ── Write ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry-run — no file written.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(humc_data, fh, ensure_ascii=False, indent=2)
    logger.info("Written → %s", out_path)

    n_unknown = stats["unknown"] + stats["ev_id_missing"]
    if n_unknown:
        logger.warning(
            "%d document(s) could not be matched in ModC — "
            "supports_answer set to %r.  Run with --verbose to see details.",
            n_unknown, args.unknown_as,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Populate supports_answer in a project-schema humc.json "
            "using per-evidence majority_vote from modC.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--humc", "-i",
        default=_DEFAULT_HUMC,
        metavar="PATH",
        help="Input humc.json in project schema (output of convert_confact.py).",
    )
    p.add_argument(
        "--modc",
        default=_DEFAULT_MODC,
        metavar="PATH",
        help="modC.json in raw CONFACT format (contains majority_vote per evidence).",
    )
    p.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUT,
        metavar="PATH",
        help=(
            "Output path.  Defaults to <humc_stem>_processed.json "
            "in the same directory as --humc."
        ),
    )
    p.add_argument(
        "--unknown-as",
        default="",
        metavar="VALUE",
        help=(
            "Value to write into supports_answer when the evidence_id is "
            "absent from ModC.  Use 'neither' to treat unmatched evidence "
            "as neutral."
        ),
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help=(
            "Skip documents that already have a non-empty supports_answer "
            "(useful for incremental re-runs on partially enriched files)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and samples but do not write the output file.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Print N randomly sampled (doc_id, ev_id, majority_vote → supports_answer) rows.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging (shows per-document lookup details).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
