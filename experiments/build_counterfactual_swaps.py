#!/usr/bin/env python3
"""Build counterfactual tone-swap dataset for mechanism analysis.

Identifies trials where --filter-system produced an incorrect answer on a TCB
dataset (i.e., the assertive incorrect doc won), then creates a swapped dataset:
  - The correct doc (originally neutral)   → rewritten assertive
  - The incorrect doc (originally assertive) → rewritten neutral
  - Credibility labels are preserved unchanged

If the judge flips to the now-assertive correct doc after swapping, that
proves tone was the causal mechanism behind the original failure.

Typical workflow
----------------
1. Run build_counterfactual_swaps.py to produce swap_dataset.json
2. Run run_experiment.py on swap_dataset.json to produce swapped results
3. Run analyze_counterfactual_swaps.py to compute flip rates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import Document, QueryInstance, load_dataset, save_dataset
from src.tcb_builder import ToneRewriter

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Trial loading
# -----------------------------------------------------------------------

def _load_all_details(results_dir: Path) -> list[dict]:
    """Recursively collect all *_details.json trial records."""
    trials: list[dict] = []
    for path in sorted(results_dir.rglob("*_details.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            trials.extend(data)
    return trials


def _collect_failures(trials: list[dict], filter_system: str) -> list[str]:
    """Return unique query_ids where system=filter_system, classification=incorrect,
    and metadata.tone_controlled=True."""
    failures: list[str] = []
    seen: set[str] = set()
    for t in trials:
        if t.get("system") != filter_system:
            continue
        if t.get("classification") != "incorrect":
            continue
        if not (t.get("metadata") or {}).get("tone_controlled", False):
            continue
        qid = t.get("query_id", "")
        if qid and qid not in seen:
            seen.add(qid)
            failures.append(qid)
    return failures


# -----------------------------------------------------------------------
# Document helpers
# -----------------------------------------------------------------------

def _get_correct_doc(instance: QueryInstance) -> Document | None:
    neutral = [d for d in instance.documents if d.tone == "neutral"]
    if neutral:
        return neutral[0]
    correct = instance.correct_documents
    return correct[0] if correct else None


def _get_incorrect_doc(instance: QueryInstance) -> Document | None:
    assertive = [d for d in instance.documents if d.tone == "assertive"]
    if assertive:
        return assertive[0]
    incorrect = instance.incorrect_documents
    return incorrect[0] if incorrect else None


# -----------------------------------------------------------------------
# Swap builder
# -----------------------------------------------------------------------

def _swap_one(
    query_id: str,
    instance_map: dict[str, QueryInstance],
    rewriter: ToneRewriter,
    skip_meaning_check: bool,
) -> tuple[str, QueryInstance | None, str]:
    """Build one swapped instance. Returns (query_id, instance_or_None, skip_reason)."""
    instance = instance_map.get(query_id)
    if instance is None:
        return query_id, None, "missing_instance"

    correct_doc = _get_correct_doc(instance)
    incorrect_doc = _get_incorrect_doc(instance)
    if correct_doc is None or incorrect_doc is None:
        return query_id, None, "missing_docs"

    assertive_result = rewriter.rewrite_with_metadata(correct_doc.content, "assertive")
    if not skip_meaning_check and not assertive_result.get("meaning_preserved", False):
        logger.warning(
            "Instance %s assertive rewrite failed meaning check (%s); skipping.",
            query_id, assertive_result.get("meaning_failure_reason", ""),
        )
        return query_id, None, "rewrite_failed"

    neutral_result = rewriter.rewrite_with_metadata(incorrect_doc.content, "neutral")
    if not skip_meaning_check and not neutral_result.get("meaning_preserved", False):
        logger.warning(
            "Instance %s neutral rewrite failed meaning check (%s); skipping.",
            query_id, neutral_result.get("meaning_failure_reason", ""),
        )
        return query_id, None, "rewrite_failed"

    base_id = query_id.removeprefix("tcb_")

    new_correct_doc = Document(
        id=f"{base_id}_correct_assertive",
        content=assertive_result["rewritten_content"],
        source=correct_doc.source,
        credibility=correct_doc.credibility,
        supports_answer=correct_doc.supports_answer,
        tone="assertive",
        metadata={
            **correct_doc.metadata,
            "tone_controlled": True,
            "original_doc_id": correct_doc.id,
            "original_tone": "neutral",
            "counterfactual_swap": True,
            "rewrite_verification_source": assertive_result.get("verification_source", ""),
            "rewrite_used_fallback": bool(assertive_result.get("used_fallback", False)),
            "rewrite_meaning_preserved": bool(assertive_result.get("meaning_preserved", False)),
        },
    )

    new_incorrect_doc = Document(
        id=f"{base_id}_incorrect_neutral",
        content=neutral_result["rewritten_content"],
        source=incorrect_doc.source,
        credibility=incorrect_doc.credibility,
        supports_answer=incorrect_doc.supports_answer,
        tone="neutral",
        metadata={
            **incorrect_doc.metadata,
            "tone_controlled": True,
            "original_doc_id": incorrect_doc.id,
            "original_tone": "assertive",
            "counterfactual_swap": True,
            "rewrite_verification_source": neutral_result.get("verification_source", ""),
            "rewrite_used_fallback": bool(neutral_result.get("used_fallback", False)),
            "rewrite_meaning_preserved": bool(neutral_result.get("meaning_preserved", False)),
        },
    )

    swap_instance = QueryInstance(
        id=f"swap_{query_id}",
        question=instance.question,
        documents=[new_correct_doc, new_incorrect_doc],
        correct_answer=instance.correct_answer,
        metadata={
            **instance.metadata,
            "tone_controlled": True,
            "counterfactual_swap": True,
            "original_instance_id": query_id,
            "swap_description": "correct_doc_now_assertive_incorrect_doc_now_neutral",
        },
    )
    return query_id, swap_instance, ""


def build_swaps(
    failures: list[str],
    instance_map: dict[str, QueryInstance],
    rewriter: ToneRewriter,
    skip_meaning_check: bool = False,
    max_workers: int = 1,
) -> tuple[list[QueryInstance], dict[str, Any]]:
    """Build tone-swapped instances for each failure query_id.

    Returns (swapped_instances, stats_dict).  Pass max_workers > 1 to
    parallelize Gemini rewrite API calls (network-bound).
    """
    stats: dict[str, Any] = {
        "total_failures": len(failures),
        "built": 0,
        "skipped_missing_instance": 0,
        "skipped_missing_docs": 0,
        "skipped_rewrite_failed": 0,
    }

    def _process(query_id: str):
        return _swap_one(query_id, instance_map, rewriter, skip_meaning_check)

    results: list[tuple[str, QueryInstance | None, str]] = []

    if max_workers <= 1:
        for qid in failures:
            results.append(_process(qid))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, qid): qid for qid in failures}
            for fut in as_completed(futures):
                results.append(fut.result())

    # Reassemble in original order
    result_map = {qid: (inst, reason) for qid, inst, reason in results}
    swapped: list[QueryInstance] = []
    for qid in failures:
        inst, reason = result_map.get(qid, (None, "missing_instance"))
        if inst is not None:
            swapped.append(inst)
            stats["built"] += 1
        elif reason == "missing_instance":
            stats["skipped_missing_instance"] += 1
        elif reason == "missing_docs":
            stats["skipped_missing_docs"] += 1
        else:
            stats["skipped_rewrite_failed"] += 1

    return swapped, stats


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build counterfactual tone-swap dataset from TCB failure trials.\n\n"
            "Finds trials where --filter-system produced an incorrect answer, "
            "swaps tones (correct neutral→assertive, incorrect assertive→neutral), "
            "and saves a new dataset for re-running experiments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tcb-dataset",
        required=True,
        nargs="+",
        metavar="PATH",
        help="Path(s) to TCB dataset JSON file(s).",
    )
    p.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help="Results directory to search for *_details.json files (recursive).",
    )
    p.add_argument(
        "--filter-system",
        default="madam_rag",
        metavar="SYSTEM",
        help="Extract failures for this system (default: madam_rag).",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for swapped dataset JSON.",
    )
    p.add_argument(
        "--skip-meaning-check",
        action="store_true",
        help="Accept rewrites regardless of meaning-preservation check.",
    )
    p.add_argument(
        "--rewriter-provider",
        default="gemini",
        choices=["gemini", "local"],
        help="Rewrite provider (default: gemini; auto-falls back to local heuristic).",
    )
    p.add_argument(
        "--json-stats",
        default=None,
        metavar="PATH",
        help="Optional path to write build statistics as JSON.",
    )
    p.add_argument(
        "--rewrite-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for Gemini rewrite API calls (default: 1 serial). Use 4-8 to saturate Gemini free-tier (~60 RPM).",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Load TCB dataset(s)
    all_instances: list[QueryInstance] = []
    for path in args.tcb_dataset:
        all_instances.extend(load_dataset(path))
    instance_map = {inst.id: inst for inst in all_instances}
    print(f"Loaded {len(all_instances)} instances from {len(args.tcb_dataset)} TCB dataset(s).")

    # Collect failures from results
    results_dir = Path(args.results_dir)
    trials = _load_all_details(results_dir)
    print(f"Loaded {len(trials)} total trials from {results_dir}.")

    failures = _collect_failures(trials, args.filter_system)
    print(f"Found {len(failures)} unique '{args.filter_system}' failures on tone-controlled instances.")

    if not failures:
        print("No failures found — nothing to swap.")
        return

    # Build rewriter
    if args.rewriter_provider == "local":
        rewriter = ToneRewriter(provider="local", api_key="")
    else:
        rewriter = ToneRewriter(provider="gemini")

    # Build swapped instances
    swapped, stats = build_swaps(
        failures,
        instance_map,
        rewriter,
        skip_meaning_check=args.skip_meaning_check,
        max_workers=args.rewrite_workers,
    )

    print("\nSwap build statistics:")
    print(f"  Total failures:           {stats['total_failures']}")
    print(f"  Built successfully:       {stats['built']}")
    print(f"  Skipped (no instance):    {stats['skipped_missing_instance']}")
    print(f"  Skipped (no docs):        {stats['skipped_missing_docs']}")
    print(f"  Skipped (rewrite failed): {stats['skipped_rewrite_failed']}")

    if not swapped:
        print("No swapped instances built. Exiting.")
        return

    save_dataset(swapped, args.output)
    print(f"\nSaved {len(swapped)} swapped instances → {args.output}")

    print(
        f"\nNext step — re-run experiments on the swapped dataset:\n"
        f"  python experiments/run_experiment.py \\\n"
        f"    --dataset {args.output} --benchmark tcb \\\n"
        f"    --output-dir <swapped_results_dir>\n\n"
        f"Then analyze flip rates:\n"
        f"  python experiments/analyze_counterfactual_swaps.py \\\n"
        f"    --original-results {results_dir} \\\n"
        f"    --swapped-results <swapped_results_dir>"
    )

    if args.json_stats:
        stats_out: dict[str, Any] = {
            **stats,
            "output": args.output,
            "filter_system": args.filter_system,
            "swapped_instance_ids": [i.id for i in swapped],
        }
        Path(args.json_stats).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_stats, "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2)
        print(f"Stats written → {args.json_stats}")


if __name__ == "__main__":
    main()
