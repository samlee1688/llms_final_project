#!/usr/bin/env python3
"""Analyze counterfactual tone-swap experiment results to measure causal flip rate.

After building a swapped dataset with build_counterfactual_swaps.py and
re-running experiments on it, this script:
  1. Loads original experiment results and identifies failures for --filter-system
  2. Loads swapped experiment results and finds matching swapped trials
  3. Computes per-system flip rate: (originally-incorrect → swapped-correct) / total
  4. Prints a causality verdict

A high flip rate (>70%) means: when the correct doc was made assertive (and the
incorrect doc neutral), the judge switched to the correct answer → tone was the
causal driver of the original bias.

Typical usage
-------------
  python experiments/analyze_counterfactual_swaps.py \\
    --original-results results/my_run \\
    --swapped-results results/my_swapped_run \\
    --filter-system madam_rag \\
    --json-out results/my_swapped_run/counterfactual_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_THRESHOLD_STRONG = 0.70    # >70% flip → strong causal evidence
_THRESHOLD_MODERATE = 0.40  # 40–70%  → moderate evidence


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


# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

def _collect_original_failures(trials: list[dict], filter_system: str) -> set[str]:
    """Return query_ids where system=filter_system, classification=incorrect,
    and metadata.tone_controlled=True."""
    failures: set[str] = set()
    for t in trials:
        if t.get("system") != filter_system:
            continue
        if t.get("classification") != "incorrect":
            continue
        if not (t.get("metadata") or {}).get("tone_controlled", False):
            continue
        qid = t.get("query_id", "")
        if qid:
            failures.add(qid)
    return failures


def _analyze_swapped(
    swapped_trials: list[dict],
    original_failures: set[str],
) -> dict[str, dict[str, int]]:
    """Per-system flip statistics on swapped trials.

    A trial is counted only if its query_id matches swap_{original_failure_id}.
    Returns {system: {total, flipped_correct, stayed_incorrect, unclear}}.
    """
    per_system: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "flipped_correct": 0, "stayed_incorrect": 0, "unclear": 0}
    )

    for t in swapped_trials:
        qid = t.get("query_id", "")
        if not qid.startswith("swap_"):
            continue
        original_id = qid.removeprefix("swap_")
        if original_id not in original_failures:
            continue

        system = t.get("system", "unknown")
        classification = t.get("classification", "")
        s = per_system[system]
        s["total"] += 1
        if classification == "correct":
            s["flipped_correct"] += 1
        elif classification == "incorrect":
            s["stayed_incorrect"] += 1
        else:
            s["unclear"] += 1

    return dict(per_system)


def _flip_rate(stats: dict[str, int]) -> float:
    total = stats.get("total", 0)
    return stats["flipped_correct"] / total if total > 0 else 0.0


def _verdict(rate: float) -> str:
    if rate >= _THRESHOLD_STRONG:
        return "STRONG causal evidence — tone is the primary driver of bias"
    if rate >= _THRESHOLD_MODERATE:
        return "MODERATE causal evidence — tone contributes to bias"
    return "WEAK causal evidence — tone may not be the primary driver"


# -----------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------

def _print_report(
    original_failures: set[str],
    per_system: dict[str, dict[str, int]],
    filter_system: str,
) -> None:
    print("=" * 72)
    print("COUNTERFACTUAL TONE-SWAP ANALYSIS")
    print("=" * 72)
    print(f"Failure system (original):  {filter_system}")
    print(f"Original failures found:    {len(original_failures)}")
    print()

    if not per_system:
        print("No matching swapped trials found.")
        print("Ensure you ran experiments on the swapped dataset and that")
        print("query IDs follow the swap_<original_id> naming convention.")
        print("=" * 72)
        return

    col_w = 30
    header = (
        f"{'System':<{col_w}} {'Swapped':>8} {'Flipped':>8}"
        f" {'Stayed':>8} {'Unclear':>8} {'Flip%':>7}  Verdict"
    )
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, dict[str, int]]] = sorted(per_system.items())
    for system, s in rows:
        rate = _flip_rate(s)
        print(
            f"{system:<{col_w}} {s['total']:>8} {s['flipped_correct']:>8}"
            f" {s['stayed_incorrect']:>8} {s['unclear']:>8} {rate:>6.1%}  {_verdict(rate)}"
        )

    print()
    if filter_system in per_system:
        s = per_system[filter_system]
        rate = _flip_rate(s)
        print(f"Primary system ({filter_system}):")
        print(f"  Flip rate : {rate:.1%}  ({s['flipped_correct']} / {s['total']})")
        print(f"  Verdict   : {_verdict(rate)}")
    else:
        print(f"Note: '{filter_system}' not found in swapped results.")
    print("=" * 72)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze counterfactual tone-swap flip rates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--original-results",
        required=True,
        metavar="DIR",
        help="Directory containing original experiment *_details.json files.",
    )
    p.add_argument(
        "--swapped-results",
        required=True,
        metavar="DIR",
        help="Directory containing swapped-dataset experiment *_details.json files.",
    )
    p.add_argument(
        "--filter-system",
        default="madam_rag",
        metavar="SYSTEM",
        help="System whose original failures define the swap cohort (default: madam_rag).",
    )
    p.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="Optional path to write analysis results as JSON.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    original_dir = Path(args.original_results)
    swapped_dir = Path(args.swapped_results)

    original_trials = _load_all_details(original_dir)
    swapped_trials = _load_all_details(swapped_dir)

    print(f"Loaded {len(original_trials)} original trials from {original_dir}.")
    print(f"Loaded {len(swapped_trials)} swapped trials from {swapped_dir}.")

    original_failures = _collect_original_failures(original_trials, args.filter_system)
    per_system = _analyze_swapped(swapped_trials, original_failures)

    _print_report(original_failures, per_system, args.filter_system)

    if args.json_out:
        out: dict[str, Any] = {
            "filter_system": args.filter_system,
            "original_failures": len(original_failures),
            "original_failure_ids": sorted(original_failures),
            "thresholds": {
                "strong": _THRESHOLD_STRONG,
                "moderate": _THRESHOLD_MODERATE,
            },
            "per_system": {
                system: {
                    **stats,
                    "flip_rate": _flip_rate(stats),
                    "verdict": _verdict(_flip_rate(stats)),
                }
                for system, stats in per_system.items()
            },
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written → {args.json_out}")


if __name__ == "__main__":
    main()
