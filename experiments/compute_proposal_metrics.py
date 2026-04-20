#!/usr/bin/env python3
"""Compute proposal-specific metrics from *_details.json outputs.

Metrics:
- substance_alignment_rate: classification == "correct" over eligible trials
- tone_bias_rate: classification == "incorrect" over eligible trials
- scorecard_final_conflict_rate (EAJP): scorecard_privilege_applied == true over EAJP trials
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute proposal metrics from details outputs.")
    p.add_argument("--path", required=True, help="Run directory or details file path.")
    p.add_argument(
        "--tone-controlled-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict low-credibility win rate to tone-controlled instances (default: true).",
    )
    p.add_argument("--json-out", default="", help="Optional JSON output path.")
    return p.parse_args()


def _iter_detail_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*_details.json"))


def _suite_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("openai_"):
        return parent
    if parent.startswith("anthropic_"):
        return parent
    return "suite_independent"


def _bool(x: Any) -> bool:
    return bool(x)


def main() -> int:
    args = parse_args()
    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Path does not exist: {target}")

    files = _iter_detail_files(target)
    if not files:
        raise SystemExit(f"No *_details.json files found under: {target}")

    totals = defaultdict(lambda: {"eligible": 0, "correct": 0, "incorrect": 0, "eajp_total": 0, "eajp_conflicts": 0})

    for file in files:
        suite = _suite_from_path(file)
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue

        for trial in payload:
            if not isinstance(trial, dict):
                continue
            system = str(trial.get("system", ""))
            classification = str(trial.get("classification", ""))
            md = trial.get("metadata") if isinstance(trial.get("metadata"), dict) else {}

            eligible = True
            if args.tone_controlled_only:
                eligible = _bool(md.get("tone_controlled", False))

            key = (suite, system)
            if eligible:
                totals[key]["eligible"] += 1
                if classification == "correct":
                    totals[key]["correct"] += 1
                if classification == "incorrect":
                    totals[key]["incorrect"] += 1

            if system == "eajp":
                totals[key]["eajp_total"] += 1
                judge_md = md.get("judge_metadata") if isinstance(md.get("judge_metadata"), dict) else {}
                if _bool(judge_md.get("scorecard_privilege_applied", False)):
                    totals[key]["eajp_conflicts"] += 1

    rows = []
    overall_eligible = 0
    overall_correct = 0
    overall_incorrect = 0
    for (suite, system), agg in sorted(totals.items()):
        eligible = agg["eligible"]
        correct = agg["correct"]
        incorrect = agg["incorrect"]
        eajp_total = agg["eajp_total"]
        eajp_conflicts = agg["eajp_conflicts"]
        substance_alignment_rate = (correct / eligible) if eligible else 0.0
        tone_bias_rate = (incorrect / eligible) if eligible else 0.0
        conflict_rate = (eajp_conflicts / eajp_total) if eajp_total else 0.0
        rows.append(
            {
                "suite": suite,
                "system": system,
                "eligible_trials": eligible,
                "substance_aligned_wins": correct,
                "substance_alignment_rate": substance_alignment_rate,
                "low_credibility_wins": incorrect,
                "low_credibility_win_rate": tone_bias_rate,
                "tone_bias_rate": tone_bias_rate,
                "eajp_trials": eajp_total,
                "scorecard_final_conflicts": eajp_conflicts,
                "scorecard_final_conflict_rate": conflict_rate,
            }
        )
        overall_eligible += eligible
        overall_correct += correct
        overall_incorrect += incorrect

    print(
        f"{'suite':<30} {'system':<12} {'eligible':>8} {'low_cred_wins':>14} "
        f"{'low_cred_rate':>14} {'eajp_trials':>12} {'flip_rate':>10}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"{r['suite']:<30} {r['system']:<12} {r['eligible_trials']:>8} {r['low_credibility_wins']:>14} "
            f"{r['low_credibility_win_rate']:>14.4f} {r['eajp_trials']:>12} {r['scorecard_final_conflict_rate']:>10.4f}"
        )

    if args.json_out:
        overall_sar = (overall_correct / overall_eligible) if overall_eligible else 0.0
        overall_tbr = (overall_incorrect / overall_eligible) if overall_eligible else 0.0
        resolved = overall_correct + overall_incorrect
        overall_acc = (overall_correct / resolved) if resolved else 0.0

        # Emit a dict-shaped payload so run_experiments summary aggregation can
        # safely consume this file when it matches the *_metrics.json glob.
        payload = {
            "system": "proposal_metrics",
            "model": "aggregate",
            "total": overall_eligible,
            "substance_alignment_rate": overall_sar,
            "tone_bias_rate": overall_tbr,
            "accuracy": overall_acc,
            "rows": rows,
        }

        out = Path(args.json_out)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
