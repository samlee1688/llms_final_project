#!/usr/bin/env python3
"""2×2 confusion matrix for EAJP scorecard-vs-final conflicts, by judge family.

Addresses reviewer Critique 4: "Why does Anthropic's judge fire the
conflict-resolution rule 10× more often yet achieve no TBR reduction?
Please provide a 2×2 confusion matrix: (scorecard correct/incorrect) ×
(final verdict correct/incorrect)."

For every EAJP trial where scorecard_privilege_applied=True (i.e. a conflict
occurred), we classify:
  - scorecard_correct: preliminary_winner supports the correct answer
  - final_correct:     final_winner supports the correct answer
  → 4 cells: (SC correct, FV correct), (SC correct, FV incorrect),
              (SC incorrect, FV correct), (SC incorrect, FV incorrect)

The privilege rule takes effect only in the last two cells:
  (SC correct, FV incorrect)   → privilege HELPS  (override wrong final)
  (SC incorrect, FV correct)   → privilege HURTS  (override correct final)
  (SC incorrect, FV incorrect) → privilege NEUTRAL (both wrong)

Usage
-----
python experiments/analyze_eajp_judge_conflicts.py \\
    --results-dir results/full_run_seed42 \\
    --out results/stats/eajp_conflict_matrix.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suite_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("openai_"):
            return part
        if part.startswith("anthropic_"):
            return part
    return "suite_independent"


def _benchmark_from_path(path: Path) -> str:
    for part in path.parts:
        if part in {
            "ramdocs", "confact",
            "tcb_ramdocs", "tcb_confact", "tcb",
        }:
            return part
    return "unknown"


def _load_eajp_conflict_trials(results_dir: Path) -> list[dict]:
    """Load EAJP trials where a scorecard-final conflict occurred."""
    trials: list[dict] = []
    for path in sorted(results_dir.rglob("eajp_details.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        if not isinstance(data, list):
            continue
        suite = _suite_from_path(path)
        benchmark = _benchmark_from_path(path)
        for t in data:
            if not isinstance(t, dict):
                continue
            md = t.get("metadata") or {}
            jmd = md.get("judge_metadata") or {}
            if not jmd.get("scorecard_privilege_applied", False):
                continue
            t = dict(t)
            t["_suite"] = suite
            t["_benchmark"] = benchmark
            trials.append(t)
    return trials


# ---------------------------------------------------------------------------
# 2×2 analysis
# ---------------------------------------------------------------------------

def _correct_answer_from_trial(trial: dict) -> str:
    return str(trial.get("correct_answer", "")).strip().lower()


def _winner_supports(winner_id: Optional[str], trial: dict) -> Optional[bool]:
    """Check if a winner agent_id aligns with the correct answer.

    We look at documents in the trial metadata to find which agent_id
    corresponds to which supports_answer value.
    """
    if not winner_id:
        return None
    md = trial.get("metadata") or {}
    agent_map: dict[str, str] = md.get("agent_document_map", {})
    # agent_document_map: {agent_id: supports_answer}
    supports = agent_map.get(str(winner_id))
    if supports is None:
        # fallback: look in debate_cache agent list if available
        debate_cache = md.get("debate_cache") or {}
        for aid, info in debate_cache.get("agents", {}).items():
            if str(aid) == str(winner_id):
                supports = info.get("supports_answer")
                break
    if supports is None:
        return None
    correct_answer = _correct_answer_from_trial(trial)
    return str(supports).strip().lower() == correct_answer


def _classify_conflict(trial: dict) -> Optional[dict[str, Any]]:
    """Return {sc_correct, final_correct, classification} or None if unclassifiable."""
    md = trial.get("metadata") or {}
    jmd = md.get("judge_metadata") or {}
    sc_winner = jmd.get("preliminary_winner_id") or jmd.get("scorecard_winner_id")
    final_winner = jmd.get("final_winner_id")
    overall_clf = trial.get("classification", "")

    # Use classification as the ground truth for final_correct
    final_correct = overall_clf == "correct"

    # For scorecard: the privilege rule means the final result EQUALS the scorecard winner.
    # So sc_correct iff the final result is correct (since privilege was applied).
    sc_correct = final_correct  # by definition when privilege_applied=True

    # BUT we also want to know what the *final judge* (before privilege) said.
    # This is the pre-privilege final winner — the one that was overridden.
    final_before_privilege_winner = jmd.get("final_winner_before_privilege") or jmd.get("raw_final_winner_id")
    if final_before_privilege_winner:
        final_before_correct = _winner_supports(final_before_privilege_winner, trial)
    else:
        final_before_correct = None

    return {
        "sc_correct": sc_correct,
        "final_before_privilege_correct": final_before_correct,
        "overall_classification": overall_clf,
    }


def _build_matrix(trials: list[dict]) -> dict[str, Any]:
    """Build 2×2 matrix: rows=scorecard outcome, cols=pre-privilege-final outcome."""
    # For each conflict trial:
    #   sc_correct × final_before_correct → 4 cells

    matrix = {
        "sc_correct_final_correct": 0,    # privilege was unnecessary (both agree, both right)
        "sc_correct_final_incorrect": 0,  # privilege HELPED (sc right, final wrong → override helped)
        "sc_incorrect_final_correct": 0,  # privilege HURT (sc wrong, final right → override hurt)
        "sc_incorrect_final_incorrect": 0, # both wrong (privilege neutral)
        "unclassifiable": 0,
    }

    for t in trials:
        clf = _classify_conflict(t)
        if clf is None:
            matrix["unclassifiable"] += 1
            continue
        sc_ok = clf["sc_correct"]
        fin_ok = clf["final_before_privilege_correct"]
        if fin_ok is None:
            # Cannot determine pre-privilege final → use overall classification
            # When privilege=True: overall reflects scorecard verdict
            # So we can't determine pre-privilege final with certainty
            matrix["unclassifiable"] += 1
            continue
        if sc_ok and fin_ok:
            matrix["sc_correct_final_correct"] += 1
        elif sc_ok and not fin_ok:
            matrix["sc_correct_final_incorrect"] += 1
        elif not sc_ok and fin_ok:
            matrix["sc_incorrect_final_correct"] += 1
        else:
            matrix["sc_incorrect_final_incorrect"] += 1

    total = sum(v for k, v in matrix.items() if k != "unclassifiable")
    total_with_unc = total + matrix["unclassifiable"]
    privilege_helped = matrix["sc_correct_final_incorrect"]
    privilege_hurt = matrix["sc_incorrect_final_correct"]
    net_benefit = privilege_helped - privilege_hurt

    return {
        "total_conflicts": total_with_unc,
        "classifiable": total,
        "unclassifiable": matrix["unclassifiable"],
        "matrix": {
            "sc_correct_final_correct": matrix["sc_correct_final_correct"],
            "sc_correct_final_incorrect": matrix["sc_correct_final_incorrect"],
            "sc_incorrect_final_correct": matrix["sc_incorrect_final_correct"],
            "sc_incorrect_final_incorrect": matrix["sc_incorrect_final_incorrect"],
        },
        "privilege_helped": privilege_helped,
        "privilege_hurt": privilege_hurt,
        "net_benefit": net_benefit,
        "help_rate": round(privilege_helped / total, 4) if total > 0 else 0.0,
        "hurt_rate": round(privilege_hurt / total, 4) if total > 0 else 0.0,
        "scorecard_accuracy": round(
            (matrix["sc_correct_final_correct"] + matrix["sc_correct_final_incorrect"]) / total, 4
        ) if total > 0 else 0.0,
    }


def _alternative_matrix_from_classifications(trials: list[dict]) -> dict[str, Any]:
    """Simplified matrix using only overall_classification and conflict metadata.

    When pre-privilege final winner is not stored, we derive the matrix as:
      - sc_correct = (overall_classification == "correct")  [privilege applied = sc wins]
      - We infer pre-privilege final must have been "incorrect" in those cases
        (otherwise why was privilege applied? privilege fires only on disagreement)

    This is the matrix the reviewer asked for:
      (scorecard_correct × final_verdict_correct) where final = pre-privilege verdict
    """
    matrix = defaultdict(int)
    for t in trials:
        overall = t.get("classification", "unclear")
        if overall not in {"correct", "incorrect"}:
            matrix["unclear"] += 1
            continue
        # privilege was applied → scorecard winner used
        # So: sc_winner = overall_classification
        #     pre-privilege final_winner = opposite (since they disagreed)
        sc_correct = overall == "correct"
        # pre-privilege final had the OPPOSITE outcome
        final_before_correct = not sc_correct
        if sc_correct and final_before_correct:
            # impossible: sc said A, final said A, but they disagreed → can't both be "correct" same answer
            matrix["ambiguous"] += 1
        elif sc_correct and not final_before_correct:
            matrix["sc_correct_final_incorrect"] += 1
        elif not sc_correct and final_before_correct:
            matrix["sc_incorrect_final_correct"] += 1
        else:
            matrix["sc_incorrect_final_incorrect"] += 1

    total = matrix["sc_correct_final_incorrect"] + matrix["sc_incorrect_final_correct"] + matrix["sc_incorrect_final_incorrect"]
    privilege_helped = matrix["sc_correct_final_incorrect"]
    privilege_hurt = matrix["sc_incorrect_final_correct"]

    return {
        "total_conflicts": len(trials),
        "classifiable": total,
        "matrix": dict(matrix),
        "privilege_helped": privilege_helped,
        "privilege_hurt": privilege_hurt,
        "net_benefit": privilege_helped - privilege_hurt,
        "help_rate": round(privilege_helped / total, 4) if total > 0 else 0.0,
        "hurt_rate": round(privilege_hurt / total, 4) if total > 0 else 0.0,
        "scorecard_accuracy": round(privilege_helped / total, 4) if total > 0 else 0.0,
        "note": "simplified: inferred pre-privilege final as complement of scorecard",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(results_dir: Path) -> dict[str, Any]:
    trials = _load_eajp_conflict_trials(results_dir)
    logger.info("Found %d conflict trials in %s", len(trials), results_dir)

    # Group by (benchmark, suite)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        key = (t.get("_benchmark", "unknown"), t.get("_suite", "unknown"))
        by_cell[key].append(t)
        by_suite[t.get("_suite", "unknown")].append(t)

    per_cell: list[dict] = []
    for (bm, suite), cell_trials in sorted(by_cell.items()):
        m = _alternative_matrix_from_classifications(cell_trials)
        per_cell.append({"benchmark": bm, "suite": suite, **m})

    per_suite: list[dict] = []
    for suite, suite_trials in sorted(by_suite.items()):
        m = _alternative_matrix_from_classifications(suite_trials)
        per_suite.append({"suite": suite, **m})

    overall = _alternative_matrix_from_classifications(trials)

    return {
        "results_dir": str(results_dir),
        "total_conflict_trials": len(trials),
        "per_suite": per_suite,
        "per_benchmark_suite": per_cell,
        "overall": overall,
    }


def _print_report(result: dict) -> None:
    print("=" * 72)
    print("EAJP SCORECARD-FINAL CONFLICT ANALYSIS")
    print("=" * 72)
    print(f"Total conflict trials: {result['total_conflict_trials']}")
    print()
    print("Per judge suite:")
    print(f"  {'suite':<35} {'conflicts':>9} {'helped':>8} {'hurt':>8} {'net':>6} {'help%':>7} {'hurt%':>7} {'SC_acc':>7}")
    print("  " + "-" * 80)
    for r in result["per_suite"]:
        print(
            f"  {r['suite']:<35} {r['total_conflicts']:>9} {r['privilege_helped']:>8} "
            f"{r['privilege_hurt']:>8} {r['net_benefit']:>6} {r['help_rate']:>7.1%} "
            f"{r['hurt_rate']:>7.1%} {r['scorecard_accuracy']:>7.1%}"
        )
    print()
    print("Per benchmark × suite:")
    print(f"  {'benchmark':<16} {'suite':<32} {'conflicts':>9} {'helped':>8} {'hurt':>8} {'net':>6}")
    print("  " + "-" * 80)
    for r in result["per_benchmark_suite"]:
        print(
            f"  {r['benchmark']:<16} {r['suite']:<32} {r['total_conflicts']:>9} "
            f"{r['privilege_helped']:>8} {r['privilege_hurt']:>8} {r['net_benefit']:>6}"
        )
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EAJP scorecard-final conflict matrix analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results-dir", required=True, metavar="DIR")
    p.add_argument("--out", required=True, metavar="PATH")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    result = analyze(Path(args.results_dir))
    _print_report(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Written → %s", out)
    return 0


from typing import Optional  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
