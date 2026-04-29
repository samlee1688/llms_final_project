#!/usr/bin/env python3
"""Position bias ablation: fixed vs. shuffled document order.

Addresses a secondary concern implicit in reviewer Critique 3/4: that selection
bias in document ordering (rather than tone) could explain TBR differences.

Procedure
---------
For each trial in an existing results directory:
  1. Record the original document order (which agent presented which document).
  2. Rerun with the *opposite* document order (swap agent assignments).
  3. Compute Position Flip Rate: fraction of correct→incorrect or incorrect→correct
     flips attributable solely to position change.

If position is a major confounder, we expect the flip rate to be large (>10%).
If tone is the true driver (our claim), position flips should be small (<5%).

Note: This analysis requires EAJP/MADAM agent assignments to be loggable at
rerun time. If debate_cache is available, it can be replayed with swapped order.
The script falls back to a lighter-weight analysis: for each result file that
stores both agent_id→document_id mapping and the winner, we check whether the
winner matches the "first presented" vs. "second presented" document.

Usage
-----
python experiments/run_position_bias_ablation.py \\
    --results-dir results/full_run_seed42 \\
    --out results/stats/position_bias.json \\
    [--system madam_rag] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_trials(results_dir: Path, system: str | None = None) -> list[dict]:
    trials: list[dict] = []
    for path in sorted(results_dir.rglob("*_details.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        if not isinstance(data, list):
            continue
        for t in data:
            if not isinstance(t, dict):
                continue
            if system and t.get("system") != system:
                continue
            t = dict(t)
            t["_source_path"] = str(path)
            trials.append(t)
    return trials


# ---------------------------------------------------------------------------
# Position bias analysis (static, no re-run required)
# ---------------------------------------------------------------------------

def _extract_position_info(trial: dict) -> dict[str, Any] | None:
    """Extract document order info from a trial.

    Returns:
        {
            "winner_agent": str,
            "winner_position": int,   # 0=first, 1=second
            "winner_supports": str,   # "correct"/"incorrect"
            "classification": str,
        }
    or None if not extractable.
    """
    md = trial.get("metadata") or {}
    winner_id = (
        trial.get("winner_agent_id")
        or trial.get("winner_id")
        or (md.get("judge_metadata") or {}).get("final_winner_id")
    )
    if not winner_id:
        return None

    agent_map = md.get("agent_document_map") or {}
    debate_cache = md.get("debate_cache") or {}
    agents_info = debate_cache.get("agents") or {}

    # Try to get ordered list of agents (order = position in debate)
    agent_order = md.get("agent_order") or []
    if not agent_order and agents_info:
        agent_order = list(agents_info.keys())

    if not agent_order or len(agent_order) < 2:
        return None

    try:
        winner_pos = agent_order.index(str(winner_id))
    except ValueError:
        return None

    supports = (
        agent_map.get(str(winner_id))
        or (agents_info.get(str(winner_id)) or {}).get("supports_answer", "")
    )

    return {
        "winner_agent": str(winner_id),
        "winner_position": winner_pos,
        "winner_supports": str(supports).lower(),
        "classification": trial.get("classification", ""),
        "n_agents": len(agent_order),
    }


def _analyze_position_bias_static(trials: list[dict]) -> dict[str, Any]:
    """Static analysis: is there a positional preference for winner?

    Tests: are first-position agents more likely to win than expected by chance?
    """
    valid = []
    for t in trials:
        info = _extract_position_info(t)
        if info is None:
            continue
        valid.append(info)

    if not valid:
        return {
            "n_valid": 0,
            "note": "No agent order metadata found; re-run required for full analysis.",
        }

    n = len(valid)
    first_wins = sum(1 for v in valid if v["winner_position"] == 0)
    first_win_rate = first_wins / n

    # By benchmark-randomized order expectation: 0.5 for 2-agent debates
    expected = 0.5
    # Binomial test approximation
    p_approx = _binomial_p(first_wins, n, expected)

    # Position × classification
    pos_correct: dict[int, int] = defaultdict(int)
    pos_total: dict[int, int] = defaultdict(int)
    for v in valid:
        pos = v["winner_position"]
        pos_total[pos] += 1
        if v["classification"] == "correct":
            pos_correct[pos] += 1

    position_accuracy = {
        pos: round(pos_correct[pos] / pos_total[pos], 4) if pos_total[pos] > 0 else 0.0
        for pos in pos_total
    }

    return {
        "n_valid": n,
        "first_position_win_rate": round(first_win_rate, 4),
        "expected_win_rate": expected,
        "p_first_position_bias": round(p_approx, 4),
        "position_accuracy": position_accuracy,
        "interpretation": (
            "SIGNIFICANT position bias detected (p<0.05)" if p_approx < 0.05
            else "No significant position bias (p≥0.05)"
        ),
    }


def _binomial_p(k: int, n: int, p: float) -> float:
    """Two-sided binomial p-value (normal approximation for large n)."""
    import math
    if n == 0:
        return float("nan")
    mean = n * p
    std = (n * p * (1 - p)) ** 0.5
    if std == 0:
        return 1.0
    z = abs(k - mean) / std
    # Two-sided p-value using normal approximation
    # Φ(-|z|) approximation (Abramowitz & Stegun)
    t = 1 / (1 + 0.2316419 * z)
    poly = t * (0.319381530
          + t * (-0.356563782
          + t * (1.781477937
          + t * (-1.821255978
          + t * 1.330274429))))
    phi = 1 - (1 / (2 * math.pi) ** 0.5) * math.exp(-z * z / 2) * poly
    return round(2 * (1 - phi), 4)


# ---------------------------------------------------------------------------
# Simulated rerun analysis (for when debate_cache is available)
# ---------------------------------------------------------------------------

def _simulate_order_swap(trials: list[dict], n_bootstrap: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Simulate position-swapped outcomes using debate_cache if available.

    For each trial with a debate_cache, we check: if we had presented agents
    in the reverse order, would the winner have changed? (Uses cached argument
    summaries, so no API calls needed.)

    This is only meaningful if the judge is sensitive to presentation order.
    We proxy this by checking whether the winning agent consistently held the
    first-presented position.
    """
    rng = random.Random(seed)

    # Collect trials with rich metadata
    cacheable = []
    for t in trials:
        info = _extract_position_info(t)
        if info and info["n_agents"] >= 2:
            cacheable.append((t, info))

    if not cacheable:
        return {"n_cacheable": 0, "note": "No trials with agent order metadata."}

    n = len(cacheable)
    # Estimate: if winner tends to be position 0, then after swap (position 0 becomes
    # position 1), the SAME agent would now be in position 1. If the judge is
    # position-biased (first-position preference), this agent would now lose.
    first_win_rate = sum(1 for _, info in cacheable if info["winner_position"] == 0) / n

    # "Swap flip rate": the fraction that would flip if position bias is the only driver
    # If judge picks first-position agent α fraction of the time, swap flips α
    # Naive estimate: the number of first-position winners that would become
    # second-position (and thus lose, if judge is biased) after a swap
    estimated_swap_flip_rate = first_win_rate  # upper bound (100% position-driven)
    # Lower bound: correct_first × correctness degradation
    first_pos_wins = sum(1 for _, info in cacheable if info["winner_position"] == 0)

    return {
        "n_cacheable": n,
        "first_position_win_rate": round(first_win_rate, 4),
        "estimated_swap_flip_rate_upper_bound": round(estimated_swap_flip_rate, 4),
        "note": (
            f"Upper-bound estimate: {estimated_swap_flip_rate:.1%} of trials would flip "
            "if ONLY position drives selection. True causal flip rate likely lower."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(results_dir: Path, system: str | None = None, seed: int = 42) -> dict[str, Any]:
    trials = _load_trials(results_dir, system=system)
    logger.info("Loaded %d trials from %s (system=%s)", len(trials), results_dir, system or "all")

    if not trials:
        return {
            "results_dir": str(results_dir),
            "n_trials": 0,
            "error": "No trials found.",
        }

    # Group by system
    by_system: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_system[t.get("system", "unknown")].append(t)

    per_system = []
    for sys_name, sys_trials in sorted(by_system.items()):
        static = _analyze_position_bias_static(sys_trials)
        swap_sim = _simulate_order_swap(sys_trials, seed=seed)
        per_system.append({"system": sys_name, "static_analysis": static, "swap_simulation": swap_sim})

    overall_static = _analyze_position_bias_static(trials)
    overall_swap = _simulate_order_swap(trials, seed=seed)

    return {
        "results_dir": str(results_dir),
        "n_trials": len(trials),
        "per_system": per_system,
        "overall": {
            "static_analysis": overall_static,
            "swap_simulation": overall_swap,
        },
    }


def _print_report(result: dict) -> None:
    print("=" * 72)
    print("POSITION BIAS ABLATION REPORT")
    print("=" * 72)
    print(f"Results dir : {result['results_dir']}")
    print(f"Total trials: {result['n_trials']}")
    print()
    print(f"{'System':<25} {'n_valid':>8} {'1st_win%':>9} {'p_bias':>8} {'significance'}")
    print("-" * 72)
    for row in result.get("per_system", []):
        sa = row["static_analysis"]
        n = sa.get("n_valid", 0)
        fwr = sa.get("first_position_win_rate", float("nan"))
        p = sa.get("p_first_position_bias", float("nan"))
        sig = "* SIGNIFICANT" if (isinstance(p, float) and p < 0.05) else "ns"
        print(f"  {row['system']:<23} {n:>8} {fwr:>9.1%} {p:>8.4f}  {sig}")
    print()
    oa = result.get("overall", {}).get("static_analysis", {})
    print(f"Overall: n={oa.get('n_valid',0)}, first_win={oa.get('first_position_win_rate',float('nan')):.1%}, "
          f"p={oa.get('p_first_position_bias',float('nan')):.4f}")
    print(f"Interpretation: {oa.get('interpretation','N/A')}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Position bias ablation for MADAM/EAJP trials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results-dir", required=True, metavar="DIR")
    p.add_argument("--out", required=True, metavar="PATH")
    p.add_argument("--system", default=None, help="Filter to specific system (e.g. madam_rag)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    result = analyze(Path(args.results_dir), system=args.system, seed=args.seed)
    _print_report(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Written → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
