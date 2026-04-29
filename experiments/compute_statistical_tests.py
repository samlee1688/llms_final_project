#!/usr/bin/env python3
"""Bootstrap confidence intervals and permutation tests for TBR.

Addresses reviewer Critique 1: "single-seed point estimates; no CIs or
significance tests."

Usage
-----
# Single-seed CI (bootstrap over instances):
python experiments/compute_statistical_tests.py \\
    --results-dir results/full_run_seed42 \\
    --out results/stats/ci_seed42.json

# Multi-seed CI (bootstrap over seeds):
python experiments/compute_statistical_tests.py \\
    --results-dir results/full_run_seed42 results/full_run_seed7 \\
        results/full_run_seed123 results/full_run_seed256 results/full_run_seed999 \\
    --out results/stats/ci_multiseed.json

# With permutation tests (EAJP vs. MADAM-RAG, per benchmark × judge):
python experiments/compute_statistical_tests.py \\
    --results-dir results/full_run_seed42 ... \\
    --permutation-test \\
    --n-permutations 10000 \\
    --bonferroni-cells 9 \\
    --out results/stats/full_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_trials(results_dir: Path, tone_controlled_only: bool = True) -> list[dict]:
    """Load all *_details.json trial records from a run directory."""
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
            if tone_controlled_only:
                md = t.get("metadata") or {}
                if not md.get("tone_controlled", False):
                    continue
            suite = _suite_from_path(path)
            t = dict(t)
            t["_suite"] = suite
            t["_benchmark"] = _benchmark_from_path(path)
            trials.append(t)
    return trials


def _suite_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("openai_"):
            return part
        if part.startswith("anthropic_"):
            return part
    return "suite_independent"


def _benchmark_from_path(path: Path) -> str:
    for part in path.parts:
        if part in {"ramdocs", "confact", "tcb_ramdocs", "tcb_confact", "tcb"}:
            return part
    return "unknown"


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _tbr(trials: list[dict]) -> float:
    eligible = [t for t in trials if t.get("classification") in {"correct", "incorrect"}]
    if not eligible:
        return float("nan")
    incorrect = sum(1 for t in eligible if t.get("classification") == "incorrect")
    return incorrect / len(eligible)


def bootstrap_ci(
    trials: list[dict],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for TBR over a list of trial dicts.

    Returns {"mean": float, "ci_lower": float, "ci_upper": float, "n": int}.
    """
    rng = random.Random(seed)
    n = len(trials)
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}

    observed = _tbr(trials)
    boot_tbrs: list[float] = []
    for _ in range(n_bootstrap):
        sample = [trials[rng.randint(0, n - 1)] for _ in range(n)]
        boot_tbrs.append(_tbr(sample))

    boot_tbrs.sort()
    alpha = 1.0 - ci_level
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, lo_idx)
    hi_idx = min(n_bootstrap - 1, hi_idx)

    return {
        "mean": round(observed, 4),
        "ci_lower": round(boot_tbrs[lo_idx], 4),
        "ci_upper": round(boot_tbrs[hi_idx], 4),
        "n": n,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    trials_a: list[dict],
    trials_b: list[dict],
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> dict:
    """Paired permutation test: H0 = TBR(a) == TBR(b).

    Pairs trials by query_id; unpaired instances are dropped.
    Returns {"observed_diff": float, "p_value": float, "n_pairs": int}.
    """
    a_by_id: dict[str, str] = {}
    for t in trials_a:
        qid = t.get("query_id", "")
        clf = t.get("classification", "")
        if qid and clf in {"correct", "incorrect"}:
            a_by_id[qid] = clf

    b_by_id: dict[str, str] = {}
    for t in trials_b:
        qid = t.get("query_id", "")
        clf = t.get("classification", "")
        if qid and clf in {"correct", "incorrect"}:
            b_by_id[qid] = clf

    common = sorted(a_by_id.keys() & b_by_id.keys())
    if not common:
        return {"observed_diff": float("nan"), "p_value": float("nan"), "n_pairs": 0}

    a_labels = [1 if a_by_id[q] == "incorrect" else 0 for q in common]
    b_labels = [1 if b_by_id[q] == "incorrect" else 0 for q in common]
    n = len(common)

    tbr_a = sum(a_labels) / n
    tbr_b = sum(b_labels) / n
    observed_diff = tbr_a - tbr_b

    rng = random.Random(seed)
    count = 0
    for _ in range(n_permutations):
        perm_a, perm_b = list(a_labels), list(b_labels)
        for i in range(n):
            if rng.random() < 0.5:
                perm_a[i], perm_b[i] = perm_b[i], perm_a[i]
        diff = sum(perm_a) / n - sum(perm_b) / n
        if alternative == "two-sided":
            if abs(diff) >= abs(observed_diff):
                count += 1
        elif alternative == "greater":
            if diff >= observed_diff:
                count += 1
        else:
            if diff <= observed_diff:
                count += 1

    p_value = count / n_permutations
    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "n_pairs": n,
        "n_permutations": n_permutations,
        "alternative": alternative,
    }


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------

def bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni correction to a list of p-values."""
    m = len(p_values)
    return [
        {
            "p_value": p,
            "corrected_alpha": round(alpha / m, 6),
            "reject_h0": p < alpha / m,
        }
        for p in p_values
    ]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _group_trials(
    all_trials: list[dict],
) -> dict[tuple[str, str, str], list[dict]]:
    """Group trials by (benchmark, suite, system)."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for t in all_trials:
        key = (t.get("_benchmark", "unknown"), t.get("_suite", "unknown"), t.get("system", "unknown"))
        groups[key].append(t)
    return dict(groups)


def run_analysis(
    results_dirs: list[Path],
    n_bootstrap: int = 2000,
    n_permutations: int = 10000,
    ci_level: float = 0.95,
    run_permutation: bool = True,
    bonferroni_cells: int = 9,
    systems_a: list[str] | None = None,
    systems_b: list[str] | None = None,
    tone_controlled_only: bool = True,
    seed: int = 42,
) -> dict:
    """Full statistical analysis over one or more seed-run directories."""
    if systems_a is None:
        systems_a = ["eajp"]
    if systems_b is None:
        systems_b = ["madam_rag"]

    all_trials: list[dict] = []
    for d in results_dirs:
        run_trials = _load_trials(d, tone_controlled_only=tone_controlled_only)
        logger.info("Loaded %d trials from %s", len(run_trials), d)
        all_trials.extend(run_trials)

    logger.info("Total trials: %d", len(all_trials))
    groups = _group_trials(all_trials)

    # Per-cell CIs
    ci_results: list[dict] = []
    benchmarks = sorted({k[0] for k in groups})
    suites = sorted({k[1] for k in groups})
    systems = sorted({k[2] for k in groups})

    for bm in benchmarks:
        for suite in suites:
            for sys in systems:
                trials = groups.get((bm, suite, sys), [])
                if not trials:
                    continue
                ci = bootstrap_ci(trials, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed)
                ci_results.append(
                    {"benchmark": bm, "suite": suite, "system": sys, **ci}
                )

    # Permutation tests: systems_a vs systems_b per benchmark × suite
    perm_results: list[dict] = []
    if run_permutation:
        primary_cells = [
            (bm, suite)
            for bm in benchmarks
            for suite in suites
            if suite != "suite_independent"
        ]
        raw_p_values: list[float] = []
        cell_meta: list[dict] = []

        for bm, suite in primary_cells:
            for sys_a in systems_a:
                for sys_b in systems_b:
                    ta = groups.get((bm, suite, sys_a), [])
                    tb = groups.get((bm, suite, sys_b), [])
                    if not ta or not tb:
                        continue
                    res = permutation_test(ta, tb, n_permutations=n_permutations, seed=seed)
                    raw_p_values.append(res["p_value"])
                    cell_meta.append(
                        {"benchmark": bm, "suite": suite, "system_a": sys_a, "system_b": sys_b, **res}
                    )

        corrections = bonferroni_correct(raw_p_values, alpha=0.05)
        for meta, corr in zip(cell_meta, corrections):
            perm_results.append({**meta, **corr})

    # Aggregate TBR across all seeds (multi-seed mean ± std)
    aggregate: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key, trials in groups.items():
        tbr = _tbr(trials)
        if not (tbr != tbr):  # not nan
            aggregate[key].append(tbr)

    multi_seed_summary: list[dict] = []
    for (bm, suite, sys), values in sorted(aggregate.items()):
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
        std = variance ** 0.5
        multi_seed_summary.append(
            {
                "benchmark": bm,
                "suite": suite,
                "system": sys,
                "n_seeds": len(values),
                "tbr_mean": round(mean, 4),
                "tbr_std": round(std, 4),
                "tbr_values": [round(v, 4) for v in values],
            }
        )

    return {
        "results_dirs": [str(d) for d in results_dirs],
        "n_total_trials": len(all_trials),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations if run_permutation else 0,
        "bonferroni_m": bonferroni_cells,
        "per_cell_bootstrap_ci": ci_results,
        "permutation_tests": perm_results,
        "multi_seed_summary": multi_seed_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap CIs and permutation tests for TBR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more run directories (one per seed).",
    )
    p.add_argument("--out", required=True, metavar="PATH", help="Output JSON path.")
    p.add_argument("--n-bootstrap", type=int, default=2000, metavar="N")
    p.add_argument("--n-permutations", type=int, default=10000, metavar="N")
    p.add_argument(
        "--no-permutation-test", action="store_true",
        help="Skip permutation tests (faster).",
    )
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--bonferroni-cells", type=int, default=9)
    p.add_argument(
        "--system-a",
        nargs="+",
        default=["eajp"],
        help="System(s) for comparison A (default: eajp).",
    )
    p.add_argument(
        "--system-b",
        nargs="+",
        default=["madam_rag"],
        help="System(s) for comparison B (default: madam_rag).",
    )
    p.add_argument(
        "--all-instances",
        action="store_true",
        help="Include non-tone-controlled instances.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    import math  # noqa: F401 (needed in bootstrap_ci scope)
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dirs = [Path(d) for d in args.results_dir]
    for d in dirs:
        if not d.exists():
            raise SystemExit(f"Directory not found: {d}")

    result = run_analysis(
        results_dirs=dirs,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        ci_level=args.ci_level,
        run_permutation=not args.no_permutation_test,
        bonferroni_cells=args.bonferroni_cells,
        systems_a=args.system_a,
        systems_b=args.system_b,
        tone_controlled_only=not args.all_instances,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Written → %s", out_path)

    # Print summary
    print("\n=== MULTI-SEED TBR SUMMARY ===")
    print(f"{'benchmark':<18} {'suite':<32} {'system':<22} {'n_seeds':>7} {'TBR_mean':>9} {'TBR_std':>9}")
    print("-" * 100)
    for row in result["multi_seed_summary"]:
        print(
            f"{row['benchmark']:<18} {row['suite']:<32} {row['system']:<22} "
            f"{row['n_seeds']:>7} {row['tbr_mean']:>9.4f} {row['tbr_std']:>9.4f}"
        )

    if result["permutation_tests"]:
        print("\n=== PERMUTATION TESTS (Bonferroni-corrected) ===")
        alpha_c = 0.05 / args.bonferroni_cells
        print(f"Corrected α = 0.05 / {args.bonferroni_cells} = {alpha_c:.5f}")
        print(f"{'benchmark':<18} {'suite':<32} {'a':<22} {'b':<22} {'diff':>7} {'p':>7} {'reject':>7}")
        print("-" * 120)
        for r in result["permutation_tests"]:
            reject = "YES *" if r.get("reject_h0") else "no"
            print(
                f"{r['benchmark']:<18} {r['suite']:<32} {r['system_a']:<22} {r['system_b']:<22} "
                f"{r['observed_diff']:>7.4f} {r['p_value']:>7.4f} {reject:>7}"
            )
    return 0


# needed in bootstrap_ci when called from __main__ context
import math  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
