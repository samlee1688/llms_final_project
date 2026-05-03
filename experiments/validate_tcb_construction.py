#!/usr/bin/env python3
"""TCB construction validity and cross-benchmark TBR consistency analysis.

Two studies:

Study 1 — Fallback-rate report
  Scans one or more TCB JSON files (built by src/tcb_builder.py) for
  document-level rewrite metadata and reports what fraction of rewrites used
  the local deterministic fallback vs. Gemini, per-tone breakdown, and
  meaning-preservation rates.  A high fallback rate means many instances were
  rewritten with the simple prefix heuristic rather than a neural model,
  making TCB quality heterogeneous.

Study 2 — Cross-benchmark TBR consistency
  Loads *_details.json files from two or more results directories (each
  labelled with a short benchmark name) and computes per-system TBR on each
  benchmark.  If bias appears only on TCB (rewritten) data and not on
  native-tone benchmarks (RAMDocs), the construction process may be
  inflating bias rather than exposing it.

Usage examples:

  # Study 1 only — scan TCB JSON file(s) for fallback stats
  python experiments/validate_tcb_construction.py \
      --tcb-paths data/sample200_gemini_preprocess/tcb_ramdocs.json

  # Study 2 only — compare TBR across benchmarks from existing results
  python experiments/validate_tcb_construction.py \
      --compare tcb:results/run/suite_openai \
      --compare ramdocs:results/ramdocs_run/suite_openai

  # Both studies together
  python experiments/validate_tcb_construction.py \
      --tcb-paths data/sample200_gemini_preprocess/tcb_ramdocs.json \
      --compare tcb:results/tcb_run/suite_openai \
      --compare ramdocs:results/ramdocs_run/suite_openai \
      --json-out results/tcb_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Study 1 – Fallback-rate analysis
# ---------------------------------------------------------------------------

def _scan_tcb_file(path: Path) -> dict[str, Any]:
    """Scan one TCB JSON file for rewrite quality metadata.

    Returns aggregated counts at both instance and document level so callers
    can report whichever granularity is most appropriate.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {"error": "top-level JSON is not a list", "path": str(path)}

    inst_total = 0
    inst_gemini = 0
    inst_fallback = 0          # ≥1 document used fallback
    inst_meaning_ok = 0
    inst_missing_rewrite_meta = 0

    doc_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "fallback": 0, "meaning_ok": 0})

    for inst in data:
        if not isinstance(inst, dict):
            continue
        inst_meta = inst.get("metadata") or {}
        if not isinstance(inst_meta, dict):
            inst_meta = {}

        rewrite_ver = inst_meta.get("rewrite_verification")
        if not isinstance(rewrite_ver, dict):
            inst_missing_rewrite_meta += 1
        else:
            inst_total += 1
            inst_fallback_flag = False
            for tone_key in ("neutral", "assertive"):
                tone_info = rewrite_ver.get(tone_key) or {}
                if isinstance(tone_info, dict) and bool(tone_info.get("used_fallback", False)):
                    inst_fallback_flag = True
            if bool(inst_meta.get("meaning_preserved", False)):
                inst_meaning_ok += 1
            if inst_fallback_flag:
                inst_fallback += 1
            else:
                inst_gemini += 1

        # Document-level counts (more detailed)
        for doc in inst.get("documents") or []:
            if not isinstance(doc, dict):
                continue
            doc_meta = doc.get("metadata") or {}
            if not isinstance(doc_meta, dict):
                continue
            tone = str(doc.get("tone") or "unknown").lower()
            if "rewrite_used_fallback" not in doc_meta:
                continue
            doc_counts[tone]["total"] += 1
            if bool(doc_meta.get("rewrite_used_fallback", False)):
                doc_counts[tone]["fallback"] += 1
            if bool(doc_meta.get("rewrite_meaning_preserved", False)):
                doc_counts[tone]["meaning_ok"] += 1

    def _pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    result: dict[str, Any] = {
        "path": str(path),
        "instances_with_rewrite_meta": inst_total,
        "instances_missing_rewrite_meta": inst_missing_rewrite_meta,
        "gemini_instances": inst_gemini,
        "fallback_instances": inst_fallback,
        "fallback_rate_pct": _pct(inst_fallback, inst_total),
        "meaning_preserved_instances": inst_meaning_ok,
        "meaning_preserved_rate_pct": _pct(inst_meaning_ok, inst_total),
        "per_tone_docs": {},
    }
    for tone, cnts in sorted(doc_counts.items()):
        result["per_tone_docs"][tone] = {
            "total": cnts["total"],
            "fallback": cnts["fallback"],
            "fallback_rate_pct": _pct(cnts["fallback"], cnts["total"]),
            "meaning_ok": cnts["meaning_ok"],
            "meaning_ok_rate_pct": _pct(cnts["meaning_ok"], cnts["total"]),
        }
    return result


def run_fallback_study(tcb_paths: list[Path]) -> list[dict[str, Any]]:
    results = []
    for p in tcb_paths:
        if not p.exists():
            print(f"WARNING: TCB path not found — {p}", file=sys.stderr)
            continue
        results.append(_scan_tcb_file(p))
    return results


def print_fallback_report(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No TCB files scanned.")
        return
    print("\n" + "=" * 72)
    print("STUDY 1 — TCB CONSTRUCTION FALLBACK-RATE REPORT")
    print("=" * 72)
    for r in results:
        if "error" in r:
            print(f"\n  {r['path']}: ERROR — {r['error']}")
            continue
        total = r["instances_with_rewrite_meta"]
        missing = r["instances_missing_rewrite_meta"]
        print(f"\n  File: {r['path']}")
        print(f"  Instances with rewrite metadata : {total}")
        if missing:
            print(f"  Instances WITHOUT rewrite meta  : {missing}  (pre-Gemini pipeline?)")
        print(f"  Gemini-rewritten instances      : {r['gemini_instances']}"
              f"  ({100 - r['fallback_rate_pct']:.1f}%)")
        print(f"  Fallback (local heuristic)       : {r['fallback_instances']}"
              f"  ({r['fallback_rate_pct']:.1f}%)")
        print(f"  Meaning-preserved instances      : {r['meaning_preserved_instances']}"
              f"  ({r['meaning_preserved_rate_pct']:.1f}%)")
        tone_docs = r.get("per_tone_docs", {})
        if tone_docs:
            print("  Per-tone breakdown (documents):")
            for tone, td in tone_docs.items():
                print(f"    {tone:12s} : total={td['total']}"
                      f"  fallback={td['fallback']} ({td['fallback_rate_pct']:.1f}%)"
                      f"  meaning_ok={td['meaning_ok']} ({td['meaning_ok_rate_pct']:.1f}%)")
        # Quality assessment
        fb_rate = r["fallback_rate_pct"]
        if fb_rate == 0.0 and total > 0:
            verdict = "PASS — all instances used Gemini (no fallback)"
        elif fb_rate < 5.0:
            verdict = "PASS — negligible fallback rate (<5%)"
        elif fb_rate < 20.0:
            verdict = "WARN — moderate fallback rate; review fallback instances"
        else:
            verdict = "FAIL — high fallback rate; TCB quality is heterogeneous"
        print(f"  Quality assessment               : {verdict}")


# ---------------------------------------------------------------------------
# Study 2 – Cross-benchmark TBR consistency
# ---------------------------------------------------------------------------

def _iter_detail_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*_details.json"))


def _load_details(paths: list[Path]) -> list[dict]:
    trials: list[dict] = []
    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            trials.extend(t for t in payload if isinstance(t, dict))
    return trials


def _is_contentious(supports_answers: list[str]) -> bool:
    vals = {v.strip().lower() for v in supports_answers if v.strip()}
    return len(vals) > 1


def _compute_tbr_stats(trials: list[dict]) -> dict[str, Any]:
    """
    Compute per-system TBR and related metrics from a flat list of trial dicts.

    For each system:
      - tone_controlled_total: trials where metadata.tone_controlled == True
      - tbr: incorrect / (correct + incorrect) on tone_controlled trials
      - all_total, all_incorrect_rate: over all trials (for cross-check)
    """
    by_system: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "tc_total": 0, "tc_correct": 0, "tc_incorrect": 0,
            "all_total": 0, "all_correct": 0, "all_incorrect": 0,
        }
    )

    for trial in trials:
        system = str(trial.get("system", "unknown"))
        classification = str(trial.get("classification", ""))
        meta = trial.get("metadata") or {}
        tc = bool(meta.get("tone_controlled", False))

        by_system[system]["all_total"] += 1
        if classification == "correct":
            by_system[system]["all_correct"] += 1
        elif classification == "incorrect":
            by_system[system]["all_incorrect"] += 1

        if tc:
            by_system[system]["tc_total"] += 1
            if classification == "correct":
                by_system[system]["tc_correct"] += 1
            elif classification == "incorrect":
                by_system[system]["tc_incorrect"] += 1

    def _rate(n: int, d: int) -> float | None:
        return round(n / d, 6) if d else None

    out: dict[str, Any] = {}
    for system, c in sorted(by_system.items()):
        tc_resolved = c["tc_correct"] + c["tc_incorrect"]
        all_resolved = c["all_correct"] + c["all_incorrect"]
        out[system] = {
            "tone_controlled_total": c["tc_total"],
            "tone_controlled_correct": c["tc_correct"],
            "tone_controlled_incorrect": c["tc_incorrect"],
            "tbr": _rate(c["tc_incorrect"], tc_resolved),
            "tc_accuracy": _rate(c["tc_correct"], tc_resolved),
            "all_total": c["all_total"],
            "all_incorrect_rate": _rate(c["all_incorrect"], all_resolved),
        }
    return out


def run_crossbench_study(
    compare: list[tuple[str, Path]],
) -> dict[str, Any]:
    """Load results from each (label, results_dir) pair and compute TBR stats."""
    bench_results: dict[str, dict[str, Any]] = {}
    for label, path in compare:
        files = _iter_detail_files(path)
        if not files:
            print(f"WARNING: no *_details.json found under {path}", file=sys.stderr)
            continue
        trials = _load_details(files)
        bench_results[label] = _compute_tbr_stats(trials)
    return bench_results


def print_crossbench_report(bench_results: dict[str, dict[str, Any]]) -> None:
    if not bench_results:
        print("No benchmark results to compare.")
        return

    # Collect all systems across all benchmarks
    all_systems: list[str] = sorted(
        {sys for bstats in bench_results.values() for sys in bstats}
    )
    benchmarks = sorted(bench_results.keys())

    print("\n" + "=" * 72)
    print("STUDY 2 — CROSS-BENCHMARK TBR CONSISTENCY")
    print("=" * 72)
    print("\nTone Bias Rate (TBR = low-credibility win rate on tone-controlled trials)\n")

    # Header
    bench_w = max(14, max(len(b) for b in benchmarks))
    sys_w = max(20, max(len(s) for s in all_systems))
    header = f"  {'System':<{sys_w}}"
    for b in benchmarks:
        header += f"  {b:>{bench_w}}"
    print(header)
    print("  " + "-" * (sys_w + (bench_w + 2) * len(benchmarks)))

    system_tbrs: dict[str, list[float]] = defaultdict(list)
    for system in all_systems:
        row = f"  {system:<{sys_w}}"
        for b in benchmarks:
            stat = bench_results[b].get(system)
            if stat is None:
                row += f"  {'—':>{bench_w}}"
            else:
                tbr = stat.get("tbr")
                tc = stat.get("tone_controlled_total", 0)
                if tbr is None or tc == 0:
                    row += f"  {'n/a':>{bench_w}}"
                else:
                    cell = f"{tbr:.3f} (n={tc})"
                    row += f"  {cell:>{bench_w}}"
                    system_tbrs[system].append(tbr)
        print(row)

    # Consistency assessment per system
    print("\nConsistency assessment (max - min TBR across benchmarks):")
    for system in all_systems:
        vals = system_tbrs[system]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        if spread <= 0.05:
            verdict = "CONSISTENT — bias signal appears on all benchmarks"
        elif spread <= 0.15:
            verdict = "MODERATE SPREAD — investigate which benchmark is outlier"
        else:
            verdict = "INCONSISTENT — large TBR gap; TCB may be inflating bias"
        print(f"  {system:<{sys_w}}  spread={spread:.3f}  {verdict}")

    # Accuracy rows
    print("\nAccuracy (correct / resolved on tone-controlled trials):\n")
    print(header.replace("TBR", "Acc"))
    print("  " + "-" * (sys_w + (bench_w + 2) * len(benchmarks)))
    for system in all_systems:
        row = f"  {system:<{sys_w}}"
        for b in benchmarks:
            stat = bench_results[b].get(system)
            if stat is None:
                row += f"  {'—':>{bench_w}}"
            else:
                acc = stat.get("tc_accuracy")
                tc = stat.get("tone_controlled_total", 0)
                if acc is None or tc == 0:
                    row += f"  {'n/a':>{bench_w}}"
                else:
                    cell = f"{acc:.3f} (n={tc})"
                    row += f"  {cell:>{bench_w}}"
        print(row)


def _write_json(
    out_path: Path,
    fallback_results: list[dict[str, Any]],
    crossbench_results: dict[str, Any],
) -> None:
    payload = {
        "study1_fallback_rate": fallback_results,
        "study2_cross_benchmark_tbr": crossbench_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote JSON: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TCB construction validity and cross-benchmark TBR consistency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tcb-paths",
        nargs="+",
        metavar="PATH",
        default=[],
        help=(
            "One or more TCB JSON files built by src/tcb_builder.py "
            "(Study 1 — fallback-rate analysis)."
        ),
    )
    p.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL:RESULTS_DIR",
        help=(
            "Benchmark label and results directory to compare, e.g. "
            "tcb:results/tcb_run/openai_gpt-4o. "
            "Repeat for each benchmark. Directories are searched recursively "
            "for *_details.json files (Study 2 — cross-benchmark TBR)."
        ),
    )
    p.add_argument(
        "--json-out",
        default="",
        metavar="PATH",
        help="Optional path to write a JSON summary of both studies.",
    )
    return p.parse_args()


def _parse_compare(raw: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for entry in raw:
        if ":" not in entry:
            print(f"WARNING: --compare entry has no ':' separator — skipping '{entry}'",
                  file=sys.stderr)
            continue
        label, _, rest = entry.partition(":")
        parsed.append((label.strip(), Path(rest.strip())))
    return parsed


def main() -> int:
    args = parse_args()

    tcb_paths = [Path(p) for p in args.tcb_paths]
    compare = _parse_compare(args.compare)

    if not tcb_paths and not compare:
        print("Nothing to do. Pass --tcb-paths and/or --compare. See --help.", file=sys.stderr)
        return 2

    fallback_results: list[dict[str, Any]] = []
    crossbench_results: dict[str, Any] = {}

    if tcb_paths:
        fallback_results = run_fallback_study(tcb_paths)
        print_fallback_report(fallback_results)

    if compare:
        crossbench_results = run_crossbench_study(compare)
        print_crossbench_report(crossbench_results)

    if args.json_out:
        _write_json(Path(args.json_out), fallback_results, crossbench_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
