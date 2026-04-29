#!/usr/bin/env python3
"""Analyze why EAJP behaves differently across judge families (OpenAI vs Anthropic).

Three hypotheses are tested:

  H1. The scorecard privilege rule fires at different rates across families.
      If it fires rarely for one family, the override mechanism has little
      opportunity to correct that family's tone bias.

  H2. One family's final judge naturally aligns with the scorecard without
      needing the privilege override (low scorecard↔final conflict rate,
      and/or the final phase adds accuracy on top of the scorecard baseline).
      This would mean structured two-phase prompting adds less net signal.

  H3. One family's MADAM-RAG baseline already has lower tone bias, leaving
      less room for EAJP to improve — a ceiling/floor effect rather than a
      genuine judge-family interaction with the EAJP mechanism.

A fourth section examines scorecard quality: whether each judge family produces
calibrated support assessments (strong_support for high-credibility docs,
weak/contradicted for low-credibility ones).

Input layout (one run root directory):
  <results_dir>/<benchmark>/<suite>/eajp_details.json
  <results_dir>/<benchmark>/<suite>/eajp_eajp_attribution.json   (pre-computed)
  <results_dir>/<benchmark>/<suite>/eajp_metrics.json
  <results_dir>/<benchmark>/<suite>/madam_rag_metrics.json

Output (default: <results_dir>/judge_divergence/):
  judge_divergence_report.txt         — human-readable analysis
  judge_divergence_summary.json       — machine-readable aggregates
  judge_divergence_per_suite.csv      — per benchmark × suite table
  plots/h1_privilege_rates.{png,pdf}
  plots/h2_phase_accuracy.{png,pdf}
  plots/h3_eajp_benefit.{png,pdf}
  plots/scorecard_support_quality.{png,pdf}

Usage:
  python experiments/analyze_judge_divergence.py \\
      --results-dir results/fast_pass_sample100

Dependencies: matplotlib and scipy are optional; the report and CSV are always
produced, and statistical tests fall back to chi-squared when scipy is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scipy.stats import fisher_exact as _scipy_fisher  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Judge-family detection
# ---------------------------------------------------------------------------

def _family(suite: str) -> str:
    s = suite.lower()
    if s.startswith("openai"):
        return "openai"
    if s.startswith("anthropic"):
        return "anthropic"
    return "other"


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_suite_dirs(results_dir: Path, system: str = "eajp") -> list[tuple[str, str, Path]]:
    """Return (benchmark, suite, suite_path) for every suite dir containing
    <system>_details.json, sorted for deterministic ordering."""
    found: list[tuple[str, str, Path]] = []
    for detail_file in sorted(results_dir.rglob(f"{system}_details.json")):
        suite_dir = detail_file.parent
        benchmark_dir = suite_dir.parent
        suite = suite_dir.name
        benchmark = benchmark_dir.name
        if suite == benchmark:
            # Flat layout — no benchmark sub-dir; skip.
            continue
        found.append((benchmark, suite, suite_dir))
    return found


# ---------------------------------------------------------------------------
# Per-trial metrics extracted from eajp_details.json
# ---------------------------------------------------------------------------

def _extract_trial_metrics(detail_path: Path) -> dict[str, Any]:
    """Parse every EAJP trial in a details file and count key events."""
    data = _load_json(detail_path)
    trials: list[dict] = data if isinstance(data, list) else []

    n = 0
    n_privilege = 0
    n_short_circuit = 0
    n_scorecard_only = 0
    n_prelim_empty = 0
    n_conflict = 0          # prelim_winner_id ≠ final_winner_id_before_privilege

    support_counts: dict[str, int] = defaultdict(int)
    credibility_counts: dict[str, int] = defaultdict(int)
    n_scorecard_entries = 0

    for trial in trials:
        if not isinstance(trial, dict):
            continue
        md = trial.get("metadata") if isinstance(trial.get("metadata"), dict) else {}
        jmd = md.get("judge_metadata") if isinstance(md.get("judge_metadata"), dict) else {}
        if not jmd:
            continue

        n += 1
        if bool(jmd.get("scorecard_privilege_applied", False)):
            n_privilege += 1
        if bool(jmd.get("short_circuit_used", False)):
            n_short_circuit += 1
        if bool(jmd.get("scorecard_only_used", False)):
            n_scorecard_only += 1

        prelim_id = str(jmd.get("preliminary_winner_id") or "").strip()
        before_id = str(jmd.get("final_winner_id_before_privilege") or "").strip()

        if not prelim_id:
            n_prelim_empty += 1
        if prelim_id and before_id and prelim_id != before_id:
            n_conflict += 1

        # Scorecard quality: iterate per-entry assessments
        sc_blob = jmd.get("scorecard") if isinstance(jmd.get("scorecard"), dict) else {}
        for entry in sc_blob.get("scorecard") or []:
            if not isinstance(entry, dict):
                continue
            n_scorecard_entries += 1
            sup = str(entry.get("document_support_assessment") or "").strip()
            cred = str(entry.get("credibility_tier") or "").strip()
            if sup:
                support_counts[sup] += 1
            if cred:
                credibility_counts[cred] += 1

    return {
        "n": n,
        "n_privilege": n_privilege,
        "n_short_circuit": n_short_circuit,
        "n_scorecard_only": n_scorecard_only,
        "n_prelim_empty": n_prelim_empty,
        "n_conflict": n_conflict,
        "n_scorecard_entries": n_scorecard_entries,
        "support_counts": dict(support_counts),
        "credibility_counts": dict(credibility_counts),
    }


# ---------------------------------------------------------------------------
# Load precomputed EAJP attribution and aggregate metrics
# ---------------------------------------------------------------------------

def _load_attribution(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _load_metrics(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _rate(n: int, d: int) -> float:
    return n / d if d > 0 else float("nan")


def _merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _test_proportions(
    a_yes: int, a_total: int, b_yes: int, b_total: int
) -> tuple[float | None, str]:
    """Two-sided test for H0: p_a == p_b.  Returns (p_value, method_name)."""
    a_no = a_total - a_yes
    b_no = b_total - b_yes
    if a_total == 0 or b_total == 0:
        return None, "N/A"

    if HAS_SCIPY:
        try:
            _, p = _scipy_fisher([[a_yes, a_no], [b_yes, b_no]])
            return float(p), "Fisher's exact"
        except Exception:
            pass

    # Fallback: Yates-corrected chi-squared (1 df)
    n = a_total + b_total
    r1 = a_yes + b_yes
    r2 = a_no + b_no
    if r1 == 0 or r2 == 0:
        return None, "N/A"
    exp_vals = [r1 * a_total / n, r2 * a_total / n, r1 * b_total / n, r2 * b_total / n]
    obs_vals = [a_yes, a_no, b_yes, b_no]
    if any(e <= 0 for e in exp_vals):
        return None, "N/A"
    chi2 = sum(
        (abs(o - e) - 0.5) ** 2 / e
        for o, e in zip(obs_vals, exp_vals)
    )
    p = float(math.erfc(math.sqrt(chi2 / 2.0)))
    return p, "chi-squared (Yates)"


# ---------------------------------------------------------------------------
# Core analysis: build per-record table and family aggregates
# ---------------------------------------------------------------------------

def build_analysis(results_dir: Path, system: str = "eajp") -> dict[str, Any]:
    suite_dirs = _iter_suite_dirs(results_dir, system=system)
    if not suite_dirs:
        raise FileNotFoundError(
            f"No {system}_details.json files found under {results_dir}.\n"
            "Pass the run root directory, e.g. results/fast_pass_sample100.\n"
            f"Use --system to specify a different system (e.g. eajp_no_privilege)."
        )

    records: list[dict[str, Any]] = []

    for benchmark, suite, suite_path in suite_dirs:
        fam = _family(suite)

        tm = _extract_trial_metrics(suite_path / f"{system}_details.json")

        attr = _load_attribution(suite_path / f"{system}_{system}_attribution.json")
        counts = attr.get("counts") if isinstance(attr.get("counts"), dict) else {}

        eajp_m = _load_metrics(suite_path / f"{system}_metrics.json")
        madam_m = _load_metrics(suite_path / "madam_rag_metrics.json")

        records.append({
            "benchmark": benchmark,
            "suite": suite,
            "family": fam,
            # H1 — privilege / conflict counters (from detail file)
            "n_trials": tm["n"],
            "n_privilege": tm["n_privilege"],
            "n_short_circuit": tm["n_short_circuit"],
            "n_scorecard_only": tm["n_scorecard_only"],
            "n_prelim_empty": tm["n_prelim_empty"],
            "n_conflict": tm["n_conflict"],
            # H2 — phase accuracy (from precomputed attribution JSON)
            "prelim_acc": _safe_float(attr.get("preliminary_accuracy")),
            "before_priv_acc": _safe_float(attr.get("final_before_privilege_accuracy")),
            "after_priv_acc": _safe_float(attr.get("final_after_privilege_accuracy")),
            "attr_prelim_total": int(counts.get("preliminary_total", 0)),
            "attr_before_total": int(counts.get("final_before_total", 0)),
            "attr_after_total": int(counts.get("final_after_total", 0)),
            "attr_conflict_total": int(counts.get("conflict_total", 0)),
            "attr_conflict_toward": int(counts.get("conflict_toward_total", 0)),
            "attr_conflict_away": int(counts.get("conflict_away_total", 0)),
            # H3 — EAJP vs MADAM-RAG benefit (from metrics JSONs)
            "eajp_tbr": _safe_float(eajp_m.get("tone_bias_rate")),
            "eajp_acc": _safe_float(eajp_m.get("accuracy")),
            "eajp_total": int(eajp_m.get("total", 0)),
            "madam_tbr": _safe_float(madam_m.get("tone_bias_rate")),
            "madam_acc": _safe_float(madam_m.get("accuracy")),
            "madam_total": int(madam_m.get("total", 0)),
            # Scorecard quality
            "n_scorecard_entries": tm["n_scorecard_entries"],
            "support_counts": tm["support_counts"],
            "credibility_counts": tm["credibility_counts"],
        })

    # ----- Aggregate by family ---------------------------------------------
    families = sorted({r["family"] for r in records})
    agg: dict[str, dict[str, Any]] = {}

    for fam in families:
        recs = [r for r in records if r["family"] == fam]

        # Counts for H1
        n_trials = sum(r["n_trials"] for r in recs)
        n_privilege = sum(r["n_privilege"] for r in recs)
        n_short_circuit = sum(r["n_short_circuit"] for r in recs)
        n_scorecard_only = sum(r["n_scorecard_only"] for r in recs)
        n_prelim_empty = sum(r["n_prelim_empty"] for r in recs)
        n_conflict = sum(r["n_conflict"] for r in recs)

        # Counts for H2 (attribution)
        attr_prelim_total = sum(r["attr_prelim_total"] for r in recs)
        attr_before_total = sum(r["attr_before_total"] for r in recs)
        attr_after_total = sum(r["attr_after_total"] for r in recs)
        attr_conflict_total = sum(r["attr_conflict_total"] for r in recs)
        attr_conflict_toward = sum(r["attr_conflict_toward"] for r in recs)
        attr_conflict_away = sum(r["attr_conflict_away"] for r in recs)

        # Counts-weighted phase accuracy (more rigorous than simple mean)
        def _wacc(acc_key: str, total_key: str) -> float:
            pairs = [
                (r[acc_key], r[total_key])
                for r in recs
                if r[total_key] > 0 and not math.isnan(r[acc_key])
            ]
            total_w = sum(w for _, w in pairs)
            return sum(v * w for v, w in pairs) / total_w if total_w > 0 else float("nan")

        prelim_acc = _wacc("prelim_acc", "attr_prelim_total")
        before_priv_acc = _wacc("before_priv_acc", "attr_before_total")
        after_priv_acc = _wacc("after_priv_acc", "attr_after_total")

        # H3 — weighted TBR/ACC
        def _wavg_metric(key: str, total_key: str) -> float:
            pairs = [
                (r[key], r[total_key])
                for r in recs
                if r[total_key] > 0 and not math.isnan(r[key])
            ]
            total_w = sum(w for _, w in pairs)
            return sum(v * w for v, w in pairs) / total_w if total_w > 0 else float("nan")

        eajp_tbr = _wavg_metric("eajp_tbr", "eajp_total")
        madam_tbr = _wavg_metric("madam_tbr", "madam_total")
        eajp_acc = _wavg_metric("eajp_acc", "eajp_total")
        madam_acc = _wavg_metric("madam_acc", "madam_total")

        # Scorecard quality
        merged_support: dict[str, int] = {}
        merged_cred: dict[str, int] = {}
        for r in recs:
            merged_support = _merge_counts(merged_support, r["support_counts"])
            merged_cred = _merge_counts(merged_cred, r["credibility_counts"])
        n_sc_entries = sum(r["n_scorecard_entries"] for r in recs)

        agg[fam] = {
            # H1
            "n_trials": n_trials,
            "n_privilege": n_privilege,
            "n_short_circuit": n_short_circuit,
            "n_scorecard_only": n_scorecard_only,
            "n_prelim_empty": n_prelim_empty,
            "n_conflict": n_conflict,
            "privilege_rate": _rate(n_privilege, n_trials),
            "short_circuit_rate": _rate(n_short_circuit, n_trials),
            "prelim_empty_rate": _rate(n_prelim_empty, n_trials),
            "conflict_rate": _rate(n_conflict, n_trials),
            # H2
            "attr_prelim_total": attr_prelim_total,
            "attr_before_total": attr_before_total,
            "attr_after_total": attr_after_total,
            "attr_conflict_total": attr_conflict_total,
            "attr_conflict_toward": attr_conflict_toward,
            "attr_conflict_away": attr_conflict_away,
            "attr_conflict_rate": _rate(attr_conflict_total, attr_after_total),
            "attr_conflict_toward_rate": _rate(attr_conflict_toward, attr_conflict_total),
            "attr_conflict_away_rate": _rate(attr_conflict_away, attr_conflict_total),
            "prelim_acc": prelim_acc,
            "before_priv_acc": before_priv_acc,
            "after_priv_acc": after_priv_acc,
            # H3
            "eajp_tbr": eajp_tbr,
            "madam_tbr": madam_tbr,
            "eajp_acc": eajp_acc,
            "madam_acc": madam_acc,
            "delta_tbr": (madam_tbr - eajp_tbr) if not any(math.isnan(x) for x in [madam_tbr, eajp_tbr]) else float("nan"),
            "delta_acc": (eajp_acc - madam_acc) if not any(math.isnan(x) for x in [eajp_acc, madam_acc]) else float("nan"),
            # Scorecard quality
            "n_scorecard_entries": n_sc_entries,
            "support_counts": merged_support,
            "credibility_counts": merged_cred,
            # Metadata
            "benchmarks": sorted({r["benchmark"] for r in recs}),
            "suites": sorted({r["suite"] for r in recs}),
        }

    return {"records": records, "agg": agg, "families": families}


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def _fmt_rate(n: int, d: int) -> str:
    if d == 0:
        return "     N/A"
    return f"{n / d:6.2%}  ({n:4d}/{d:4d})"


def _fmt_pct(v: float) -> str:
    return f"{v:.2%}" if not math.isnan(v) else "N/A"


def _fmt_f4(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "  N/A"


def _fmt_delta(v: float) -> str:
    if math.isnan(v):
        return "  N/A"
    return (f"+{v:.4f}" if v >= 0 else f"{v:.4f}")


def _fmt_p(p: float | None, method: str) -> str:
    if p is None:
        return "(insufficient data)"
    stars = ""
    if p < 0.001:
        stars = " ***"
    elif p < 0.01:
        stars = " **"
    elif p < 0.05:
        stars = " *"
    else:
        stars = " (n.s.)"
    return f"p = {p:.4f}{stars}  [{method}]"


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(data: dict[str, Any], output_dir: Path, system: str = "eajp") -> None:
    agg = data["agg"]
    families = data["families"]
    records = data["records"]
    benchmarks = sorted({r["benchmark"] for r in records})

    lines: list[str] = []
    nl = lines.append
    rule = lambda c="─", n=74: nl(c * n)

    nl("=" * 74)
    nl(f"  JUDGE-FAMILY DIVERGENCE ANALYSIS  [{system.upper()}]")
    nl("=" * 74)
    nl("")
    nl("Three hypotheses about why OpenAI and Anthropic judges respond differently")
    nl("to Evidence-Anchored Judge Prompting (EAJP).")
    nl("")
    for fam in families:
        a = agg[fam]
        nl(f"  {fam.upper():<14}  suites : {', '.join(a['suites'])}")
        nl(f"               benchmarks: {', '.join(a['benchmarks'])}")
        nl(f"               EAJP trials: {a['n_trials']}")
    nl("")

    # -------------------------------------------------------------------
    # H1: Privilege rule firing rate
    # -------------------------------------------------------------------
    rule()
    nl("H1: DOES THE PRIVILEGE RULE FIRE AT DIFFERENT RATES ACROSS FAMILIES?")
    rule()
    nl("")
    nl("  Privilege fires when the final judge's choice differs from the scorecard")
    nl("  preliminary verdict and the scorecard winner is substituted in code.")
    nl("  Low firing rate → rule rarely overrides → limited bias-correction effect.")
    nl("")

    col_w = 42
    val_w = 22
    header = f"  {'Metric':<{col_w}}" + "".join(f"  {f.upper():>{val_w}}" for f in families)
    nl(header)
    rule("-", 74)

    h1_rows = [
        ("Privilege-applied rate", "n_privilege", "n_trials"),
        ("Short-circuit rate (final phase skipped)", "n_short_circuit", "n_trials"),
        ("Scorecard parse failure (empty prelim ID)", "n_prelim_empty", "n_trials"),
        ("Conflict rate (prelim ≠ final before override)", "n_conflict", "n_trials"),
    ]
    for label, num, den in h1_rows:
        row = f"  {label:<{col_w}}"
        for f in families:
            v = _fmt_rate(agg[f][num], agg[f][den])
            row += f"  {v:>{val_w}}"
        nl(row)
    nl("")

    # Statistical tests
    if len(families) == 2:
        f1, f2 = families[0], families[1]
        for label, num, den in [
            ("privilege rate", "n_privilege", "n_trials"),
            ("conflict rate", "n_conflict", "n_trials"),
        ]:
            p, method = _test_proportions(
                agg[f1][num], agg[f1][den],
                agg[f2][num], agg[f2][den],
            )
            nl(f"  Test ({label}  {f1} vs {f2}): {_fmt_p(p, method)}")
    nl("")

    nl("  Per-benchmark privilege-applied rate:")
    for bm in benchmarks:
        parts = []
        for fam in families:
            r = next((x for x in records if x["benchmark"] == bm and x["family"] == fam), None)
            if r is None:
                parts.append(f"{fam}: N/A")
            else:
                parts.append(f"{fam}: {_fmt_rate(r['n_privilege'], r['n_trials'])}")
        nl(f"    {bm:<28}  " + "   |   ".join(parts))
    nl("")

    # Interpretation
    nl("  INTERPRETATION:")
    for fam in families:
        a = agg[fam]
        nl(f"    {fam.upper()}: privilege_rate={_fmt_pct(a['privilege_rate'])}, "
           f"conflict_rate={_fmt_pct(a['conflict_rate'])}, "
           f"short_circuit_rate={_fmt_pct(a['short_circuit_rate'])}")
        if a["n_trials"] > 0:
            if a["privilege_rate"] < 0.02:
                nl(f"      → Rule fires rarely: scorecard and final judge mostly agree naturally.")
            elif a["privilege_rate"] > 0.05:
                nl(f"      → Rule fires often: judge frequently deviates from scorecard.")
        if a["prelim_empty_rate"] > 0.05:
            nl(f"      ⚠ High parse-failure rate ({_fmt_pct(a['prelim_empty_rate'])}): scorecard "
               f"JSON unparseable in >{a['n_prelim_empty']} trials — judge may not follow "
               f"the structured output format reliably.")
    nl("")

    # -------------------------------------------------------------------
    # H2: Natural scorecard alignment
    # -------------------------------------------------------------------
    rule()
    nl("H2: DOES ONE FAMILY'S FINAL JUDGE NATURALLY FOLLOW THE SCORECARD?")
    rule()
    nl("")
    nl("  Phase accuracy = fraction of trials where the chosen document was correct")
    nl("  at each decision stage. A high preliminary accuracy that does NOT drop in")
    nl("  the final phase (before privilege) suggests the judge reads evidence first")
    nl("  even without explicit enforcement — a natural instruction-following effect.")
    nl("")

    nl(f"  {'Phase':<{col_w}}" + "".join(f"  {f.upper():>{val_w}}" for f in families))
    rule("-", 74)
    phase_rows = [
        ("Preliminary — scorecard verdict", "prelim_acc"),
        ("Final before privilege — judge's raw choice", "before_priv_acc"),
        ("Final after privilege — reported answer", "after_priv_acc"),
    ]
    for label, key in phase_rows:
        row = f"  {label:<{col_w}}"
        for f in families:
            v = agg[f][key]
            row += f"  {_fmt_pct(v):>{val_w}}"
        nl(row)
    nl("")

    nl("  Phase delta: final_before_privilege − preliminary (positive = judge improves on scorecard)")
    for fam in families:
        a = agg[fam]
        p, b = a["prelim_acc"], a["before_priv_acc"]
        if not any(math.isnan(x) for x in [p, b]):
            delta = b - p
            if abs(delta) < 0.015:
                label = "judge ≈ scorecard"
            elif delta > 0:
                label = f"judge improves by {abs(delta):.2%} → argument reading adds signal"
            else:
                label = f"judge degrades by {abs(delta):.2%} → argument reading introduces bias"
            nl(f"    {fam.upper()}: {_fmt_delta(delta)}  ({label})")
    nl("")

    nl("  Conflict resolution direction (attribution-computed conflicts):")
    nl(f"  {'Metric':<{col_w}}" + "".join(f"  {f.upper():>{val_w}}" for f in families))
    rule("-", 74)
    conflict_rows = [
        ("Conflict rate (scorecard ≠ final judge)", "attr_conflict_rate"),
        ("Of conflicts: toward correctness (sc✓ final✗)", "attr_conflict_toward_rate"),
        ("Of conflicts: away from correctness (sc✗ final✓)", "attr_conflict_away_rate"),
    ]
    for label, key in conflict_rows:
        row = f"  {label:<{col_w}}"
        for f in families:
            v = agg[f][key]
            row += f"  {_fmt_pct(v):>{val_w}}"
        nl(row)
    nl("")

    nl("  INTERPRETATION:")
    for fam in families:
        a = agg[fam]
        cr = a["attr_conflict_rate"]
        toward = a["attr_conflict_toward_rate"]
        away = a["attr_conflict_away_rate"]
        p, b = a["prelim_acc"], a["before_priv_acc"]
        nl(f"    {fam.upper()}:")
        if not math.isnan(cr):
            nl(f"      conflict_rate={_fmt_pct(cr)}, toward={_fmt_pct(toward)}, "
               f"away={_fmt_pct(away)}")
        if not any(math.isnan(x) for x in [toward, away]) and (toward + away) > 0:
            net = toward - away
            if net > 0.15:
                nl(f"      → Privilege rule is beneficial ({fam}): corrects more errors than it introduces.")
            elif net < -0.15:
                nl(f"      → Privilege rule is counter-productive ({fam}): introduces more errors than it corrects.")
            else:
                nl(f"      → Privilege rule has near-neutral net direction for {fam}.")
        if not any(math.isnan(x) for x in [p, b]):
            gap = b - p
            if abs(gap) < 0.015:
                nl(f"      → {fam} final judge accuracy ≈ scorecard: argument read adds minimal "
                   f"accuracy signal (structured prompting may be redundant for this family).")
            elif gap > 0:
                nl(f"      → {fam} final judge improves on scorecard: argument reading is beneficial.")
            else:
                nl(f"      → {fam} final judge degrades vs scorecard: argument reading introduces tone bias.")
    nl("")

    # -------------------------------------------------------------------
    # H3: Ceiling / floor effect
    # -------------------------------------------------------------------
    rule()
    nl("H3: IS EAJP'S SMALLER BENEFIT FOR ONE FAMILY A CEILING EFFECT?")
    rule()
    nl("")
    nl("  If MADAM-RAG baseline TBR is already near zero for one family, EAJP")
    nl("  has less room to improve — a floor effect, not a judge-family interaction.")
    nl("  Comparing MADAM baselines across families isolates this confound.")
    nl("")

    nl(f"  {'Metric':<{col_w}}" + "".join(f"  {f.upper():>{val_w}}" for f in families))
    rule("-", 74)
    sys_label = system.upper().replace("_", "-")
    h3_rows = [
        ("MADAM-RAG TBR baseline", "madam_tbr"),
        (f"{sys_label} TBR", "eajp_tbr"),
        (f"TBR delta: MADAM − {sys_label}  (+ve = reduces bias)", "delta_tbr"),
        ("MADAM-RAG accuracy baseline", "madam_acc"),
        (f"{sys_label} accuracy", "eajp_acc"),
        (f"Accuracy delta: {sys_label} − MADAM  (+ve = helps)", "delta_acc"),
    ]
    for label, key in h3_rows:
        row = f"  {label:<{col_w}}"
        for f in families:
            v = agg[f][key]
            row += f"  {_fmt_f4(v):>{val_w}}"
        nl(row)
    nl("")

    nl("  Per-benchmark TBR delta (MADAM − EAJP, positive = EAJP reduces bias):")
    for bm in benchmarks:
        parts = []
        for fam in families:
            r = next((x for x in records if x["benchmark"] == bm and x["family"] == fam), None)
            if r is None or math.isnan(r["madam_tbr"]) or math.isnan(r["eajp_tbr"]):
                parts.append(f"{fam}: N/A")
            else:
                delta = r["madam_tbr"] - r["eajp_tbr"]
                parts.append(f"{fam}: {_fmt_delta(delta)}")
        nl(f"    {bm:<28}  " + "   |   ".join(parts))
    nl("")

    nl("  INTERPRETATION:")
    if len(families) == 2:
        f1, f2 = families[0], families[1]
        a1, a2 = agg[f1], agg[f2]
        m1, m2 = a1["madam_tbr"], a2["madam_tbr"]
        d1, d2 = a1["delta_tbr"], a2["delta_tbr"]
        if not any(math.isnan(x) for x in [m1, m2, d1, d2]):
            base_diff = abs(m1 - m2)
            benefit_diff = abs(d1 - d2)
            lower_base_fam = f1 if m1 < m2 else f2
            higher_benefit_fam = f1 if d1 > d2 else f2
            nl(f"    MADAM-RAG baseline TBR difference: |{m1:.4f} − {m2:.4f}| = {base_diff:.4f}")
            if base_diff < 0.02:
                nl(f"    → Baselines are similar across families. A ceiling effect is UNLIKELY.")
                nl(f"      The difference in EAJP benefit ({benefit_diff:.4f}) reflects a")
                nl(f"      genuine judge-family interaction with the EAJP mechanism.")
            else:
                nl(f"    → {lower_base_fam.upper()} has a lower MADAM baseline (less room to improve).")
                nl(f"      Ceiling effect is PLAUSIBLE for {lower_base_fam.upper()}.")
                nl(f"      Cannot be ruled out without controlling for initial bias level.")
            nl(f"    TBR improvement (delta) is larger for {higher_benefit_fam.upper()} "
               f"({d1:.4f} vs {d2:.4f}), consistent with")
            nl(f"    {higher_benefit_fam.upper()} judge being more responsive to evidence-first prompting.")
    nl("")

    # -------------------------------------------------------------------
    # Scorecard quality
    # -------------------------------------------------------------------
    rule()
    nl("SCORECARD QUALITY: HOW DOES EACH FAMILY ASSESS DOCUMENT SUPPORT?")
    rule()
    nl("")
    nl("  A well-calibrated scorecard assigns strong_support to high-credibility")
    nl("  documents and weak_support/contradicted to low-credibility ones. If one")
    nl("  family's scorecard systematically under-differentiates (e.g., most entries")
    nl("  are 'mixed'), the evidence signal from the scorecard phase is weaker.")
    nl("")

    all_support = sorted({k for f in families for k in agg[f]["support_counts"]})
    nl("  Document support assessment distribution (% of all scored doc entries):")
    nl(f"  {'Label':<30}" + "".join(f"  {f.upper():>{val_w}}" for f in families))
    rule("-", 74)
    for lbl in all_support:
        row = f"  {lbl:<30}"
        for f in families:
            sc = agg[f]["support_counts"]
            total = agg[f]["n_scorecard_entries"]
            row += f"  {_fmt_rate(sc.get(lbl, 0), total):>{val_w}}"
        nl(row)
    nl("")

    all_cred = sorted({k for f in families for k in agg[f]["credibility_counts"]})
    nl("  Credibility tier assigned in scorecard (% of all scored doc entries):")
    nl(f"  {'Tier':<30}" + "".join(f"  {f.upper():>{val_w}}" for f in families))
    rule("-", 74)
    for lbl in all_cred:
        row = f"  {lbl:<30}"
        for f in families:
            cc = agg[f]["credibility_counts"]
            total = agg[f]["n_scorecard_entries"]
            row += f"  {_fmt_rate(cc.get(lbl, 0), total):>{val_w}}"
        nl(row)
    nl("")

    nl("  INTERPRETATION:")
    for fam in families:
        a = agg[fam]
        sc = a["support_counts"]
        total = a["n_scorecard_entries"]
        if total > 0:
            strong_rate = sc.get("strong_support", 0) / total
            mixed_rate = sc.get("mixed", 0) / total
            weak_rate = (sc.get("weak_support", 0) + sc.get("contradicted", 0)) / total
            nl(f"    {fam.upper()}: strong={strong_rate:.2%}, mixed={mixed_rate:.2%}, "
               f"weak+contradicted={weak_rate:.2%}")
            if mixed_rate > 0.5:
                nl(f"      ⚠ Most entries are 'mixed' — scorecard may not be discriminating "
                   f"confidently for {fam}. This reduces the informational value of the "
                   f"preliminary verdict and weakens the privilege override.")
            elif strong_rate > 0.5:
                nl(f"      → Scorecard is assertive (mostly strong_support). If high-credibility "
                   f"docs consistently get strong_support, this indicates good calibration.")
    nl("")

    # -------------------------------------------------------------------
    # Consolidated finding
    # -------------------------------------------------------------------
    rule("═")
    nl("CONSOLIDATED FINDING")
    rule("═")
    nl("")
    if len(families) == 2:
        f1, f2 = families[0], families[1]
        a1, a2 = agg[f1], agg[f2]

        confirmed: list[str] = []
        not_confirmed: list[str] = []

        pr1, pr2 = a1["privilege_rate"], a2["privilege_rate"]
        if not any(math.isnan(x) for x in [pr1, pr2]):
            if abs(pr1 - pr2) > 0.02:
                higher = f1 if pr1 > pr2 else f2
                confirmed.append(
                    f"H1 CONFIRMED — privilege rates differ ({pr1:.2%} vs {pr2:.2%}). "
                    f"{higher.upper()} overrides more often."
                )
            else:
                not_confirmed.append(f"H1 NOT CONFIRMED — privilege rates similar ({pr1:.2%} vs {pr2:.2%}).")

        cr1, cr2 = a1["attr_conflict_rate"], a2["attr_conflict_rate"]
        b1, b2 = a1["before_priv_acc"], a2["before_priv_acc"]
        p1_acc, p2_acc = a1["prelim_acc"], a2["prelim_acc"]
        if not any(math.isnan(x) for x in [b1, b2, p1_acc, p2_acc]):
            gap1, gap2 = b1 - p1_acc, b2 - p2_acc
            if abs(gap1 - gap2) > 0.02:
                more_natural = f1 if gap1 > gap2 else f2
                confirmed.append(
                    f"H2 CONFIRMED — {more_natural.upper()} final judge adds more accuracy "
                    f"on top of scorecard ({_fmt_delta(max(gap1, gap2))}), suggesting "
                    f"stronger natural evidence alignment in the argument-reading phase."
                )
            else:
                not_confirmed.append(
                    f"H2 NOT CONFIRMED — both families show similar scorecard→final accuracy delta."
                )

        m1, m2 = a1["madam_tbr"], a2["madam_tbr"]
        d1, d2 = a1["delta_tbr"], a2["delta_tbr"]
        if not any(math.isnan(x) for x in [m1, m2, d1, d2]):
            if abs(m1 - m2) < 0.02 and abs(d1 - d2) > 0.02:
                not_confirmed.append(
                    f"H3 NOT CONFIRMED — MADAM baselines similar ({m1:.4f} vs {m2:.4f}). "
                    f"EAJP benefit difference ({d1:.4f} vs {d2:.4f}) is a judge-family "
                    f"interaction, not a ceiling effect."
                )
            elif abs(m1 - m2) >= 0.02:
                confirmed.append(
                    f"H3 PARTIALLY CONSISTENT — MADAM baselines differ "
                    f"({m1:.4f} vs {m2:.4f}). Ceiling effect possible for "
                    f"{(f1 if m1 < m2 else f2).upper()}."
                )
            else:
                not_confirmed.append("H3 NOT CONFIRMED — baselines and deltas both similar.")

        for s in confirmed:
            nl(f"  ✓ {s}")
        for s in not_confirmed:
            nl(f"  ✗ {s}")
    else:
        nl("  (Consolidated finding requires exactly two judge families.)")
    nl("")
    rule("═")
    nl("")

    report_path = output_dir / "judge_divergence_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Report saved → {report_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], output_dir: Path) -> None:
    path = output_dir / "judge_divergence_per_suite.csv"
    scalar_fields = [
        "benchmark", "suite", "family",
        "n_trials", "n_privilege", "n_short_circuit", "n_scorecard_only",
        "n_prelim_empty", "n_conflict",
        "prelim_acc", "before_priv_acc", "after_priv_acc",
        "attr_prelim_total", "attr_before_total", "attr_after_total",
        "attr_conflict_total", "attr_conflict_toward", "attr_conflict_away",
        "eajp_tbr", "eajp_acc", "eajp_total",
        "madam_tbr", "madam_acc", "madam_total",
        "n_scorecard_entries",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in scalar_fields})
    print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# JSON summary output
# ---------------------------------------------------------------------------

def write_summary_json(data: dict, output_dir: Path) -> None:
    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    payload = {
        "families": data["families"],
        "agg": _clean(data["agg"]),
        "n_records": len(data["records"]),
    }
    path = output_dir / "judge_divergence_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(data: dict, output_dir: Path, dpi: int = 200) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    agg = data["agg"]
    families = data["families"]
    records = data["records"]
    benchmarks = sorted({r["benchmark"] for r in records})

    FAMILY_COLORS = {"openai": "#1f77b4", "anthropic": "#d62728", "other": "#7f7f7f"}
    SUPPORT_COLORS = {
        "strong_support": "#2ca02c",
        "mixed": "#ff7f0e",
        "weak_support": "#d62728",
        "contradicted": "#7f7f7f",
    }

    def _savefig(fig: plt.Figure, stem: str) -> None:
        for ext in ("png", "pdf"):
            kw: dict[str, Any] = {"bbox_inches": "tight"}
            if ext == "png":
                kw["dpi"] = dpi
            fig.savefig(plots_dir / f"{stem}.{ext}", **kw)
        plt.close(fig)
        print(f"  Plot → {plots_dir / stem}.png")

    # -- H1: Privilege, conflict, short-circuit rates -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    rate_panels = [
        ("Privilege-applied rate", "privilege_rate"),
        ("Conflict rate (prelim ≠ final)", "conflict_rate"),
        ("Short-circuit rate", "short_circuit_rate"),
    ]
    for ax, (title, key) in zip(axes, rate_panels):
        vals = [agg[f].get(key, float("nan")) for f in families]
        bar_colors = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]
        y = [v if not math.isnan(v) else 0 for v in vals]
        bars = ax.bar(range(len(families)), y, color=bar_colors, width=0.5, edgecolor="white")
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([f.upper() for f in families], fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Rate")
        ymax = max((v for v in vals if not math.isnan(v)), default=0.15)
        ax.set_ylim(0, max(0.12, ymax * 1.35))
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            if not math.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.03,
                    f"{val:.2%}",
                    ha="center", va="bottom", fontsize=9,
                )
    fig.suptitle("H1: Privilege Rule Activation Rates by Judge Family", fontsize=12)
    _savefig(fig, "h1_privilege_rates")

    # -- H2: Phase accuracy -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    phase_keys = ["prelim_acc", "before_priv_acc", "after_priv_acc"]
    phase_labels = ["Preliminary\n(scorecard)", "Final before\nprivilege", "Final after\nprivilege"]
    x_pos = range(len(phase_labels))
    width = 0.35
    for i, fam in enumerate(families):
        offset = [xi + (i - len(families) / 2 + 0.5) * width for xi in x_pos]
        vals = [agg[fam].get(k, 0) for k in phase_keys]
        bars = ax.bar(
            offset, vals, width=width,
            label=fam.upper(), color=FAMILY_COLORS.get(fam, "#7f7f7f"), alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2%}", ha="center", va="bottom", fontsize=8.5,
                )
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_ylabel("Accuracy (fraction with correct doc selected)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("H2: Phase-by-Phase Accuracy — Does Each Phase Add Signal?", fontsize=11)
    _savefig(fig, "h2_phase_accuracy")

    # -- H3: TBR and Accuracy per benchmark, MADAM vs EAJP -----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    bm_x = range(len(benchmarks))
    n_fam = len(families)
    slot_w = 0.8 / (n_fam * 2)

    for ax, (m_key, e_key, ylabel, title) in zip(axes, [
        ("madam_tbr", "eajp_tbr", "Tone Bias Rate", "TBR: MADAM vs EAJP  (lower = better)"),
        ("madam_acc", "eajp_acc", "Accuracy", "Accuracy: MADAM vs EAJP  (higher = better)"),
    ]):
        for i, fam in enumerate(families):
            c = FAMILY_COLORS.get(fam, "#7f7f7f")
            m_vals = []
            e_vals = []
            for bm in benchmarks:
                r = next((x for x in records if x["benchmark"] == bm and x["family"] == fam), None)
                m_vals.append(r[m_key] if r and not math.isnan(r[m_key]) else 0)
                e_vals.append(r[e_key] if r and not math.isnan(r[e_key]) else 0)
            base_offset = [xi - 0.4 + i * 2 * slot_w + slot_w / 2 for xi in bm_x]
            eajp_offset = [o + slot_w for o in base_offset]
            ax.bar(base_offset, m_vals, width=slot_w, color=c, alpha=0.45,
                   hatch="//", label=f"{fam.upper()} MADAM" if i == 0 or True else "")
            ax.bar(eajp_offset, e_vals, width=slot_w, color=c, alpha=0.9,
                   label=f"{fam.upper()} EAJP" if i == 0 or True else "")
        ax.set_xticks(list(bm_x))
        ax.set_xticklabels(
            [b.replace("tcb_", "TCB-").replace("_", "\n") for b in benchmarks],
            fontsize=9,
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 0.55)
        ax.legend(fontsize=8, ncol=n_fam * 2, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(title, fontsize=10)
    fig.suptitle("H3: EAJP Benefit vs MADAM-RAG Baseline by Family and Benchmark", fontsize=12)
    _savefig(fig, "h3_eajp_benefit")

    # -- Scorecard quality --------------------------------------------------
    all_support_labels = sorted({k for f in families for k in agg[f]["support_counts"]})
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4.8), constrained_layout=True)
    if len(families) == 1:
        axes = [axes]
    for ax, fam in zip(axes, families):
        sc = agg[fam]["support_counts"]
        total = agg[fam]["n_scorecard_entries"]
        vals = [sc.get(lbl, 0) / total if total > 0 else 0 for lbl in all_support_labels]
        bar_colors_s = [SUPPORT_COLORS.get(lbl, "#aec7e8") for lbl in all_support_labels]
        bars = ax.bar(range(len(all_support_labels)), vals, color=bar_colors_s, width=0.55,
                      edgecolor="white")
        ax.set_xticks(range(len(all_support_labels)))
        ax.set_xticklabels([lbl.replace("_", "\n") for lbl in all_support_labels], fontsize=9)
        ax.set_title(f"{fam.upper()} — document support distribution", fontsize=10)
        ax.set_ylabel("Fraction of scored entries")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9,
                )
    fig.suptitle("Scorecard Quality: Document Support Assessment Distribution", fontsize=12)
    _savefig(fig, "scorecard_support_quality")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze EAJP judge-family divergence (OpenAI vs Anthropic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\nUsage:")[1] if "\nUsage:" in __doc__ else "",
    )
    p.add_argument(
        "--results-dir", required=True,
        help="Run results root directory (contains benchmark sub-directories).",
    )
    p.add_argument(
        "--system", default="eajp",
        help=(
            "Which EAJP-family system to analyze (default: eajp). "
            "Use 'eajp_no_privilege' to analyze the no-privilege ablation, "
            "which isolates scorecard framing without the mechanical override."
        ),
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <results-dir>/judge_divergence/).",
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="PNG plot resolution (default: 200).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    system = args.system
    subdir = "judge_divergence" if system == "eajp" else f"judge_divergence_{system}"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning results under: {results_dir}  [system={system}]")
    data = build_analysis(results_dir, system=system)

    write_report(data, output_dir, system=system)
    write_csv(data["records"], output_dir)
    write_summary_json(data, output_dir)
    make_plots(data, output_dir, dpi=args.dpi)

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
