#!/usr/bin/env python3
"""Estimate fast-pass cost and runtime from a tiny calibration run.

The script can either:
1) run a tiny calibration sample via run_fast_pass_preprocessed_sample200.sh, or
2) reuse an existing calibration output folder.

Then it scales token/runtime measurements to a target sample size and prints a
breakdown by stage/system/provider and totals.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_DATASETS = [
    ("ramdocs", "ramdocs_test.json"),
    ("tcb_ramdocs", "tcb_ramdocs_test.json"),
    ("confact", "confact/humc_processed_cleaned.json"),
    ("tcb_confact", "confact/tcb_humc_processed_cleaned.json"),
]

# Per-1M token rates in USD, keyed by model-id prefix.
DEFAULT_RATES = {
    "gpt-5.4-nano-2026-03-17": (0.20, 1.25),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "gpt-5.4-mini": (0.75, 4.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class DatasetStat:
    name: str
    total: int
    contentious_ratio: float


@dataclass
class GroupAgg:
    stage: str
    system: str
    provider: str
    model: str
    observed_instances: float = 0.0
    observed_input_tokens: float = 0.0
    observed_output_tokens: float = 0.0
    observed_runtime_units: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate fast-pass cost/runtime from a tiny calibration run."
    )
    parser.add_argument(
        "--runner-script",
        default="./run_fast_pass_preprocessed_sample200.sh",
        help="Path to the fast-pass shell runner script.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/sample200_gemini_preprocess",
        help="Folder containing source datasets used by the runner.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[f"{k}:{v}" for k, v in DEFAULT_DATASETS],
        help="Dataset mapping entries benchmark:file, e.g. ramdocs:ramdocs_test.json",
    )
    parser.add_argument(
        "--target-sample-size",
        type=int,
        default=200,
        help="Target sample size for final estimate.",
    )
    parser.add_argument(
        "--tiny-sample-size",
        type=int,
        default=1,
        help="Tiny calibration sample size.",
    )
    parser.add_argument(
        "--target-judge-rounds",
        type=int,
        default=3,
        help="Target debate rounds.",
    )
    parser.add_argument(
        "--tiny-judge-rounds",
        type=int,
        default=1,
        help="Debate rounds used in tiny calibration run.",
    )
    parser.add_argument(
        "--judge-suite",
        choices=["openai", "anthropic", "both"],
        default="both",
        help="Judge suite to estimate.",
    )
    parser.add_argument(
        "--judge-only-on-contention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether MADAM/EAJP only run on contentious instances.",
    )
    parser.add_argument(
        "--disable-cache-for-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set LLM_CACHE_ENABLED=0 while running calibration for fresh-run estimates.",
    )
    parser.add_argument(
        "--calibration-output-root",
        default="",
        help="Output folder for calibration run (auto-generated if empty).",
    )
    parser.add_argument(
        "--reuse-output-root",
        default="",
        help="Skip calibration run and reuse this existing output folder.",
    )
    parser.add_argument(
        "--rate",
        action="append",
        default=[],
        help=(
            "Override/add token pricing as model_prefix=input_rate,output_rate in USD/1M tokens; "
            "repeatable."
        ),
    )
    parser.add_argument(
        "--extra-runner-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to runner script (repeatable).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write full estimate report JSON.",
    )
    return parser.parse_args()


def parse_dataset_entries(entries: Iterable[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid --datasets entry: {entry!r} (expected benchmark:file)")
        benchmark, file_name = entry.split(":", 1)
        benchmark = benchmark.strip()
        file_name = file_name.strip()
        if not benchmark or not file_name:
            raise ValueError(f"Invalid --datasets entry: {entry!r}")
        out.append((benchmark, file_name))
    return out


def parse_rates(overrides: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    rates: Dict[str, Tuple[float, float]] = dict(DEFAULT_RATES)
    for row in overrides:
        if "=" not in row:
            raise ValueError(f"Invalid --rate value: {row!r}")
        model_prefix, payload = row.split("=", 1)
        model_prefix = model_prefix.strip()
        parts = [p.strip() for p in payload.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid --rate payload: {row!r}")
        in_rate = float(parts[0])
        out_rate = float(parts[1])
        rates[model_prefix] = (in_rate, out_rate)
    return rates


def detect_provider(suite_name: str, model: str) -> str:
    s = (suite_name or "").lower()
    m = (model or "").lower()
    if s.startswith("openai_") or m.startswith("gpt-") or m.startswith("o"):
        return "openai"
    if s.startswith("anthropic_") or m.startswith("claude-"):
        return "anthropic"
    return "unknown"


def stage_name(system: str) -> str:
    if system == "standard_rag":
        return "Standard RAG"
    if system == "madam_rag":
        return "MADAM-RAG"
    if system == "eajp_scorecard_only":
        return "EAJP Scorecard-Only"
    if system == "eajp":
        return "EAJP"
    return system


def best_rate_for_model(model: str, rates: Dict[str, Tuple[float, float]]) -> Tuple[float, float] | None:
    model_l = (model or "").lower()
    best_key = ""
    best_val: Tuple[float, float] | None = None
    for key, val in rates.items():
        k = key.lower()
        if model_l.startswith(k) and len(k) > len(best_key):
            best_key = k
            best_val = val
    return best_val


def dataset_stats(dataset_root: Path, datasets: List[Tuple[str, str]]) -> List[DatasetStat]:
    stats: List[DatasetStat] = []
    for bench, file_name in datasets:
        path = dataset_root / file_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Dataset is not a JSON list: {path}")
        total = len(payload)
        contentious = 0
        for inst in payload:
            if not isinstance(inst, dict):
                continue
            docs = inst.get("documents", [])
            answers = {
                (d.get("supports_answer") or "").strip().lower()
                for d in docs
                if isinstance(d, dict) and (d.get("supports_answer") or "").strip()
            }
            if len(answers) > 1:
                contentious += 1
        ratio = (contentious / total) if total else 0.0
        stats.append(DatasetStat(name=bench, total=total, contentious_ratio=ratio))
    return stats


def run_calibration(args: argparse.Namespace, output_root: Path) -> float:
    runner_path = Path(args.runner_script)
    if not runner_path.is_absolute():
        runner_path = (Path.cwd() / runner_path).resolve()

    cmd = [
        str(runner_path),
        "--sample-size",
        str(args.tiny_sample_size),
        "--judge-rounds",
        str(args.tiny_judge_rounds),
        "--judge-suite",
        args.judge_suite,
        "--output-root",
        str(output_root),
    ]
    for item in args.extra_runner_arg:
        cmd.extend([item])

    env = dict(os.environ)
    if args.disable_cache_for_calibration:
        env["LLM_CACHE_ENABLED"] = "0"

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    return time.perf_counter() - t0


def load_calibration_groups(output_root: Path) -> Dict[Tuple[str, str, str], GroupAgg]:
    groups: Dict[Tuple[str, str, str], GroupAgg] = {}

    for details_path in output_root.rglob("*_details.json"):
        suite_name = details_path.parent.name
        payload = json.loads(details_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue

        for trial in payload:
            if not isinstance(trial, dict):
                continue
            system = str(trial.get("system", ""))
            model = str(trial.get("model", ""))
            provider = detect_provider(suite_name, model)
            key = (system, provider, model)
            if key not in groups:
                groups[key] = GroupAgg(
                    stage=stage_name(system),
                    system=system,
                    provider=provider,
                    model=model,
                )

            md = trial.get("metadata") or {}
            tele = md.get("telemetry") or {}
            usage = tele.get("usage") or {}

            in_tok = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            out_tok = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            runtime_units = md.get("runtime_seconds", 0) or 0

            groups[key].observed_instances += 1.0
            groups[key].observed_input_tokens += float(in_tok)
            groups[key].observed_output_tokens += float(out_tok)
            groups[key].observed_runtime_units += float(runtime_units)

    return groups


def expected_instance_totals(
    stats: List[DatasetStat],
    sample_size: int,
    judge_suite: str,
    judge_only_on_contention: bool,
) -> Dict[Tuple[str, str], float]:
    standard_total = 0.0
    judge_total = 0.0
    for st in stats:
        n = float(min(sample_size, st.total))
        standard_total += n
        if judge_only_on_contention:
            judge_total += n * st.contentious_ratio
        else:
            judge_total += n

    totals: Dict[Tuple[str, str], float] = {
        ("standard_rag", "openai"): standard_total,
    }
    if judge_suite in {"openai", "both"}:
        totals[("madam_rag", "openai")] = judge_total
        totals[("eajp", "openai")] = judge_total
    if judge_suite in {"anthropic", "both"}:
        totals[("madam_rag", "anthropic")] = judge_total
        totals[("eajp", "anthropic")] = judge_total
    return totals


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_num(x: float) -> str:
    return f"{x:,.0f}"


def fmt_hours(x: float) -> str:
    return f"{x:,.2f}h"


def main() -> int:
    args = parse_args()

    if args.target_sample_size < 1 or args.tiny_sample_size < 1:
        raise ValueError("Sample sizes must be >= 1")

    datasets = parse_dataset_entries(args.datasets)
    rates = parse_rates(args.rate)
    dataset_root = Path(args.dataset_root)

    stats = dataset_stats(dataset_root, datasets)

    if args.reuse_output_root:
        output_root = Path(args.reuse_output_root)
        wall_seconds_observed = 0.0
    else:
        if args.calibration_output_root:
            output_root = Path(args.calibration_output_root)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_root = Path(f"results/estimate_calibration_{ts}")
        wall_seconds_observed = run_calibration(args, output_root)

    groups = load_calibration_groups(output_root)
    if not groups:
        raise RuntimeError(f"No calibration detail files found under {output_root}")

    observed_expected = expected_instance_totals(
        stats=stats,
        sample_size=args.tiny_sample_size,
        judge_suite=args.judge_suite,
        judge_only_on_contention=args.judge_only_on_contention,
    )
    target_expected = expected_instance_totals(
        stats=stats,
        sample_size=args.target_sample_size,
        judge_suite=args.judge_suite,
        judge_only_on_contention=args.judge_only_on_contention,
    )

    rows = []
    total_est_input = 0.0
    total_est_output = 0.0
    total_est_cost = 0.0

    # Runtime weights for stage breakdown.
    runtime_weight_total = 0.0
    observed_runtime_weight_total = 0.0

    for group in groups.values():
        key = (group.system, group.provider)
        obs_instances_expected = observed_expected.get(key, 0.0)
        tgt_instances_expected = target_expected.get(key, 0.0)

        # Use observed instance counts for stability, but scale by expected tiny->target ratio.
        if obs_instances_expected > 0:
            scale = tgt_instances_expected / obs_instances_expected
        else:
            scale = 0.0

        # MADAM/EAJP are debate-round dependent, so scale by round ratio.
        if group.system in {"madam_rag", "eajp"} and args.tiny_judge_rounds > 0:
            scale *= (args.target_judge_rounds / args.tiny_judge_rounds)

        est_input = group.observed_input_tokens * scale
        est_output = group.observed_output_tokens * scale

        rates_for_model = best_rate_for_model(group.model, rates)
        if rates_for_model is None:
            est_cost = float("nan")
            in_rate = out_rate = float("nan")
        else:
            in_rate, out_rate = rates_for_model
            est_cost = (est_input / 1_000_000.0) * in_rate + (est_output / 1_000_000.0) * out_rate

        # Runtime weight: prefer runtime telemetry, fall back to token volume.
        observed_weight = group.observed_runtime_units
        if observed_weight <= 0:
            observed_weight = (group.observed_input_tokens + group.observed_output_tokens) / 1_000_000.0
        runtime_weight = observed_weight * scale

        observed_runtime_weight_total += observed_weight
        runtime_weight_total += runtime_weight

        rows.append(
            {
                "stage": group.stage,
                "system": group.system,
                "provider": group.provider,
                "suite_label": "mixed",
                "model": group.model,
                "observed_instances": group.observed_instances,
                "estimated_instances": tgt_instances_expected,
                "estimated_input_tokens": est_input,
                "estimated_output_tokens": est_output,
                "input_rate_per_mtok": in_rate,
                "output_rate_per_mtok": out_rate,
                "estimated_cost_usd": est_cost,
                "runtime_weight": runtime_weight,
            }
        )

        total_est_input += est_input
        total_est_output += est_output
        if est_cost == est_cost:  # NaN-safe check
            total_est_cost += est_cost

    # Wall time extrapolation uses tiny calibration wall time when available.
    if wall_seconds_observed > 0:
        # Scale by runtime-weight ratio so round changes are reflected.
        scale_total = (
            runtime_weight_total / observed_runtime_weight_total
            if observed_runtime_weight_total > 0
            else 0.0
        )
        est_wall_seconds_total = wall_seconds_observed * scale_total
    else:
        est_wall_seconds_total = float("nan")

    # Assign wall time by runtime weight share.
    for row in rows:
        if est_wall_seconds_total == est_wall_seconds_total and runtime_weight_total > 0:
            row["estimated_wall_seconds"] = est_wall_seconds_total * (
                row["runtime_weight"] / runtime_weight_total
            )
        else:
            row["estimated_wall_seconds"] = float("nan")

    rows.sort(key=lambda r: (r["stage"], r["provider"], r["model"]))

    # Provider totals
    provider_totals: Dict[str, Dict[str, float]] = {}
    for r in rows:
        p = r["provider"]
        if p not in provider_totals:
            provider_totals[p] = {
                "cost": 0.0,
                "input": 0.0,
                "output": 0.0,
                "wall": 0.0,
            }
        provider_totals[p]["input"] += r["estimated_input_tokens"]
        provider_totals[p]["output"] += r["estimated_output_tokens"]
        if r["estimated_cost_usd"] == r["estimated_cost_usd"]:
            provider_totals[p]["cost"] += r["estimated_cost_usd"]
        if r["estimated_wall_seconds"] == r["estimated_wall_seconds"]:
            provider_totals[p]["wall"] += r["estimated_wall_seconds"]

    # Stage totals
    stage_totals: Dict[str, Dict[str, float]] = {}
    for r in rows:
        s = r["stage"]
        if s not in stage_totals:
            stage_totals[s] = {"cost": 0.0, "input": 0.0, "output": 0.0, "wall": 0.0}
        stage_totals[s]["input"] += r["estimated_input_tokens"]
        stage_totals[s]["output"] += r["estimated_output_tokens"]
        if r["estimated_cost_usd"] == r["estimated_cost_usd"]:
            stage_totals[s]["cost"] += r["estimated_cost_usd"]
        if r["estimated_wall_seconds"] == r["estimated_wall_seconds"]:
            stage_totals[s]["wall"] += r["estimated_wall_seconds"]

    print("\nCalibration")
    print(f"- output_root: {output_root}")
    if wall_seconds_observed > 0:
        print(f"- observed_wall_time: {wall_seconds_observed:.2f}s")
    else:
        print("- observed_wall_time: reused existing output (not measured by this run)")

    print("\nDataset stats")
    for st in stats:
        print(
            f"- {st.name}: total={st.total}, contentious_ratio={st.contentious_ratio:.4f}"
        )

    print("\nEstimated breakdown by stage/system/provider/model")
    header = (
        f"{'stage':<14} {'system':<12} {'provider':<10} {'model':<36} "
        f"{'est_in_tok':>12} {'est_out_tok':>12} {'est_cost':>10} {'est_wall_h':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        est_wall_h = (r["estimated_wall_seconds"] / 3600.0) if r["estimated_wall_seconds"] == r["estimated_wall_seconds"] else float("nan")
        est_wall_s = fmt_hours(est_wall_h) if est_wall_h == est_wall_h else "n/a"
        cost_s = fmt_money(r["estimated_cost_usd"]) if r["estimated_cost_usd"] == r["estimated_cost_usd"] else "n/a"
        print(
            f"{r['stage']:<14} {r['system']:<12} {r['provider']:<10} {r['model']:<36} "
            f"{fmt_num(r['estimated_input_tokens']):>12} {fmt_num(r['estimated_output_tokens']):>12} {cost_s:>10} {est_wall_s:>10}"
        )

    print("\nTotals by provider")
    for provider, v in sorted(provider_totals.items()):
        print(
            f"- {provider}: input={fmt_num(v['input'])}, output={fmt_num(v['output'])}, "
            f"cost={fmt_money(v['cost'])}, wall={fmt_hours(v['wall']/3600.0) if v['wall'] == v['wall'] else 'n/a'}"
        )

    print("\nTotals by stage")
    for stage, v in sorted(stage_totals.items()):
        print(
            f"- {stage}: input={fmt_num(v['input'])}, output={fmt_num(v['output'])}, "
            f"cost={fmt_money(v['cost'])}, wall={fmt_hours(v['wall']/3600.0) if v['wall'] == v['wall'] else 'n/a'}"
        )

    est_total_wall_h = (est_wall_seconds_total / 3600.0) if est_wall_seconds_total == est_wall_seconds_total else float("nan")
    print("\nGrand totals")
    print(f"- input_tokens: {fmt_num(total_est_input)}")
    print(f"- output_tokens: {fmt_num(total_est_output)}")
    print(f"- estimated_cost_usd: {fmt_money(total_est_cost)}")
    print(f"- estimated_wall_time: {fmt_hours(est_total_wall_h) if est_total_wall_h == est_total_wall_h else 'n/a'}")

    if args.json_out:
        report = {
            "config": {
                "runner_script": args.runner_script,
                "dataset_root": str(dataset_root),
                "datasets": datasets,
                "target_sample_size": args.target_sample_size,
                "tiny_sample_size": args.tiny_sample_size,
                "target_judge_rounds": args.target_judge_rounds,
                "tiny_judge_rounds": args.tiny_judge_rounds,
                "judge_suite": args.judge_suite,
                "judge_only_on_contention": args.judge_only_on_contention,
                "disable_cache_for_calibration": args.disable_cache_for_calibration,
                "calibration_output_root": str(output_root),
                "observed_wall_seconds": wall_seconds_observed,
            },
            "dataset_stats": [st.__dict__ for st in stats],
            "rows": rows,
            "provider_totals": provider_totals,
            "stage_totals": stage_totals,
            "grand_totals": {
                "input_tokens": total_est_input,
                "output_tokens": total_est_output,
                "cost_usd": total_est_cost,
                "wall_seconds": est_wall_seconds_total,
            },
        }
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
