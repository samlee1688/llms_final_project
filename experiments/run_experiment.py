#!/usr/bin/env python3
"""Main experiment runner for the tone-bias study.

Runs Standard RAG, MADAM-RAG, and EAJP on AmbigDocs, FaithEval, and
Tone-Controlled Benchmark style datasets with configurable OpenAI and
Anthropic judge suites.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import random
import re
import sys
import threading
import time

# Ensure repo root is on the Python path so that ``import config`` and
# ``import src.*`` resolve correctly regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.data_loader import load_benchmark, QueryInstance
from src.models import LLMClient
from src.standard_rag import StandardRAG
from src.madam_rag import MadamRAG
from src.madam_rag import EvidenceAnchoredAggregator
from src.evaluation import (
    classify_answer,
    compute_metrics,
    print_metrics,
    save_results,
    TrialResult,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _build_client(provider: str, model: str | None = None) -> LLMClient:
    return LLMClient(provider=provider, model=model)


def _suite_label(provider: str, model: str) -> str:
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(model or "").strip())
    if not model_slug:
        model_slug = "default"
    return f"{provider}_{model_slug}"


def _suite_configs(
    judge_suite: str,
    openai_judge_model: str,
    anthropic_judge_model: str,
) -> list[tuple[str, LLMClient]]:
    """Return the judge model configurations to evaluate."""
    openai_client = _build_client("openai", openai_judge_model)
    anthropic_client = _build_client("anthropic", anthropic_judge_model)

    if judge_suite == "openai":
        return [(_suite_label("openai", openai_client.model), openai_client)]
    if judge_suite == "anthropic":
        return [(_suite_label("anthropic", anthropic_client.model), anthropic_client)]
    return [
        (_suite_label("openai", openai_client.model), openai_client),
        (_suite_label("anthropic", anthropic_client.model), anthropic_client),
    ]


def _trial_to_dict(t: TrialResult) -> dict:
    return {
        "query_id": t.query_id,
        "system": t.system,
        "model": t.model,
        "classification": t.classification,
        "final_answer": t.final_answer,
        "correct_answer": t.correct_answer,
        "metadata": t.metadata,
    }


def _trial_from_dict(d: dict) -> TrialResult:
    return TrialResult(
        query_id=str(d.get("query_id", "")),
        system=str(d.get("system", "")),
        model=str(d.get("model", "")),
        classification=str(d.get("classification", "unclear")),
        final_answer=str(d.get("final_answer", "")),
        correct_answer=str(d.get("correct_answer", "")),
        metadata=dict(d.get("metadata", {})),
    )


def _partial_path(output_dir: str, tag: str) -> str:
    return os.path.join(output_dir, f"{tag}_partial.jsonl")


def _load_partial(path: str) -> dict[str, TrialResult]:
    if not os.path.exists(path):
        return {}
    done: dict[str, TrialResult] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                trial = _trial_from_dict(data)
                done[trial.query_id] = trial
            except Exception:
                continue
    return done


def _append_partial(path: str, trial: TrialResult, lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_trial_to_dict(trial), ensure_ascii=False) + "\n")


def _ordered_trials(
    instances: list[QueryInstance],
    trial_map: dict[str, TrialResult],
) -> list[TrialResult]:
    ordered: list[TrialResult] = []
    for inst in instances:
        trial = trial_map.get(inst.id)
        if trial is not None:
            ordered.append(trial)
    return ordered


def _sample_instances(
    instances: list[QueryInstance],
    limit: int,
    seed: int,
) -> list[QueryInstance]:
    if limit <= 0 or limit >= len(instances):
        return instances
    rng = random.Random(seed)
    idx = list(range(len(instances)))
    rng.shuffle(idx)
    keep = sorted(idx[:limit])
    return [instances[i] for i in keep]


def _error_trial(
    *,
    instance: QueryInstance,
    system_name: str,
    model_name: str,
    benchmark_name: str,
    error: Exception,
) -> TrialResult:
    return TrialResult(
        query_id=instance.id,
        system=system_name,
        model=model_name,
        classification="unclear",
        final_answer="",
        correct_answer=instance.correct_answer,
        metadata={
            "benchmark": benchmark_name,
            "tone_controlled": bool(instance.metadata.get("tone_controlled", False)),
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def _is_contentious_instance(instance: QueryInstance) -> bool:
    supported = {
        (d.supports_answer or "").strip().lower()
        for d in instance.documents
        if (d.supports_answer or "").strip()
    }
    return len(supported) > 1


def _split_contention(
    instances: list[QueryInstance],
) -> tuple[list[QueryInstance], list[QueryInstance]]:
    contentious: list[QueryInstance] = []
    uncontested: list[QueryInstance] = []
    for inst in instances:
        if _is_contentious_instance(inst):
            contentious.append(inst)
        else:
            uncontested.append(inst)
    return contentious, uncontested


def _evaluate_system(
    *,
    system_name: str,
    instances: list[QueryInstance],
    run_one,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
    benchmark_name: str,
    model_name: str,
) -> list[TrialResult]:
    partial = _partial_path(output_dir, partial_tag)
    existing = _load_partial(partial) if resume else {}
    pending = [inst for inst in instances if inst.id not in existing]
    logger.info(
        "%s: %d total, %d already complete, %d pending",
        system_name,
        len(instances),
        len(existing),
        len(pending),
    )

    if not pending:
        return _ordered_trials(instances, existing)

    write_lock = threading.Lock()
    completed = 0
    total_pending = len(pending)

    def _handle_trial(trial: TrialResult) -> None:
        nonlocal completed
        existing[trial.query_id] = trial
        _append_partial(partial, trial, write_lock)
        completed += 1
        if completed % 25 == 0 or completed == total_pending:
            logger.info(
                "%s progress: %d/%d pending complete",
                system_name,
                completed,
                total_pending,
            )

    if max_workers <= 1:
        for inst in pending:
            try:
                trial = run_one(inst)
            except Exception as exc:
                logger.exception("%s failed for query_id=%s", system_name, inst.id)
                trial = _error_trial(
                    instance=inst,
                    system_name=system_name,
                    model_name=model_name,
                    benchmark_name=benchmark_name,
                    error=exc,
                )
            _handle_trial(trial)
    else:
        logger.info("%s: running with %d workers", system_name, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one, inst): inst for inst in pending}
            for fut in as_completed(futures):
                inst = futures[fut]
                try:
                    trial = fut.result()
                except Exception as exc:
                    logger.exception("%s failed for query_id=%s", system_name, inst.id)
                    trial = _error_trial(
                        instance=inst,
                        system_name=system_name,
                        model_name=model_name,
                        benchmark_name=benchmark_name,
                        error=exc,
                    )
                _handle_trial(trial)

    return _ordered_trials(instances, existing)


def _run_standard_rag(
    instances: list[QueryInstance],
    client: LLMClient,
    benchmark_name: str,
    output_dir: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    pipeline = StandardRAG(client)

    def _run_one(inst: QueryInstance) -> TrialResult:
        t0 = time.perf_counter()
        out = pipeline.answer(inst)
        elapsed = time.perf_counter() - t0
        classification = classify_answer(out.final_answer, inst)
        telemetry = {
            "runtime_seconds": round(elapsed, 4),
            "usage": out.usage or {},
            "provider": client.provider,
            "model": out.model,
        }
        return TrialResult(
            query_id=inst.id,
            system="standard_rag",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "telemetry": telemetry,
                **(out.metadata or {}),
            },
        )

    return _evaluate_system(
        system_name="standard_rag",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag="standard_rag",
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=client.model,
    )


def _run_madam_rag(
    instances: list[QueryInstance],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    num_rounds: int | None,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
    max_parallel_agents: int,
    compact_doc_chars: int,
    argument_summary_chars: int,
    early_stop_patience: int,
    include_full_document_in_debate: bool,
) -> list[TrialResult]:
    pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        max_parallel_agents=max_parallel_agents,
        compact_doc_chars=compact_doc_chars,
        argument_summary_chars=argument_summary_chars,
        early_stop_patience=early_stop_patience,
        include_full_document_in_debate=include_full_document_in_debate,
    )
    def _run_one(inst: QueryInstance) -> TrialResult:
        out = pipeline.answer(inst)
        classification = classify_answer(out.final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system="madam_rag",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "telemetry": {
                    "usage": out.usage or {},
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": out.model,
                },
                **(out.metadata or {}),
            },
        )

    return _evaluate_system(
        system_name="madam_rag",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_eajp(
    instances: list[QueryInstance],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    num_rounds: int | None,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
    max_parallel_agents: int,
    compact_doc_chars: int,
    argument_summary_chars: int,
    early_stop_patience: int,
    include_full_document_in_debate: bool,
    eajp_short_circuit: bool,
) -> list[TrialResult]:
    pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=EvidenceAnchoredAggregator(
            judge_client or agent_client,
            short_circuit_on_confident_scorecard=eajp_short_circuit,
            compact_doc_chars=compact_doc_chars,
        ),
        max_parallel_agents=max_parallel_agents,
        compact_doc_chars=compact_doc_chars,
        argument_summary_chars=argument_summary_chars,
        early_stop_patience=early_stop_patience,
        include_full_document_in_debate=include_full_document_in_debate,
    )
    def _run_one(inst: QueryInstance) -> TrialResult:
        out = pipeline.answer(inst)
        classification = classify_answer(out.final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system="eajp",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "judge_mode": "eajp",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "telemetry": {
                    "usage": out.usage or {},
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": out.model,
                },
                **(out.metadata or {}),
            },
        )

    return _evaluate_system(
        system_name="eajp",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _log_stage_models(
    *,
    systems: list[str],
    standard_client: LLMClient,
    madam_agent_client: LLMClient,
    eajp_agent_client: LLMClient,
    suite_configs: list[tuple[str, LLMClient]],
) -> None:
    logger.info(
        "Model plan: standard_rag -> provider=%s model=%s",
        standard_client.provider,
        standard_client.model,
    )
    for suite_name, judge_client in suite_configs:
        if "madam_rag" in systems:
            logger.info(
                "Model plan: madam_rag [%s] -> agents=(%s, %s), judge=(%s, %s)",
                suite_name,
                madam_agent_client.provider,
                madam_agent_client.model,
                judge_client.provider,
                judge_client.model,
            )
        if "eajp" in systems:
            logger.info(
                "Model plan: eajp [%s] -> agents=(%s, %s), evidence_judge=(%s, %s)",
                suite_name,
                eajp_agent_client.provider,
                eajp_agent_client.model,
                judge_client.provider,
                judge_client.model,
            )


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run tone-bias RAG experiments."
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to the benchmark JSON file.",
    )
    p.add_argument(
        "--benchmark",
        choices=["custom", "ambigdocs", "faitheval", "tcb"],
        default="custom",
        help="Benchmark family used to load and filter the dataset.",
    )
    p.add_argument(
        "--systems", nargs="+",
        choices=["standard_rag", "madam_rag", "eajp"],
        default=["standard_rag", "madam_rag", "eajp"],
        help="Which systems to evaluate.",
    )
    p.add_argument(
        "--agent-provider",
        choices=["openai", "anthropic"],
        default=config.AGENT_PROVIDER,
        help="LLM provider for document agents and standard RAG.",
    )
    p.add_argument(
        "--agent-model",
        default=config.AGENT_MODEL,
        help="Model used for document agents and standard RAG.",
    )
    p.add_argument(
        "--standard-rag-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="Provider for standard RAG only (defaults to --agent-provider).",
    )
    p.add_argument(
        "--standard-rag-model",
        default=None,
        help="Model for standard RAG only (defaults to --agent-model).",
    )
    p.add_argument(
        "--madam-agent-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="Provider for MADAM-RAG agents only (defaults to --agent-provider).",
    )
    p.add_argument(
        "--madam-agent-model",
        default=None,
        help="Model for MADAM-RAG agents only (defaults to --agent-model).",
    )
    p.add_argument(
        "--eajp-agent-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="Provider for EAJP agents only (defaults to --agent-provider).",
    )
    p.add_argument(
        "--eajp-agent-model",
        default=None,
        help="Model for EAJP agents only (defaults to --agent-model).",
    )
    p.add_argument(
        "--openai-judge-model",
        default=config.OPENAI_MODEL,
        help="Model for OpenAI judge suite.",
    )
    p.add_argument(
        "--anthropic-judge-model",
        default=config.ANTHROPIC_MODEL,
        help="Model for Anthropic judge suite.",
    )
    p.add_argument(
        "--judge-suite",
        choices=["openai", "anthropic", "both"],
        default="both",
        help="Judge-model suite to run for MADAM-RAG and EAJP.",
    )
    p.add_argument(
        "--judge-rounds", type=int, default=None,
        help="Number of MADAM-RAG / EAJP debate rounds (default from config).",
    )
    p.add_argument(
        "--madam-max-parallel-agents", type=int, default=4,
        help="Max concurrent document-agent calls per MADAM/EAJP round.",
    )
    p.add_argument(
        "--madam-compact-doc-chars", type=int, default=1200,
        help="Character budget for per-agent document excerpt in debate prompts.",
    )
    p.add_argument(
        "--madam-argument-summary-chars", type=int, default=240,
        help="Character budget for each opposing argument passed into debate rounds.",
    )
    p.add_argument(
        "--madam-early-stop-patience", type=int, default=1,
        help="Consecutive stable-consensus rounds required before early stopping.",
    )
    p.add_argument(
        "--madam-include-full-document-in-debate", action=argparse.BooleanOptionalAction, default=False,
        help="Include full document every debate round (default: compact excerpt only).",
    )
    p.add_argument(
        "--eajp-short-circuit", action=argparse.BooleanOptionalAction, default=False,
        help="Skip EAJP final pass when scorecard winner is confident (default: off for strict EAJP flow).",
    )
    p.add_argument(
        "--judge-only-on-contention", action=argparse.BooleanOptionalAction, default=False,
        help="Run MADAM/EAJP only on contentious instances (fallback to standard result otherwise).",
    )
    p.add_argument(
        "--max-workers", type=int, default=int(os.getenv("EXP_MAX_WORKERS", "1")),
        help="Max per-system worker threads for instance-level parallelism.",
    )
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from partial JSONL checkpoints if available.",
    )
    p.add_argument(
        "--limit-instances", type=int, default=0,
        help="Randomly sample this many instances after loading (0 = all).",
    )
    p.add_argument(
        "--sample-seed", type=int, default=42,
        help="Random seed for sampled subsets.",
    )
    p.add_argument(
        "--fast-mode", action="store_true",
        help="Enable sampled fast benchmark mode.",
    )
    p.add_argument(
        "--fast-mode-size", type=int, default=120,
        help="Sample size used when --fast-mode is enabled and --limit-instances is not set.",
    )
    p.add_argument(
        "--output", default=None,
        help="Directory for result files. Defaults to results/<benchmark>.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    # Load data
    instances = load_benchmark(args.dataset, args.benchmark)
    if args.fast_mode and args.limit_instances <= 0:
        args.limit_instances = args.fast_mode_size
    if args.limit_instances > 0:
        before = len(instances)
        instances = _sample_instances(instances, args.limit_instances, args.sample_seed)
        logger.info(
            "Sampling enabled: %d -> %d instances (seed=%d)",
            before,
            len(instances),
            args.sample_seed,
        )
    logger.info("Loaded %d instances from %s", len(instances), args.dataset)
    if not instances:
        raise ValueError("Dataset is empty after filtering/sampling.")

    # Build per-system agent clients so model/provider can be explicit per stage.
    standard_client = _build_client(
        args.standard_rag_provider or args.agent_provider,
        args.standard_rag_model or args.agent_model,
    )
    madam_agent_client = _build_client(
        args.madam_agent_provider or args.agent_provider,
        args.madam_agent_model or args.agent_model,
    )
    eajp_agent_client = _build_client(
        args.eajp_agent_provider or args.agent_provider,
        args.eajp_agent_model or args.agent_model,
    )
    logger.info("Standard RAG client: %s", standard_client)
    logger.info("MADAM agent client: %s", madam_agent_client)
    logger.info("EAJP agent client: %s", eajp_agent_client)

    output_dir = args.output or os.path.join(config.RESULTS_DIR, args.benchmark)
    os.makedirs(output_dir, exist_ok=True)

    suite_configs = _suite_configs(
        args.judge_suite,
        args.openai_judge_model,
        args.anthropic_judge_model,
    )
    _log_stage_models(
        systems=args.systems,
        standard_client=standard_client,
        madam_agent_client=madam_agent_client,
        eajp_agent_client=eajp_agent_client,
        suite_configs=suite_configs,
    )
    max_workers = max(1, args.max_workers)
    logger.info("Execution settings: max_workers=%d, resume=%s", max_workers, args.resume)

    # Standard RAG does not depend on the judge suite, so run it once.
    standard_trials_by_id: dict[str, TrialResult] = {}
    if "standard_rag" in args.systems:
        logger.info("=== Running standard_rag (suite-independent) ===")
        trials = _run_standard_rag(
            instances,
            standard_client,
            benchmark_name=args.benchmark,
            output_dir=output_dir,
            max_workers=max_workers,
            resume=args.resume,
        )
        standard_trials_by_id = {t.query_id: t for t in trials}
        metrics = compute_metrics(trials, "standard_rag", standard_client.model)
        print_metrics(metrics)
        path = save_results(metrics, output_dir, tag="standard_rag")
        logger.info("Saved standard_rag results to %s", path)

    # Run requested systems for each judge suite.
    for suite_name, judge_client in suite_configs:
        suite_output_dir = os.path.join(output_dir, suite_name)
        os.makedirs(suite_output_dir, exist_ok=True)
        logger.info("=== Judge suite: %s (%s) ===", suite_name, judge_client)

        for system_name in args.systems:
            if system_name == "standard_rag":
                continue
            logger.info("--- Running %s ---", system_name)

            if system_name == "madam_rag":
                run_instances = instances
                fallback_trials: list[TrialResult] = []
                if args.judge_only_on_contention:
                    contentious, uncontested = _split_contention(instances)
                    run_instances = contentious
                    for inst in uncontested:
                        base = standard_trials_by_id.get(inst.id)
                        if base is not None:
                            fallback_trials.append(
                                TrialResult(
                                    query_id=inst.id,
                                    system="madam_rag",
                                    model=base.model,
                                    classification=base.classification,
                                    final_answer=base.final_answer,
                                    correct_answer=inst.correct_answer,
                                    metadata={
                                        **dict(base.metadata or {}),
                                        "judge_skipped_non_contentious": True,
                                        "judge_suite": suite_name,
                                    },
                                )
                            )
                        else:
                            fallback_trials.append(
                                TrialResult(
                                    query_id=inst.id,
                                    system="madam_rag",
                                    model=judge_client.model,
                                    classification="unclear",
                                    final_answer="",
                                    correct_answer=inst.correct_answer,
                                    metadata={
                                        "judge_skipped_non_contentious": True,
                                        "judge_suite": suite_name,
                                    },
                                )
                            )
                trials = _run_madam_rag(
                    run_instances,
                    madam_agent_client,
                    judge_client=judge_client,
                    num_rounds=args.judge_rounds,
                    benchmark_name=args.benchmark,
                    output_dir=suite_output_dir,
                    partial_tag=f"{suite_name}_madam_rag",
                    max_workers=max_workers,
                    resume=args.resume,
                    max_parallel_agents=args.madam_max_parallel_agents,
                    compact_doc_chars=args.madam_compact_doc_chars,
                    argument_summary_chars=args.madam_argument_summary_chars,
                    early_stop_patience=args.madam_early_stop_patience,
                    include_full_document_in_debate=args.madam_include_full_document_in_debate,
                )
                if fallback_trials:
                    trials = _ordered_trials(instances, {t.query_id: t for t in (trials + fallback_trials)})
                model_name = judge_client.model
            elif system_name == "eajp":
                run_instances = instances
                fallback_trials = []
                if args.judge_only_on_contention:
                    contentious, uncontested = _split_contention(instances)
                    run_instances = contentious
                    for inst in uncontested:
                        base = standard_trials_by_id.get(inst.id)
                        if base is not None:
                            fallback_trials.append(
                                TrialResult(
                                    query_id=inst.id,
                                    system="eajp",
                                    model=base.model,
                                    classification=base.classification,
                                    final_answer=base.final_answer,
                                    correct_answer=inst.correct_answer,
                                    metadata={
                                        **dict(base.metadata or {}),
                                        "judge_skipped_non_contentious": True,
                                        "judge_suite": suite_name,
                                    },
                                )
                            )
                        else:
                            fallback_trials.append(
                                TrialResult(
                                    query_id=inst.id,
                                    system="eajp",
                                    model=judge_client.model,
                                    classification="unclear",
                                    final_answer="",
                                    correct_answer=inst.correct_answer,
                                    metadata={
                                        "judge_skipped_non_contentious": True,
                                        "judge_suite": suite_name,
                                    },
                                )
                            )
                trials = _run_eajp(
                    run_instances,
                    eajp_agent_client,
                    judge_client=judge_client,
                    num_rounds=args.judge_rounds,
                    benchmark_name=args.benchmark,
                    output_dir=suite_output_dir,
                    partial_tag=f"{suite_name}_eajp",
                    max_workers=max_workers,
                    resume=args.resume,
                    max_parallel_agents=args.madam_max_parallel_agents,
                    compact_doc_chars=args.madam_compact_doc_chars,
                    argument_summary_chars=args.madam_argument_summary_chars,
                    early_stop_patience=args.madam_early_stop_patience,
                    include_full_document_in_debate=args.madam_include_full_document_in_debate,
                    eajp_short_circuit=args.eajp_short_circuit,
                )
                if fallback_trials:
                    trials = _ordered_trials(instances, {t.query_id: t for t in (trials + fallback_trials)})
                model_name = judge_client.model
            else:
                raise ValueError(f"Unknown system: {system_name}")

            metrics = compute_metrics(trials, system_name, model_name)
            print_metrics(metrics)
            path = save_results(metrics, suite_output_dir, tag=system_name)
            logger.info("Saved %s results to %s", system_name, path)


if __name__ == "__main__":
    main()
