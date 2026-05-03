#!/usr/bin/env python3
"""Main experiment runner for the tone-bias study.

Runs Standard RAG, MADAM-RAG, EAJP scorecard-only, and EAJP on RAMDocs
and Tone-Controlled Benchmark style datasets with configurable
OpenAI and Anthropic judge suites.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import random
import re
import sys
import threading
import time
from typing import Any

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
from src.madam_rag import (
    EvidenceAnchoredAggregator,
    CredibilityHintAggregator,
    ReliabilityAwareAggregator,
    ChainOfThoughtAggregator,
)
from src.madam_rag import Aggregator
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


def _claim_from_text(text: str) -> str:
    if not text.strip():
        return ""
    return text.strip().splitlines()[0].strip()[:1000]


def _extract_cached_debate(trial: TrialResult) -> tuple[dict[str, str], dict[str, str], int] | None:
    cache = dict((trial.metadata or {}).get("debate_cache", {}) or {})
    final_arguments = cache.get("final_arguments")
    if not isinstance(final_arguments, dict) or not final_arguments:
        return None
    final_arguments_typed: dict[str, str] = {
        str(k): str(v) for k, v in final_arguments.items()
    }

    agent_claims_raw = cache.get("agent_claims", {})
    if isinstance(agent_claims_raw, dict):
        agent_claims = {str(k): str(v) for k, v in agent_claims_raw.items()}
    else:
        agent_claims = {}

    rounds_executed_raw = cache.get("rounds_executed", 0)
    try:
        rounds_executed = max(0, int(rounds_executed_raw))
    except Exception:
        rounds_executed = 0
    return final_arguments_typed, agent_claims, rounds_executed


def _extract_cached_round_arguments(trial: TrialResult) -> list[dict[str, str]]:
    cache = dict((trial.metadata or {}).get("debate_cache", {}) or {})
    rows = cache.get("round_arguments")
    if not isinstance(rows, list):
        return []
    typed: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            typed.append({})
            continue
        typed.append({str(k): str(v) for k, v in row.items()})
    return typed


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


def _write_round_ablation_summary_csv(output_dir: str, rows: list[dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "round_ablation_summary.csv")
    fieldnames = [
        "system",
        "round",
        "model",
        "total",
        "tone_controlled_total",
        "correct",
        "incorrect",
        "unclear",
        "tone_bias_rate",
        "accuracy",
        "preliminary_accuracy",
        "final_before_privilege_accuracy",
        "final_after_privilege_accuracy",
        "conflict_rate",
        "conflict_toward_correctness_rate",
        "conflict_away_from_correctness_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def _write_cross_suite_round_ablation_summary_csv(output_dir: str, rows: list[dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "round_ablation_summary_all_suites.csv")
    fieldnames = [
        "suite",
        "system",
        "round",
        "model",
        "total",
        "tone_controlled_total",
        "correct",
        "incorrect",
        "unclear",
        "tone_bias_rate",
        "accuracy",
        "preliminary_accuracy",
        "final_before_privilege_accuracy",
        "final_after_privilege_accuracy",
        "conflict_rate",
        "conflict_toward_correctness_rate",
        "conflict_away_from_correctness_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


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


def _winner_correctness(instance: QueryInstance, winner_id: str) -> bool | None:
    if not winner_id:
        return None
    for doc in instance.documents:
        if str(doc.id) != str(winner_id):
            continue
        supported = str(doc.supports_answer or "").strip().lower()
        if not supported:
            return None
        return supported == str(instance.correct_answer or "").strip().lower()
    return None


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


_CRED_WEIGHTS: dict[str, float] = {"high": 3.0, "medium": 2.0, "low": 1.0}


def _compute_credibility_vote(instance: QueryInstance) -> str:
    """Deterministic credibility-weighted majority vote over documents.

    Each document casts a weighted vote for its implied answer
    (doc.supports_answer), weighted by its credibility tier.
    Unknown tiers receive weight 1.5.  Returns the winning answer string,
    or empty string if no document has a supports_answer value.
    """
    votes: dict[str, float] = {}
    for doc in instance.documents:
        ans = (doc.supports_answer or "").strip().lower()
        if not ans:
            continue
        w = _CRED_WEIGHTS.get((doc.credibility or "").strip().lower(), 1.5)
        votes[ans] = votes.get(ans, 0.0) + w
    return max(votes, key=lambda k: votes[k]) if votes else ""


def _run_credibility_vote(
    instances: list[QueryInstance],
    benchmark_name: str,
    output_dir: str,
) -> list[TrialResult]:
    """Run deterministic credibility-weighted majority vote (no LLM call)."""
    results: list[TrialResult] = []
    for inst in instances:
        final_answer = _compute_credibility_vote(inst)
        classification = classify_answer(final_answer, inst)
        results.append(
            TrialResult(
                query_id=inst.id,
                system="credibility_vote",
                model="deterministic",
                classification=classification,
                final_answer=final_answer,
                correct_answer=inst.correct_answer,
                metadata={
                    "benchmark": benchmark_name,
                    "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                    "vote_model": "credibility_weighted_majority",
                },
            )
        )
    return results


def _compute_eajp_attribution_metrics(
    trials: list[TrialResult],
    instances_by_id: dict[str, QueryInstance],
) -> dict[str, Any]:
    prelim_total = 0
    prelim_correct = 0
    before_total = 0
    before_correct = 0
    after_total = 0
    after_correct = 0

    conflict_total = 0
    toward_total = 0
    away_total = 0

    for trial in trials:
        meta = dict(trial.metadata or {})
        judge_meta = dict(meta.get("judge_metadata", {}) or {})
        scorecard = dict(judge_meta.get("scorecard", {}) or {})
        prelim_id = str(scorecard.get("preliminary_winner_id", "")).strip() or str(judge_meta.get("preliminary_winner_id", "")).strip()
        before_id = str(judge_meta.get("final_winner_id_before_privilege", "")).strip()
        after_id = str(judge_meta.get("final_winner_id", "")).strip()

        inst = instances_by_id.get(trial.query_id)
        if inst is None:
            continue

        prelim_ok = _winner_correctness(inst, prelim_id)
        before_ok = _winner_correctness(inst, before_id)
        after_ok = _winner_correctness(inst, after_id)

        if prelim_ok is not None:
            prelim_total += 1
            prelim_correct += int(prelim_ok)
        if before_ok is not None:
            before_total += 1
            before_correct += int(before_ok)
        if after_ok is not None:
            after_total += 1
            after_correct += int(after_ok)

        if prelim_id and before_id and prelim_id != before_id and prelim_ok is not None and before_ok is not None:
            conflict_total += 1
            if (not prelim_ok) and before_ok:
                toward_total += 1
            if prelim_ok and (not before_ok):
                away_total += 1

    return {
        "preliminary_accuracy": round(_safe_rate(prelim_correct, prelim_total), 6),
        "final_before_privilege_accuracy": round(_safe_rate(before_correct, before_total), 6),
        "final_after_privilege_accuracy": round(_safe_rate(after_correct, after_total), 6),
        "conflict_rate": round(_safe_rate(conflict_total, after_total), 6),
        "conflict_toward_correctness_rate": round(_safe_rate(toward_total, conflict_total), 6),
        "conflict_away_from_correctness_rate": round(_safe_rate(away_total, conflict_total), 6),
        "counts": {
            "preliminary_total": prelim_total,
            "final_before_total": before_total,
            "final_after_total": after_total,
            "conflict_total": conflict_total,
            "conflict_toward_total": toward_total,
            "conflict_away_total": away_total,
        },
    }


def _save_eajp_attribution_metrics(
    output_dir: str,
    tag: str,
    metrics: dict[str, Any],
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{tag}_eajp_attribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(output_dir, f"{tag}_eajp_attribution.csv")
    fieldnames = [
        "preliminary_accuracy",
        "final_before_privilege_accuracy",
        "final_after_privilege_accuracy",
        "conflict_rate",
        "conflict_toward_correctness_rate",
        "conflict_away_from_correctness_rate",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: metrics.get(k, 0.0) for k in fieldnames})
    return json_path, csv_path


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
        round0 = out.debate_history.rounds[0] if out.debate_history.rounds else []
        round_last = out.debate_history.rounds[-1] if out.debate_history.rounds else []
        cached_agent_claims = {
            resp.agent_id: _claim_from_text(resp.text)
            for resp in round0
        }
        cached_round_arguments = []
        for rnd in out.debate_history.rounds:
            cached_round_arguments.append({resp.agent_id: resp.text for resp in rnd})
        cached_final_arguments = {
            resp.agent_id: resp.text
            for resp in round_last
        }
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
                "debate_cache": {
                    "rounds_executed": int((out.metadata or {}).get("rounds_executed", 0)),
                    "agent_claims": cached_agent_claims,
                    "round_arguments": cached_round_arguments,
                    "final_arguments": cached_final_arguments,
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


def _run_madam_judge_only(
    instances: list[QueryInstance],
    canonical_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    configured_rounds: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    """Run only the aggregator judge using debate round_arguments cached from a
    prior MADAM run (e.g. the first judge suite).

    Avoids re-executing the full MadamRAG debate pipeline for 2nd+ judge
    suites whose agent LLM calls would otherwise be SQLite cache hits with no
    API cost but significant Python/threading overhead.  The debate_cache
    block is copied from canonical_trials_by_id into each output trial so
    that downstream EAJP reuse (--reuse-madam-debate-for-eajp) continues to
    work correctly.
    """
    judge = Aggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=configured_rounds,
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = canonical_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and configured_rounds < len(round_arguments)

        if reused:
            selected_arguments = round_arguments[configured_rounds]
            rounds_executed = configured_rounds
            agent_claims = {
                aid: _claim_from_text(text)
                for aid, text in (round_arguments[0] if round_arguments else {}).items()
            }
            judge_resp = judge.aggregate(
                inst.question,
                selected_arguments,
                rounds_executed,
                documents=inst.documents,
                agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            round_arguments = [
                {resp.agent_id: resp.text for resp in rnd}
                for rnd in out.debate_history.rounds
            ]
            rounds_executed = int((out.metadata or {}).get("rounds_executed", configured_rounds))
            agent_claims = {
                aid: _claim_from_text(text)
                for aid, text in (round_arguments[0] if round_arguments else {}).items()
            }
            selected_arguments = round_arguments[configured_rounds] if round_arguments and configured_rounds < len(round_arguments) else {}

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system="madam_rag",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
                "debate_cache": {
                    "rounds_executed": rounds_executed,
                    "agent_claims": agent_claims,
                    "round_arguments": round_arguments,
                    "final_arguments": selected_arguments,
                },
                "reused_debate_cache": reused,
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


def _run_eajp_reuse_madam_cache(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    num_rounds: int | None,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
    compact_doc_chars: int,
    eajp_short_circuit: bool,
    privilege_scorecard: bool = True,
    system_name: str = "eajp",
) -> list[TrialResult]:
    aggregator = EvidenceAnchoredAggregator(
        judge_client or agent_client,
        privilege_scorecard=privilege_scorecard,
        short_circuit_on_confident_scorecard=eajp_short_circuit,
        compact_doc_chars=compact_doc_chars,
    )
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=EvidenceAnchoredAggregator(
            judge_client or agent_client,
            privilege_scorecard=privilege_scorecard,
            short_circuit_on_confident_scorecard=eajp_short_circuit,
            compact_doc_chars=compact_doc_chars,
        ),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        cached = _extract_cached_debate(base_trial) if base_trial is not None else None
        reused = cached is not None
        if reused:
            final_arguments, agent_claims, rounds_executed = cached
            judge_resp = aggregator.aggregate(
                inst.question,
                final_arguments,
                rounds_executed,
                documents=inst.documents,
                agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
            judge_metadata: dict[str, Any] = dict(getattr(judge_resp, "metadata", {}) or {})
        else:
            # Fallback keeps execution robust for older partial files that do not
            # contain debate cache metadata yet.
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            judge_metadata = dict((out.metadata or {}).get("judge_metadata", {}) or {})
            rounds_executed = int((out.metadata or {}).get("rounds_executed", 0))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=system_name,
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": system_name,
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
                "judge_metadata": judge_metadata,
            },
        )

    return _evaluate_system(
        system_name=system_name,
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_madam_from_cache_for_round(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    target_round: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    judge = Aggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=target_round,
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and target_round < len(round_arguments)
        if reused:
            selected_arguments = round_arguments[target_round]
            rounds_executed = target_round
            judge_resp = judge.aggregate(
                inst.question,
                selected_arguments,
                rounds_executed,
                documents=inst.documents,
                agent_claims={aid: _claim_from_text(text) for aid, text in round_arguments[0].items()} if round_arguments else {},
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", target_round))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=f"madam_rag_r{target_round}",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "ablation_target_round": target_round,
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name=f"madam_rag_r{target_round}",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_eajp_from_cache_for_round(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    target_round: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
    compact_doc_chars: int,
    eajp_short_circuit: bool,
    privilege_scorecard: bool = True,
    system_name_prefix: str = "eajp",
) -> list[TrialResult]:
    aggregator = EvidenceAnchoredAggregator(
        judge_client or agent_client,
        privilege_scorecard=privilege_scorecard,
        short_circuit_on_confident_scorecard=eajp_short_circuit,
        compact_doc_chars=compact_doc_chars,
    )
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=target_round,
        judge=EvidenceAnchoredAggregator(
            judge_client or agent_client,
            privilege_scorecard=privilege_scorecard,
            short_circuit_on_confident_scorecard=eajp_short_circuit,
            compact_doc_chars=compact_doc_chars,
        ),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and target_round < len(round_arguments)
        if reused:
            selected_arguments = round_arguments[target_round]
            rounds_executed = target_round
            agent_claims = {
                aid: _claim_from_text(text)
                for aid, text in (round_arguments[0] if round_arguments else {}).items()
            }
            judge_resp = aggregator.aggregate(
                inst.question,
                selected_arguments,
                rounds_executed,
                documents=inst.documents,
                agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
            judge_metadata: dict[str, Any] = dict(getattr(judge_resp, "metadata", {}) or {})
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            judge_metadata = dict((out.metadata or {}).get("judge_metadata", {}) or {})
            rounds_executed = int((out.metadata or {}).get("rounds_executed", target_round))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=f"{system_name_prefix}_r{target_round}",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": system_name_prefix,
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "ablation_target_round": target_round,
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
                "judge_metadata": judge_metadata,
            },
        )

    return _evaluate_system(
        system_name=f"{system_name_prefix}_r{target_round}",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_eajp_scorecard_only_from_cache_for_round(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    target_round: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
    compact_doc_chars: int,
) -> list[TrialResult]:
    aggregator = EvidenceAnchoredAggregator(
        judge_client or agent_client,
        short_circuit_on_confident_scorecard=False,
        compact_doc_chars=compact_doc_chars,
        force_scorecard_only=True,
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and target_round < len(round_arguments)
        selected_arguments = round_arguments[target_round] if reused else {}
        rounds_executed = target_round if reused else 0
        agent_claims = {
            aid: _claim_from_text(text)
            for aid, text in (round_arguments[0] if round_arguments else {}).items()
        }
        judge_resp = aggregator.aggregate(
            inst.question,
            selected_arguments,
            rounds_executed,
            documents=inst.documents,
            agent_claims=agent_claims,
        )
        final_answer = judge_resp.text
        model = judge_resp.model
        usage = judge_resp.usage or {}
        judge_metadata: dict[str, Any] = dict(getattr(judge_resp, "metadata", {}) or {})
        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=f"eajp_scorecard_only_r{target_round}",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "eajp_scorecard_only",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "ablation_target_round": target_round,
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
                "judge_metadata": judge_metadata,
            },
        )

    return _evaluate_system(
        system_name=f"eajp_scorecard_only_r{target_round}",
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
    privilege_scorecard: bool = True,
    system_name: str = "eajp",
) -> list[TrialResult]:
    pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=EvidenceAnchoredAggregator(
            judge_client or agent_client,
            privilege_scorecard=privilege_scorecard,
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
            system=system_name,
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "judge_mode": system_name,
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
        system_name=system_name,
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_eajp_scorecard_only(
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
        judge=EvidenceAnchoredAggregator(
            judge_client or agent_client,
            short_circuit_on_confident_scorecard=False,
            compact_doc_chars=compact_doc_chars,
            force_scorecard_only=True,
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
            system="eajp_scorecard_only",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "judge_mode": "eajp_scorecard_only",
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
        system_name="eajp_scorecard_only",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_credibility_hint_reuse_madam_cache(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    num_rounds: int | None,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    aggregator = CredibilityHintAggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=CredibilityHintAggregator(judge_client or agent_client),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        cached = _extract_cached_debate(base_trial) if base_trial is not None else None
        reused = cached is not None
        if reused:
            final_arguments, agent_claims, rounds_executed = cached
            judge_resp = aggregator.aggregate(
                inst.question,
                final_arguments,
                rounds_executed,
                documents=inst.documents,
                agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", 0))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system="credibility_hint",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "credibility_hint",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name="credibility_hint",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_credibility_hint(
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
        judge=CredibilityHintAggregator(judge_client or agent_client),
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
            system="credibility_hint",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "judge_mode": "credibility_hint",
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
        system_name="credibility_hint",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_credibility_hint_from_cache_for_round(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    target_round: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    aggregator = CredibilityHintAggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=target_round,
        judge=CredibilityHintAggregator(judge_client or agent_client),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and target_round < len(round_arguments)
        if reused:
            selected_arguments = round_arguments[target_round]
            rounds_executed = target_round
            agent_claims = {
                aid: _claim_from_text(text)
                for aid, text in (round_arguments[0] if round_arguments else {}).items()
            }
            judge_resp = aggregator.aggregate(
                inst.question,
                selected_arguments,
                rounds_executed,
                documents=inst.documents,
                agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", target_round))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=f"credibility_hint_r{target_round}",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "credibility_hint",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "ablation_target_round": target_round,
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name=f"credibility_hint_r{target_round}",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_ra_rag_reuse_madam_cache(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    num_rounds: int | None,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    aggregator = ReliabilityAwareAggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=ReliabilityAwareAggregator(judge_client or agent_client),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        cached = _extract_cached_debate(base_trial) if base_trial is not None else None
        reused = cached is not None
        if reused:
            final_arguments, agent_claims, rounds_executed = cached
            judge_resp = aggregator.aggregate(
                inst.question, final_arguments, rounds_executed,
                documents=inst.documents, agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", 0))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system="ra_rag",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "ra_rag",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name="ra_rag",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_ra_rag(
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
        judge=ReliabilityAwareAggregator(judge_client or agent_client),
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
            system="ra_rag",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "judge_mode": "ra_rag",
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
        system_name="ra_rag",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_ra_rag_from_cache_for_round(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    target_round: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    aggregator = ReliabilityAwareAggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=target_round,
        judge=ReliabilityAwareAggregator(judge_client or agent_client),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and target_round < len(round_arguments)
        if reused:
            selected_arguments = round_arguments[target_round]
            rounds_executed = target_round
            agent_claims = {
                aid: _claim_from_text(text)
                for aid, text in (round_arguments[0] if round_arguments else {}).items()
            }
            judge_resp = aggregator.aggregate(
                inst.question, selected_arguments, rounds_executed,
                documents=inst.documents, agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", target_round))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=f"ra_rag_r{target_round}",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "ra_rag",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "ablation_target_round": target_round,
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name=f"ra_rag_r{target_round}",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_cot_judge_reuse_madam_cache(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    num_rounds: int | None,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    aggregator = ChainOfThoughtAggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=ChainOfThoughtAggregator(judge_client or agent_client),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        cached = _extract_cached_debate(base_trial) if base_trial is not None else None
        reused = cached is not None
        if reused:
            final_arguments, agent_claims, rounds_executed = cached
            judge_resp = aggregator.aggregate(
                inst.question, final_arguments, rounds_executed,
                documents=inst.documents, agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", 0))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system="cot_judge",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "cot_judge",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name="cot_judge",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_cot_judge(
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
        judge=ChainOfThoughtAggregator(judge_client or agent_client),
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
            system="cot_judge",
            model=out.model,
            classification=classification,
            final_answer=out.final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": pipeline.num_rounds,
                "num_agents": len(inst.documents),
                "judge_mode": "cot_judge",
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
        system_name="cot_judge",
        instances=instances,
        run_one=_run_one,
        output_dir=output_dir,
        partial_tag=partial_tag,
        max_workers=max_workers,
        resume=resume,
        benchmark_name=benchmark_name,
        model_name=(judge_client.model if judge_client is not None else agent_client.model),
    )


def _run_cot_judge_from_cache_for_round(
    instances: list[QueryInstance],
    madam_trials_by_id: dict[str, TrialResult],
    agent_client: LLMClient,
    judge_client: LLMClient | None,
    target_round: int,
    benchmark_name: str,
    output_dir: str,
    partial_tag: str,
    max_workers: int,
    resume: bool,
) -> list[TrialResult]:
    aggregator = ChainOfThoughtAggregator(judge_client or agent_client)
    fallback_pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=target_round,
        judge=ChainOfThoughtAggregator(judge_client or agent_client),
    )

    def _run_one(inst: QueryInstance) -> TrialResult:
        base_trial = madam_trials_by_id.get(inst.id)
        round_arguments = _extract_cached_round_arguments(base_trial) if base_trial is not None else []
        reused = bool(round_arguments) and target_round < len(round_arguments)
        if reused:
            selected_arguments = round_arguments[target_round]
            rounds_executed = target_round
            agent_claims = {
                aid: _claim_from_text(text)
                for aid, text in (round_arguments[0] if round_arguments else {}).items()
            }
            judge_resp = aggregator.aggregate(
                inst.question, selected_arguments, rounds_executed,
                documents=inst.documents, agent_claims=agent_claims,
            )
            final_answer = judge_resp.text
            model = judge_resp.model
            usage = judge_resp.usage or {}
        else:
            out = fallback_pipeline.answer(inst)
            final_answer = out.final_answer
            model = out.model
            usage = out.usage or {}
            rounds_executed = int((out.metadata or {}).get("rounds_executed", target_round))

        classification = classify_answer(final_answer, inst)
        return TrialResult(
            query_id=inst.id,
            system=f"cot_judge_r{target_round}",
            model=model,
            classification=classification,
            final_answer=final_answer,
            correct_answer=inst.correct_answer,
            metadata={
                "num_debate_rounds": rounds_executed,
                "num_agents": len(inst.documents),
                "judge_mode": "cot_judge",
                "benchmark": benchmark_name,
                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                "ablation_target_round": target_round,
                "reused_madam_debate": reused,
                "telemetry": {
                    "usage": usage,
                    "provider": (judge_client.provider if judge_client is not None else agent_client.provider),
                    "model": model,
                },
            },
        )

    return _evaluate_system(
        system_name=f"cot_judge_r{target_round}",
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
        if "eajp_scorecard_only" in systems:
            logger.info(
                "Model plan: eajp_scorecard_only [%s] -> agents=(%s, %s), evidence_judge=(%s, %s)",
                suite_name,
                eajp_agent_client.provider,
                eajp_agent_client.model,
                judge_client.provider,
                judge_client.model,
            )
        if "eajp_no_privilege" in systems:
            logger.info(
                "Model plan: eajp_no_privilege [%s] -> agents=(%s, %s), evidence_judge=(%s, %s) [no override]",
                suite_name,
                eajp_agent_client.provider,
                eajp_agent_client.model,
                judge_client.provider,
                judge_client.model,
            )
        if "credibility_hint" in systems:
            logger.info(
                "Model plan: credibility_hint [%s] -> agents=(%s, %s), judge=(%s, %s)",
                suite_name,
                madam_agent_client.provider,
                madam_agent_client.model,
                judge_client.provider,
                judge_client.model,
            )
        if "ra_rag" in systems:
            logger.info(
                "Model plan: ra_rag [%s] -> agents=(%s, %s), judge=(%s, %s) [reliability-aware]",
                suite_name,
                madam_agent_client.provider,
                madam_agent_client.model,
                judge_client.provider,
                judge_client.model,
            )
        if "cot_judge" in systems:
            logger.info(
                "Model plan: cot_judge [%s] -> agents=(%s, %s), judge=(%s, %s) [chain-of-thought]",
                suite_name,
                madam_agent_client.provider,
                madam_agent_client.model,
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
        choices=["custom", "ramdocs", "confact", "tcb"],
        default="custom",
        help="Benchmark family used to load and filter the dataset.",
    )
    p.add_argument(
        "--systems", nargs="+",
        choices=[
            "standard_rag", "credibility_vote",
            "madam_rag", "cot_judge", "ra_rag",
            "credibility_hint", "eajp_scorecard_only",
            "eajp_no_privilege", "eajp",
        ],
        default=[
            "standard_rag", "credibility_vote",
            "madam_rag", "cot_judge", "ra_rag",
            "credibility_hint", "eajp_scorecard_only",
            "eajp_no_privilege", "eajp",
        ],
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
        "--madam-compact-doc-chars", type=int, default=10000,
        help="Character budget for per-agent document excerpt in debate prompts.",
    )
    p.add_argument(
        "--madam-argument-summary-chars", type=int, default=1000,
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
        "--reuse-madam-debate-for-eajp", action=argparse.BooleanOptionalAction, default=False,
        help="Reuse cached MADAM debate outputs for EAJP judging to reduce duplicate agent calls.",
    )
    p.add_argument(
        "--ablate-rounds-from-cache", nargs="*", type=int, default=None,
        help="Extra ablation rounds to evaluate from MADAM per-round cache (e.g., 0 1 2 3).",
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

    ablation_rounds: list[int] = []
    if args.ablate_rounds_from_cache:
        ablation_rounds = sorted({int(r) for r in args.ablate_rounds_from_cache if int(r) >= 0})
        configured_rounds = args.judge_rounds if args.judge_rounds is not None else config.DEBATE_ROUNDS
        if ablation_rounds and configured_rounds < max(ablation_rounds):
            raise ValueError(
                "--judge-rounds must be >= max(--ablate-rounds-from-cache) so all requested rounds are cached."
            )

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

    # Credibility-weighted majority vote — deterministic, no LLM, suite-independent.
    if "credibility_vote" in args.systems:
        logger.info("=== Running credibility_vote (suite-independent, deterministic) ===")
        cv_trials = _run_credibility_vote(
            instances,
            benchmark_name=args.benchmark,
            output_dir=output_dir,
        )
        cv_metrics = compute_metrics(cv_trials, "credibility_vote", "deterministic")
        print_metrics(cv_metrics)
        cv_path = save_results(cv_metrics, output_dir, tag="credibility_vote")
        logger.info("Saved credibility_vote results to %s", cv_path)

    instances_by_id = {inst.id: inst for inst in instances}

    # Run requested systems for each judge suite.
    # Populated after the first suite runs MADAM; subsequent suites reuse the
    # cached debate round_arguments to skip re-executing the full debate pipeline.
    canonical_madam_debate_trials_by_id: dict[str, TrialResult] = {}
    cross_suite_ablation_rows: list[dict[str, Any]] = []
    for suite_name, judge_client in suite_configs:
        suite_output_dir = os.path.join(output_dir, suite_name)
        os.makedirs(suite_output_dir, exist_ok=True)
        logger.info("=== Judge suite: %s (%s) ===", suite_name, judge_client)
        ablation_summary_rows: list[dict[str, Any]] = []

        suite_systems = [s for s in args.systems if s not in {"standard_rag", "credibility_vote"}]
        if args.reuse_madam_debate_for_eajp and "madam_rag" in suite_systems:
            delayed = [
                s for s in (
                    "cot_judge", "ra_rag",
                    "credibility_hint", "eajp_scorecard_only",
                    "eajp_no_privilege", "eajp",
                )
                if s in suite_systems
            ]
            if delayed:
                suite_systems = [s for s in suite_systems if s not in delayed] + delayed

        madam_trials_by_id: dict[str, TrialResult] = {}

        for system_name in suite_systems:
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
                if canonical_madam_debate_trials_by_id:
                    # 2nd+ suites: skip debate re-execution; run only the new judge.
                    # Agent LLM calls would all be SQLite cache hits anyway, but
                    # re-running MadamRAG's debate machinery still incurs significant
                    # Python/threading overhead per instance (~31 min on a full dataset).
                    _configured_rounds = (
                        args.judge_rounds if args.judge_rounds is not None else config.DEBATE_ROUNDS
                    )
                    logger.info(
                        "Reusing canonical MADAM debate cache for suite %s "
                        "(judge-only re-run, rounds=%d)",
                        suite_name,
                        _configured_rounds,
                    )
                    trials = _run_madam_judge_only(
                        run_instances,
                        canonical_trials_by_id=canonical_madam_debate_trials_by_id,
                        agent_client=madam_agent_client,
                        judge_client=judge_client,
                        configured_rounds=_configured_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_madam_rag",
                        max_workers=max_workers,
                        resume=args.resume,
                    )
                else:
                    # First suite: run the full debate pipeline.
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
                    # Populate canonical debate cache for subsequent suites.
                    canonical_madam_debate_trials_by_id = {t.query_id: t for t in trials}
                if fallback_trials:
                    trials = _ordered_trials(instances, {t.query_id: t for t in (trials + fallback_trials)})
                model_name = judge_client.model
                madam_trials_by_id = {t.query_id: t for t in trials}
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
                if args.reuse_madam_debate_for_eajp and madam_trials_by_id:
                    logger.info("EAJP reusing MADAM debate cache for %d instances", len(madam_trials_by_id))
                    trials = _run_eajp_reuse_madam_cache(
                        run_instances,
                        madam_trials_by_id=madam_trials_by_id,
                        agent_client=eajp_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_eajp",
                        max_workers=max_workers,
                        resume=args.resume,
                        compact_doc_chars=args.madam_compact_doc_chars,
                        eajp_short_circuit=args.eajp_short_circuit,
                    )
                else:
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
            elif system_name == "eajp_scorecard_only":
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
                                    system="eajp_scorecard_only",
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
                                    system="eajp_scorecard_only",
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
                if args.reuse_madam_debate_for_eajp and madam_trials_by_id:
                    logger.info("EAJP scorecard-only reusing MADAM debate cache for %d instances", len(madam_trials_by_id))
                    target_round = max(0, args.judge_rounds if args.judge_rounds is not None else config.DEBATE_ROUNDS)
                    trials = _run_eajp_scorecard_only_from_cache_for_round(
                        run_instances,
                        madam_trials_by_id=madam_trials_by_id,
                        agent_client=eajp_agent_client,
                        judge_client=judge_client,
                        target_round=target_round,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_eajp_scorecard_only",
                        max_workers=max_workers,
                        resume=args.resume,
                        compact_doc_chars=args.madam_compact_doc_chars,
                    )
                else:
                    trials = _run_eajp_scorecard_only(
                        run_instances,
                        eajp_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_eajp_scorecard_only",
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
            elif system_name == "eajp_no_privilege":
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
                                    system="eajp_no_privilege",
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
                                    system="eajp_no_privilege",
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
                if args.reuse_madam_debate_for_eajp and madam_trials_by_id:
                    logger.info("eajp_no_privilege reusing MADAM debate cache for %d instances", len(madam_trials_by_id))
                    trials = _run_eajp_reuse_madam_cache(
                        run_instances,
                        madam_trials_by_id=madam_trials_by_id,
                        agent_client=eajp_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_eajp_no_privilege",
                        max_workers=max_workers,
                        resume=args.resume,
                        compact_doc_chars=args.madam_compact_doc_chars,
                        eajp_short_circuit=args.eajp_short_circuit,
                        privilege_scorecard=False,
                        system_name="eajp_no_privilege",
                    )
                else:
                    trials = _run_eajp(
                        run_instances,
                        eajp_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_eajp_no_privilege",
                        max_workers=max_workers,
                        resume=args.resume,
                        max_parallel_agents=args.madam_max_parallel_agents,
                        compact_doc_chars=args.madam_compact_doc_chars,
                        argument_summary_chars=args.madam_argument_summary_chars,
                        early_stop_patience=args.madam_early_stop_patience,
                        include_full_document_in_debate=args.madam_include_full_document_in_debate,
                        eajp_short_circuit=args.eajp_short_circuit,
                        privilege_scorecard=False,
                        system_name="eajp_no_privilege",
                    )
                if fallback_trials:
                    trials = _ordered_trials(instances, {t.query_id: t for t in (trials + fallback_trials)})
                model_name = judge_client.model
            elif system_name == "credibility_hint":
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
                                    system="credibility_hint",
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
                                    system="credibility_hint",
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
                if args.reuse_madam_debate_for_eajp and madam_trials_by_id:
                    logger.info("credibility_hint reusing MADAM debate cache for %d instances", len(madam_trials_by_id))
                    trials = _run_credibility_hint_reuse_madam_cache(
                        run_instances,
                        madam_trials_by_id=madam_trials_by_id,
                        agent_client=madam_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_credibility_hint",
                        max_workers=max_workers,
                        resume=args.resume,
                    )
                else:
                    trials = _run_credibility_hint(
                        run_instances,
                        madam_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_credibility_hint",
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
            elif system_name == "cot_judge":
                if args.reuse_madam_debate_for_eajp and madam_trials_by_id:
                    logger.info("cot_judge reusing MADAM debate cache for %d instances", len(madam_trials_by_id))
                    trials = _run_cot_judge_reuse_madam_cache(
                        instances,
                        madam_trials_by_id=madam_trials_by_id,
                        agent_client=madam_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_cot_judge",
                        max_workers=max_workers,
                        resume=args.resume,
                    )
                else:
                    trials = _run_cot_judge(
                        instances,
                        madam_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_cot_judge",
                        max_workers=max_workers,
                        resume=args.resume,
                        max_parallel_agents=args.madam_max_parallel_agents,
                        compact_doc_chars=args.madam_compact_doc_chars,
                        argument_summary_chars=args.madam_argument_summary_chars,
                        early_stop_patience=args.madam_early_stop_patience,
                        include_full_document_in_debate=args.madam_include_full_document_in_debate,
                    )
                model_name = judge_client.model
            elif system_name == "ra_rag":
                if args.reuse_madam_debate_for_eajp and madam_trials_by_id:
                    logger.info("ra_rag reusing MADAM debate cache for %d instances", len(madam_trials_by_id))
                    trials = _run_ra_rag_reuse_madam_cache(
                        instances,
                        madam_trials_by_id=madam_trials_by_id,
                        agent_client=madam_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_ra_rag",
                        max_workers=max_workers,
                        resume=args.resume,
                    )
                else:
                    trials = _run_ra_rag(
                        instances,
                        madam_agent_client,
                        judge_client=judge_client,
                        num_rounds=args.judge_rounds,
                        benchmark_name=args.benchmark,
                        output_dir=suite_output_dir,
                        partial_tag=f"{suite_name}_ra_rag",
                        max_workers=max_workers,
                        resume=args.resume,
                        max_parallel_agents=args.madam_max_parallel_agents,
                        compact_doc_chars=args.madam_compact_doc_chars,
                        argument_summary_chars=args.madam_argument_summary_chars,
                        early_stop_patience=args.madam_early_stop_patience,
                        include_full_document_in_debate=args.madam_include_full_document_in_debate,
                    )
                model_name = judge_client.model
            else:
                raise ValueError(f"Unknown system: {system_name}")

            metrics = compute_metrics(trials, system_name, model_name)
            print_metrics(metrics)
            path = save_results(metrics, suite_output_dir, tag=system_name)
            logger.info("Saved %s results to %s", system_name, path)
            if system_name in {"eajp", "eajp_scorecard_only", "eajp_no_privilege"}:
                attr_metrics = _compute_eajp_attribution_metrics(trials, instances_by_id)
                attr_json, attr_csv = _save_eajp_attribution_metrics(suite_output_dir, system_name, attr_metrics)
                logger.info("Saved %s attribution metrics to %s and %s", system_name, attr_json, attr_csv)

        if ablation_rounds and madam_trials_by_id:
            logger.info("=== Running cache-driven round ablations: %s ===", ablation_rounds)
            for target_round in ablation_rounds:
                # madam_rag, eajp, and eajp_scorecard_only at the same round read
                # from the same (read-only) madam_trials_by_id and write to separate
                # partial files — submit all three concurrently and wait together.
                _abl_futs: dict[str, Any] = {}
                with ThreadPoolExecutor(max_workers=8) as _abl_pool:
                    if "madam_rag" in args.systems:
                        _abl_futs["madam_rag"] = _abl_pool.submit(
                            _run_madam_from_cache_for_round,
                            instances, madam_trials_by_id, madam_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_madam_rag_r{target_round}",
                            max_workers, args.resume,
                        )
                    if "eajp" in args.systems:
                        _abl_futs["eajp"] = _abl_pool.submit(
                            _run_eajp_from_cache_for_round,
                            instances, madam_trials_by_id, eajp_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_eajp_r{target_round}",
                            max_workers, args.resume,
                            args.madam_compact_doc_chars, args.eajp_short_circuit,
                        )
                    if "eajp_scorecard_only" in args.systems:
                        _abl_futs["eajp_scorecard_only"] = _abl_pool.submit(
                            _run_eajp_scorecard_only_from_cache_for_round,
                            instances, madam_trials_by_id, eajp_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_eajp_scorecard_only_r{target_round}",
                            max_workers, args.resume,
                            args.madam_compact_doc_chars,
                        )
                    if "eajp_no_privilege" in args.systems:
                        _abl_futs["eajp_no_privilege"] = _abl_pool.submit(
                            _run_eajp_from_cache_for_round,
                            instances, madam_trials_by_id, eajp_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_eajp_no_privilege_r{target_round}",
                            max_workers, args.resume,
                            args.madam_compact_doc_chars, args.eajp_short_circuit,
                            False,              # privilege_scorecard=False
                            "eajp_no_privilege",  # system_name_prefix
                        )
                    if "credibility_hint" in args.systems:
                        _abl_futs["credibility_hint"] = _abl_pool.submit(
                            _run_credibility_hint_from_cache_for_round,
                            instances, madam_trials_by_id, madam_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_credibility_hint_r{target_round}",
                            max_workers, args.resume,
                        )
                    if "cot_judge" in args.systems:
                        _abl_futs["cot_judge"] = _abl_pool.submit(
                            _run_cot_judge_from_cache_for_round,
                            instances, madam_trials_by_id, madam_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_cot_judge_r{target_round}",
                            max_workers, args.resume,
                        )
                    if "ra_rag" in args.systems:
                        _abl_futs["ra_rag"] = _abl_pool.submit(
                            _run_ra_rag_from_cache_for_round,
                            instances, madam_trials_by_id, madam_agent_client,
                            judge_client, target_round, args.benchmark,
                            suite_output_dir,
                            f"{suite_name}_ra_rag_r{target_round}",
                            max_workers, args.resume,
                        )
                # All tasks complete; process results sequentially.
                if "madam_rag" in _abl_futs:
                    madam_ablate_trials = _abl_futs["madam_rag"].result()
                    madam_tag = f"madam_rag_r{target_round}"
                    madam_metrics = compute_metrics(madam_ablate_trials, madam_tag, judge_client.model)
                    print_metrics(madam_metrics)
                    madam_path = save_results(madam_metrics, suite_output_dir, tag=madam_tag)
                    logger.info("Saved %s results to %s", madam_tag, madam_path)
                    ablation_summary_rows.append(
                        {
                            "system": "madam_rag",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": madam_metrics.total,
                            "tone_controlled_total": madam_metrics.tone_controlled_total,
                            "correct": madam_metrics.correct,
                            "incorrect": madam_metrics.incorrect,
                            "unclear": madam_metrics.unclear,
                            "tone_bias_rate": round(madam_metrics.tone_bias_rate, 6),
                            "accuracy": round(madam_metrics.accuracy, 6),
                        }
                    )

                if "eajp" in _abl_futs:
                    eajp_ablate_trials = _abl_futs["eajp"].result()
                    eajp_tag = f"eajp_r{target_round}"
                    eajp_metrics = compute_metrics(eajp_ablate_trials, eajp_tag, judge_client.model)
                    print_metrics(eajp_metrics)
                    eajp_path = save_results(eajp_metrics, suite_output_dir, tag=eajp_tag)
                    logger.info("Saved %s results to %s", eajp_tag, eajp_path)
                    eajp_attr = _compute_eajp_attribution_metrics(eajp_ablate_trials, instances_by_id)
                    eajp_attr_json, eajp_attr_csv = _save_eajp_attribution_metrics(suite_output_dir, eajp_tag, eajp_attr)
                    logger.info("Saved %s attribution metrics to %s and %s", eajp_tag, eajp_attr_json, eajp_attr_csv)
                    ablation_summary_rows.append(
                        {
                            "system": "eajp",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": eajp_metrics.total,
                            "tone_controlled_total": eajp_metrics.tone_controlled_total,
                            "correct": eajp_metrics.correct,
                            "incorrect": eajp_metrics.incorrect,
                            "unclear": eajp_metrics.unclear,
                            "tone_bias_rate": round(eajp_metrics.tone_bias_rate, 6),
                            "accuracy": round(eajp_metrics.accuracy, 6),
                            "preliminary_accuracy": eajp_attr.get("preliminary_accuracy", 0.0),
                            "final_before_privilege_accuracy": eajp_attr.get("final_before_privilege_accuracy", 0.0),
                            "final_after_privilege_accuracy": eajp_attr.get("final_after_privilege_accuracy", 0.0),
                            "conflict_rate": eajp_attr.get("conflict_rate", 0.0),
                            "conflict_toward_correctness_rate": eajp_attr.get("conflict_toward_correctness_rate", 0.0),
                            "conflict_away_from_correctness_rate": eajp_attr.get("conflict_away_from_correctness_rate", 0.0),
                        }
                    )

                if "eajp_scorecard_only" in _abl_futs:
                    eajp_sc_trials = _abl_futs["eajp_scorecard_only"].result()
                    eajp_sc_tag = f"eajp_scorecard_only_r{target_round}"
                    eajp_sc_metrics = compute_metrics(eajp_sc_trials, eajp_sc_tag, judge_client.model)
                    print_metrics(eajp_sc_metrics)
                    eajp_sc_path = save_results(eajp_sc_metrics, suite_output_dir, tag=eajp_sc_tag)
                    logger.info("Saved %s results to %s", eajp_sc_tag, eajp_sc_path)
                    eajp_sc_attr = _compute_eajp_attribution_metrics(eajp_sc_trials, instances_by_id)
                    eajp_sc_attr_json, eajp_sc_attr_csv = _save_eajp_attribution_metrics(suite_output_dir, eajp_sc_tag, eajp_sc_attr)
                    logger.info("Saved %s attribution metrics to %s and %s", eajp_sc_tag, eajp_sc_attr_json, eajp_sc_attr_csv)
                    ablation_summary_rows.append(
                        {
                            "system": "eajp_scorecard_only",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": eajp_sc_metrics.total,
                            "tone_controlled_total": eajp_sc_metrics.tone_controlled_total,
                            "correct": eajp_sc_metrics.correct,
                            "incorrect": eajp_sc_metrics.incorrect,
                            "unclear": eajp_sc_metrics.unclear,
                            "tone_bias_rate": round(eajp_sc_metrics.tone_bias_rate, 6),
                            "accuracy": round(eajp_sc_metrics.accuracy, 6),
                            "preliminary_accuracy": eajp_sc_attr.get("preliminary_accuracy", 0.0),
                            "final_before_privilege_accuracy": eajp_sc_attr.get("final_before_privilege_accuracy", 0.0),
                            "final_after_privilege_accuracy": eajp_sc_attr.get("final_after_privilege_accuracy", 0.0),
                            "conflict_rate": eajp_sc_attr.get("conflict_rate", 0.0),
                            "conflict_toward_correctness_rate": eajp_sc_attr.get("conflict_toward_correctness_rate", 0.0),
                            "conflict_away_from_correctness_rate": eajp_sc_attr.get("conflict_away_from_correctness_rate", 0.0),
                        }
                    )

                if "eajp_no_privilege" in _abl_futs:
                    eajp_np_trials = _abl_futs["eajp_no_privilege"].result()
                    eajp_np_tag = f"eajp_no_privilege_r{target_round}"
                    eajp_np_metrics = compute_metrics(eajp_np_trials, eajp_np_tag, judge_client.model)
                    print_metrics(eajp_np_metrics)
                    eajp_np_path = save_results(eajp_np_metrics, suite_output_dir, tag=eajp_np_tag)
                    logger.info("Saved %s results to %s", eajp_np_tag, eajp_np_path)
                    eajp_np_attr = _compute_eajp_attribution_metrics(eajp_np_trials, instances_by_id)
                    eajp_np_attr_json, eajp_np_attr_csv = _save_eajp_attribution_metrics(suite_output_dir, eajp_np_tag, eajp_np_attr)
                    logger.info("Saved %s attribution metrics to %s and %s", eajp_np_tag, eajp_np_attr_json, eajp_np_attr_csv)
                    ablation_summary_rows.append(
                        {
                            "system": "eajp_no_privilege",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": eajp_np_metrics.total,
                            "tone_controlled_total": eajp_np_metrics.tone_controlled_total,
                            "correct": eajp_np_metrics.correct,
                            "incorrect": eajp_np_metrics.incorrect,
                            "unclear": eajp_np_metrics.unclear,
                            "tone_bias_rate": round(eajp_np_metrics.tone_bias_rate, 6),
                            "accuracy": round(eajp_np_metrics.accuracy, 6),
                            "preliminary_accuracy": eajp_np_attr.get("preliminary_accuracy", 0.0),
                            "final_before_privilege_accuracy": eajp_np_attr.get("final_before_privilege_accuracy", 0.0),
                            "final_after_privilege_accuracy": eajp_np_attr.get("final_after_privilege_accuracy", 0.0),
                            "conflict_rate": eajp_np_attr.get("conflict_rate", 0.0),
                            "conflict_toward_correctness_rate": eajp_np_attr.get("conflict_toward_correctness_rate", 0.0),
                            "conflict_away_from_correctness_rate": eajp_np_attr.get("conflict_away_from_correctness_rate", 0.0),
                        }
                    )

                if "credibility_hint" in _abl_futs:
                    ch_trials = _abl_futs["credibility_hint"].result()
                    ch_tag = f"credibility_hint_r{target_round}"
                    ch_metrics = compute_metrics(ch_trials, ch_tag, judge_client.model)
                    print_metrics(ch_metrics)
                    ch_path = save_results(ch_metrics, suite_output_dir, tag=ch_tag)
                    logger.info("Saved %s results to %s", ch_tag, ch_path)
                    ablation_summary_rows.append(
                        {
                            "system": "credibility_hint",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": ch_metrics.total,
                            "tone_controlled_total": ch_metrics.tone_controlled_total,
                            "correct": ch_metrics.correct,
                            "incorrect": ch_metrics.incorrect,
                            "unclear": ch_metrics.unclear,
                            "tone_bias_rate": round(ch_metrics.tone_bias_rate, 6),
                            "accuracy": round(ch_metrics.accuracy, 6),
                        }
                    )

                if "cot_judge" in _abl_futs:
                    cot_trials = _abl_futs["cot_judge"].result()
                    cot_tag = f"cot_judge_r{target_round}"
                    cot_metrics = compute_metrics(cot_trials, cot_tag, judge_client.model)
                    print_metrics(cot_metrics)
                    cot_path = save_results(cot_metrics, suite_output_dir, tag=cot_tag)
                    logger.info("Saved %s results to %s", cot_tag, cot_path)
                    ablation_summary_rows.append(
                        {
                            "system": "cot_judge",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": cot_metrics.total,
                            "tone_controlled_total": cot_metrics.tone_controlled_total,
                            "correct": cot_metrics.correct,
                            "incorrect": cot_metrics.incorrect,
                            "unclear": cot_metrics.unclear,
                            "tone_bias_rate": round(cot_metrics.tone_bias_rate, 6),
                            "accuracy": round(cot_metrics.accuracy, 6),
                        }
                    )

                if "ra_rag" in _abl_futs:
                    ra_trials = _abl_futs["ra_rag"].result()
                    ra_tag = f"ra_rag_r{target_round}"
                    ra_metrics = compute_metrics(ra_trials, ra_tag, judge_client.model)
                    print_metrics(ra_metrics)
                    ra_path = save_results(ra_metrics, suite_output_dir, tag=ra_tag)
                    logger.info("Saved %s results to %s", ra_tag, ra_path)
                    ablation_summary_rows.append(
                        {
                            "system": "ra_rag",
                            "round": target_round,
                            "model": judge_client.model,
                            "total": ra_metrics.total,
                            "tone_controlled_total": ra_metrics.tone_controlled_total,
                            "correct": ra_metrics.correct,
                            "incorrect": ra_metrics.incorrect,
                            "unclear": ra_metrics.unclear,
                            "tone_bias_rate": round(ra_metrics.tone_bias_rate, 6),
                            "accuracy": round(ra_metrics.accuracy, 6),
                        }
                    )

        if ablation_summary_rows:
            ablation_summary_rows.sort(key=lambda r: (str(r.get("system", "")), int(r.get("round", 0))))
            summary_csv_path = _write_round_ablation_summary_csv(suite_output_dir, ablation_summary_rows)
            logger.info("Saved round-ablation summary CSV to %s", summary_csv_path)
            cross_suite_ablation_rows.extend(
                [
                    {
                        "suite": suite_name,
                        "system": row.get("system", ""),
                        "round": row.get("round", 0),
                        "model": row.get("model", ""),
                        "total": row.get("total", 0),
                        "tone_controlled_total": row.get("tone_controlled_total", 0),
                        "correct": row.get("correct", 0),
                        "incorrect": row.get("incorrect", 0),
                        "unclear": row.get("unclear", 0),
                        "tone_bias_rate": row.get("tone_bias_rate", 0.0),
                        "accuracy": row.get("accuracy", 0.0),
                        "preliminary_accuracy": row.get("preliminary_accuracy", ""),
                        "final_before_privilege_accuracy": row.get("final_before_privilege_accuracy", ""),
                        "final_after_privilege_accuracy": row.get("final_after_privilege_accuracy", ""),
                        "conflict_rate": row.get("conflict_rate", ""),
                        "conflict_toward_correctness_rate": row.get("conflict_toward_correctness_rate", ""),
                        "conflict_away_from_correctness_rate": row.get("conflict_away_from_correctness_rate", ""),
                    }
                    for row in ablation_summary_rows
                ]
            )

    if cross_suite_ablation_rows:
        cross_suite_ablation_rows.sort(
            key=lambda r: (
                str(r.get("suite", "")),
                str(r.get("system", "")),
                int(r.get("round", 0)),
            )
        )
        cross_suite_csv_path = _write_cross_suite_round_ablation_summary_csv(
            output_dir,
            cross_suite_ablation_rows,
        )
        logger.info("Saved cross-suite round-ablation summary CSV to %s", cross_suite_csv_path)


if __name__ == "__main__":
    main()
