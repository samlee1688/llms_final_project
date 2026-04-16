#!/usr/bin/env python3
"""Main experiment runner for the tone-bias study.

Runs Standard RAG, MADAM-RAG, and EAJP on AmbigDocs, FaithEval, and
Tone-Controlled Benchmark style datasets with GPT-5.2 and Claude Sonnet
4.6 judge suites.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

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


def _suite_configs(judge_suite: str) -> list[tuple[str, LLMClient]]:
    """Return the judge model configurations to evaluate."""
    if judge_suite == "openai":
        return [("openai_gpt5.2", _build_client("openai", config.OPENAI_MODEL))]
    if judge_suite == "anthropic":
        return [("anthropic_claude4.6", _build_client("anthropic", config.ANTHROPIC_MODEL))]
    return [
        ("openai_gpt5.2", _build_client("openai", config.OPENAI_MODEL)),
        ("anthropic_claude4.6", _build_client("anthropic", config.ANTHROPIC_MODEL)),
    ]


def _run_standard_rag(
    instances: list[QueryInstance],
    client: LLMClient,
    benchmark_name: str = "custom",
) -> list[TrialResult]:
    pipeline = StandardRAG(client)
    trials: list[TrialResult] = []
    for inst in instances:
        out = pipeline.answer(inst)
        classification = classify_answer(out.final_answer, inst)
        trials.append(
            TrialResult(
                query_id=inst.id,
                system="standard_rag",
                model=out.model,
                classification=classification,
                final_answer=out.final_answer,
                correct_answer=inst.correct_answer,
                metadata={
                    "benchmark": benchmark_name,
                    "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                },
            )
        )
    return trials


def _run_madam_rag(
    instances: list[QueryInstance],
    agent_client: LLMClient,
    judge_client: LLMClient | None = None,
    num_rounds: int | None = None,
    benchmark_name: str = "custom",
) -> list[TrialResult]:
    pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
    )
    trials: list[TrialResult] = []
    for inst in instances:
        out = pipeline.answer(inst)
        classification = classify_answer(out.final_answer, inst)
        trials.append(
            TrialResult(
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
                },
            )
        )
    return trials


def _run_eajp(
    instances: list[QueryInstance],
    agent_client: LLMClient,
    judge_client: LLMClient | None = None,
    num_rounds: int | None = None,
    benchmark_name: str = "custom",
) -> list[TrialResult]:
    pipeline = MadamRAG(
        agent_llm=agent_client,
        judge_llm=judge_client,
        num_rounds=num_rounds,
        judge=EvidenceAnchoredAggregator(judge_client or agent_client),
    )
    trials: list[TrialResult] = []
    for inst in instances:
        out = pipeline.answer(inst)
        classification = classify_answer(out.final_answer, inst)
        trials.append(
            TrialResult(
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
                },
            )
        )
    return trials


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
    logger.info("Loaded %d instances from %s", len(instances), args.dataset)

    # Build the agent client once and judge clients per suite.
    agent_client = _build_client(args.agent_provider, args.agent_model)
    logger.info("Agent client: %s", agent_client)

    output_dir = args.output or os.path.join(config.RESULTS_DIR, args.benchmark)
    os.makedirs(output_dir, exist_ok=True)

    suite_configs = _suite_configs(args.judge_suite)

    # Standard RAG does not depend on the judge suite, so run it once.
    if "standard_rag" in args.systems:
        logger.info("=== Running standard_rag (suite-independent) ===")
        trials = _run_standard_rag(instances, agent_client, benchmark_name=args.benchmark)
        metrics = compute_metrics(trials, "standard_rag", agent_client.model)
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

            if system_name == "standard_rag":
                trials = _run_standard_rag(instances, agent_client, benchmark_name=args.benchmark)
                model_name = agent_client.model
            elif system_name == "madam_rag":
                trials = _run_madam_rag(
                    instances,
                    agent_client,
                    judge_client=judge_client,
                    num_rounds=args.judge_rounds,
                    benchmark_name=args.benchmark,
                )
                model_name = judge_client.model
            elif system_name == "eajp":
                pipeline = MadamRAG(
                    agent_llm=agent_client,
                    judge_llm=judge_client,
                    num_rounds=args.judge_rounds,
                    judge=EvidenceAnchoredAggregator(judge_client),
                )
                trials = []
                for inst in instances:
                    out = pipeline.answer(inst)
                    trials.append(
                        TrialResult(
                            query_id=inst.id,
                            system="eajp",
                            model=out.model,
                            classification=classify_answer(out.final_answer, inst),
                            final_answer=out.final_answer,
                            correct_answer=inst.correct_answer,
                            metadata={
                                "num_debate_rounds": pipeline.num_rounds,
                                "num_agents": len(inst.documents),
                                "judge_mode": "eajp",
                                "benchmark": args.benchmark,
                                "tone_controlled": bool(inst.metadata.get("tone_controlled", False)),
                            },
                        )
                    )
                model_name = judge_client.model
            else:
                raise ValueError(f"Unknown system: {system_name}")

            metrics = compute_metrics(trials, system_name, model_name)
            print_metrics(metrics)
            path = save_results(metrics, suite_output_dir, tag=system_name)
            logger.info("Saved %s results to %s", system_name, path)


if __name__ == "__main__":
    main()
