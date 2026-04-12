#!/usr/bin/env python3
"""
Main experiment runner for the tone-bias study.

Runs Standard RAG and/or MADAM-RAG on a given dataset and computes
evaluation metrics (Substance Alignment Rate, Tone Bias Rate).

Usage
-----
    python -m experiments.run_experiment \
        --dataset data/samples/sample_dataset.json \
        --systems standard_rag madam_rag \
        --provider openai \
        --output results/run_01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Ensure repo root is on the Python path so that ``import config`` and
# ``import src.*`` resolve correctly regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.data_loader import load_dataset, QueryInstance
from src.models import LLMClient
from src.standard_rag import StandardRAG
from src.madam_rag import MadamRAG
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


def _run_standard_rag(
    instances: list[QueryInstance],
    client: LLMClient,
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
            )
        )
    return trials


def _run_madam_rag(
    instances: list[QueryInstance],
    agent_client: LLMClient,
    judge_client: LLMClient | None = None,
    num_rounds: int | None = None,
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
        help="Path to the dataset JSON file.",
    )
    p.add_argument(
        "--systems", nargs="+",
        choices=["standard_rag", "madam_rag"],
        default=["standard_rag", "madam_rag"],
        help="Which systems to evaluate.",
    )
    p.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider for all pipeline components.",
    )
    p.add_argument(
        "--model", default=None,
        help="Override the default model name for the chosen provider.",
    )
    p.add_argument(
        "--debate-rounds", type=int, default=None,
        help="Number of MADAM-RAG debate rounds (default from config).",
    )
    p.add_argument(
        "--output", default=None,
        help="Directory for result files.  Defaults to results/<provider>.",
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
    instances = load_dataset(args.dataset)
    logger.info("Loaded %d instances from %s", len(instances), args.dataset)

    # Build LLM client
    client = _build_client(args.provider, args.model)
    logger.info("Using %s", client)

    output_dir = args.output or os.path.join(
        config.RESULTS_DIR, args.provider
    )
    os.makedirs(output_dir, exist_ok=True)

    # Run requested systems
    for system_name in args.systems:
        logger.info("=== Running %s ===", system_name)

        if system_name == "standard_rag":
            trials = _run_standard_rag(instances, client)
        elif system_name == "madam_rag":
            trials = _run_madam_rag(
                instances, client, num_rounds=args.debate_rounds,
            )
        else:
            raise ValueError(f"Unknown system: {system_name}")

        metrics = compute_metrics(trials, system_name, client.model)
        print_metrics(metrics)
        path = save_results(metrics, output_dir, tag=system_name)
        logger.info("Saved %s results to %s", system_name, path)


if __name__ == "__main__":
    main()
