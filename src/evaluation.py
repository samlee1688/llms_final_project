"""
Evaluation metrics for the tone-bias experiments.

Primary metrics from the proposal:
  - **Substance Alignment Rate (SAR)**: proportion of trials where the
    final answer reflects the correct, higher-credibility claim.
  - **Tone Bias Rate (TBR)**: proportion of trials where the system
    favours the assertive-but-incorrect claim over the better-evidenced
    neutral one.

Both metrics require an answer-matching function that determines which
document (or candidate answer) the system's output aligns with.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data_loader import QueryInstance

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Answer matching
# -----------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def answer_aligns_with(system_output: str, candidate_answer: str) -> bool:
    """Check whether *system_output* contains or clearly states
    *candidate_answer*.

    This is a conservative substring / keyword match.  For more nuanced
    evaluation (e.g. paraphrase detection) an LLM-based judge can be
    swapped in later.
    """
    norm_out = normalise(system_output)
    norm_ans = normalise(candidate_answer)
    return norm_ans in norm_out


def classify_answer(
    system_output: str,
    instance: QueryInstance,
) -> str:
    """Classify a system output as ``"correct"``, ``"incorrect"``, or
    ``"unclear"``.

    * ``"correct"``  – the output aligns with ``instance.correct_answer``.
    * ``"incorrect"`` – the output aligns with an *incorrect* document's
      answer but **not** with the correct answer.
    * ``"unclear"``  – the output does not clearly match either.
    """
    matches_correct = answer_aligns_with(system_output, instance.correct_answer)

    incorrect_answers = {
        d.supports_answer
        for d in instance.incorrect_documents
        if d.supports_answer
    }
    matches_incorrect = any(
        answer_aligns_with(system_output, ans) for ans in incorrect_answers
    )

    if matches_correct and not matches_incorrect:
        return "correct"
    if matches_incorrect and not matches_correct:
        return "incorrect"
    if matches_correct and matches_incorrect:
        # Ambiguous — count as unclear
        return "unclear"
    return "unclear"


# -----------------------------------------------------------------------
# Per-instance result
# -----------------------------------------------------------------------

@dataclass
class TrialResult:
    """Evaluation result for a single query instance."""

    query_id: str
    system: str                 # "standard_rag" | "madam_rag" | "eajp"
    model: str
    classification: str         # "correct" | "incorrect" | "unclear"
    final_answer: str
    correct_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------
# Aggregate metrics
# -----------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    """Aggregate metrics over a set of trials."""

    system: str
    model: str
    total: int
    correct: int
    incorrect: int
    unclear: int
    substance_alignment_rate: float     # correct / total
    tone_bias_rate: float               # incorrect / total (on TCB subset)
    accuracy: float                     # correct / (total - unclear)
    per_instance: List[TrialResult] = field(default_factory=list)


def compute_metrics(
    trials: List[TrialResult],
    system_name: str = "",
    model_name: str = "",
) -> ExperimentMetrics:
    """Compute aggregate metrics from a list of trial results.

    Parameters
    ----------
    trials:
        Individual trial results (one per query instance).
    system_name / model_name:
        Labels for the output metrics object.
    """
    total = len(trials)
    correct = sum(1 for t in trials if t.classification == "correct")
    incorrect = sum(1 for t in trials if t.classification == "incorrect")
    unclear = total - correct - incorrect

    denom = total if total > 0 else 1
    resolved = (correct + incorrect) if (correct + incorrect) > 0 else 1

    return ExperimentMetrics(
        system=system_name,
        model=model_name,
        total=total,
        correct=correct,
        incorrect=incorrect,
        unclear=unclear,
        substance_alignment_rate=correct / denom,
        tone_bias_rate=incorrect / denom,
        accuracy=correct / resolved,
        per_instance=trials,
    )


# -----------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------

def metrics_to_dict(m: ExperimentMetrics) -> dict:
    """Serialise metrics (excluding per-instance detail) to a plain dict."""
    return {
        "system": m.system,
        "model": m.model,
        "total": m.total,
        "correct": m.correct,
        "incorrect": m.incorrect,
        "unclear": m.unclear,
        "substance_alignment_rate": round(m.substance_alignment_rate, 4),
        "tone_bias_rate": round(m.tone_bias_rate, 4),
        "accuracy": round(m.accuracy, 4),
    }


def save_results(
    metrics: ExperimentMetrics,
    output_dir: str,
    tag: str = "",
) -> str:
    """Write metrics and per-instance results to JSON files.

    Returns the path to the summary JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{tag}_" if tag else ""

    summary_path = os.path.join(output_dir, f"{prefix}metrics.json")
    with open(summary_path, "w") as f:
        json.dump(metrics_to_dict(metrics), f, indent=2)

    detail_path = os.path.join(output_dir, f"{prefix}details.json")
    details = [
        {
            "query_id": t.query_id,
            "system": t.system,
            "model": t.model,
            "classification": t.classification,
            "final_answer": t.final_answer,
            "correct_answer": t.correct_answer,
            "metadata": t.metadata,
        }
        for t in metrics.per_instance
    ]
    with open(detail_path, "w") as f:
        json.dump(details, f, indent=2)

    logger.info("Results saved to %s", output_dir)
    return summary_path


def print_metrics(metrics: ExperimentMetrics) -> None:
    """Print a human-readable summary table to stdout."""
    d = metrics_to_dict(metrics)
    print(f"\n{'=' * 50}")
    print(f"  System: {d['system']}  |  Model: {d['model']}")
    print(f"{'=' * 50}")
    print(f"  Total instances       : {d['total']}")
    print(f"  Correct               : {d['correct']}")
    print(f"  Incorrect             : {d['incorrect']}")
    print(f"  Unclear               : {d['unclear']}")
    print(f"  Substance Alignment   : {d['substance_alignment_rate']:.2%}")
    print(f"  Tone Bias Rate        : {d['tone_bias_rate']:.2%}")
    print(f"  Accuracy (resolved)   : {d['accuracy']:.2%}")
    print(f"{'=' * 50}\n")
