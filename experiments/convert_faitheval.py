#!/usr/bin/env python3
"""Convert FaithEval from Hugging Face into the project's JSON schema.

This script loads the public FaithEval counterfactual and inconsistent
splits directly from Hugging Face with the `datasets` library, converts
rows into local `QueryInstance` objects, and saves a JSON file that can be
used by the experiment runner.

The resulting instances contain two documents:
- a neutral/high-credibility document supporting the correct answer
- an assertive/low-credibility document supporting the competing answer

For counterfactual rows, the correct answer is the provided `answer`.
For inconsistent rows, the correct answer is inferred from the modified
context; by default we use the second answer entry, which corresponds to
the new answer in the dataset card examples.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
from typing import Any, Iterable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import Document, QueryInstance, save_dataset

logger = logging.getLogger(__name__)

HF_DATASETS = {
    "counterfactual": "Salesforce/FaithEval-counterfactual-v1.0",
    "inconsistent": "Salesforce/FaithEval-inconsistent-v1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FaithEval datasets from Hugging Face to local JSON."
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        choices=list(HF_DATASETS.keys()),
        default=["counterfactual", "inconsistent"],
        help="Which FaithEval datasets to convert.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="HF split to load (FaithEval uses test by default).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to convert per dataset.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _load_hf_dataset(dataset_name: str, split: str):
    datasets_mod = importlib.import_module("datasets")
    return datasets_mod.load_dataset(dataset_name, split=split)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _replace_one(text: str, old: str, new: str) -> str:
    if not old:
        return text
    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)
    return pattern.sub(new, text, count=1)


def _synthetic_conflicting_doc(context: str, correct_answer: str, wrong_answer: str) -> str:
    """Create a lightweight conflicting variant of the context.

    If the correct answer appears in the context, swap it for the wrong
    answer once. Otherwise prepend a short claim and keep the original
    context intact.
    """
    context = _collapse_ws(context)
    swapped = _replace_one(context, correct_answer, wrong_answer)
    if swapped != context:
        return swapped
    return (
        f"This passage argues that the answer is {wrong_answer}. "
        f"{context}"
    )


def _pick_wrong_choice(choices: dict[str, Any], correct_answer: str) -> str:
    texts = list(choices.get("text", []))
    for text in texts:
        if _collapse_ws(text).lower() != _collapse_ws(correct_answer).lower():
            return text
    return texts[0] if texts else ""


def _infer_inconsistent_correct_answer(example: dict[str, Any]) -> str:
    answers = list(example.get("answers", []))
    justification = str(example.get("justification", ""))

    match = re.search(r"new answer, ['\"]([^'\"]+)['\"]", justification, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"new answer ['\"]([^'\"]+)['\"]", justification, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for ans in answers:
            if _collapse_ws(ans).lower() == _collapse_ws(candidate).lower():
                return ans

    if len(answers) >= 2:
        return answers[1]
    if answers:
        return answers[0]
    return ""


def _make_instance(
    *,
    qid: str,
    question: str,
    context: str,
    correct_answer: str,
    wrong_answer: str,
    source_benchmark: str,
    subset: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> QueryInstance:
    correct_doc = Document(
        id=f"{qid}_correct",
        content=_collapse_ws(context),
        source=f"FaithEval:{source_benchmark}",
        credibility="high",
        supports_answer=correct_answer,
        tone="neutral",
        metadata={
            "faitheval_role": "correct",
            "source_benchmark": source_benchmark,
            "subset": subset,
        },
    )
    incorrect_doc = Document(
        id=f"{qid}_incorrect",
        content=_synthetic_conflicting_doc(context, correct_answer, wrong_answer),
        source=f"FaithEval:{source_benchmark}",
        credibility="low",
        supports_answer=wrong_answer,
        tone="assertive",
        metadata={
            "faitheval_role": "incorrect",
            "source_benchmark": source_benchmark,
            "subset": subset,
        },
    )

    metadata = {
        "source_benchmark": "faitheval",
        "faitheval_subset": source_benchmark,
        "subset": subset,
        "tone_controlled": True,
        "correct_doc_id": correct_doc.id,
        "incorrect_doc_id": incorrect_doc.id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return QueryInstance(
        id=f"faitheval_{source_benchmark}_{qid}",
        question=question,
        documents=[correct_doc, incorrect_doc],
        correct_answer=correct_answer,
        metadata=metadata,
    )


def _convert_counterfactual(example: dict[str, Any]) -> QueryInstance:
    correct_answer = str(example.get("answer", ""))
    wrong_answer = _pick_wrong_choice(example.get("choices", {}), correct_answer)
    if not wrong_answer:
        wrong_answer = "Incorrect alternative"

    return _make_instance(
        qid=str(example.get("id", "")),
        question=str(example.get("question", "")),
        context=str(example.get("context", "")),
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        source_benchmark="counterfactual",
        subset=str(example.get("num of options", "")),
        extra_metadata={
            "answer_key": example.get("answerKey", ""),
            "num_options": example.get("num of options", ""),
        },
    )


def _convert_inconsistent(example: dict[str, Any]) -> QueryInstance:
    correct_answer = _infer_inconsistent_correct_answer(example)
    answers = [str(a) for a in example.get("answers", [])]
    wrong_answer = next((a for a in answers if _collapse_ws(a).lower() != _collapse_ws(correct_answer).lower()), "")
    if not wrong_answer:
        wrong_answer = "Incorrect alternative"

    return _make_instance(
        qid=str(example.get("qid", "")),
        question=str(example.get("question", "")),
        context=str(example.get("context", "")),
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        source_benchmark="inconsistent",
        subset=str(example.get("subset", "")),
        extra_metadata={
            "answers": answers,
            "justification": example.get("justification", ""),
        },
    )


def _iter_rows(dataset, limit: int | None) -> Iterable[dict[str, Any]]:
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        yield dict(row)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    instances: list[QueryInstance] = []
    for dataset_type in args.dataset_types:
        hf_name = HF_DATASETS[dataset_type]
        logger.info("Loading %s (%s) split=%s", dataset_type, hf_name, args.split)
        dataset = _load_hf_dataset(hf_name, args.split)

        for example in _iter_rows(dataset, args.limit):
            try:
                if dataset_type == "counterfactual":
                    instances.append(_convert_counterfactual(example))
                else:
                    instances.append(_convert_inconsistent(example))
            except Exception as exc:
                logger.warning(
                    "Skipping %s example %s: %s",
                    dataset_type,
                    example.get("id") or example.get("qid"),
                    exc,
                )

    save_dataset(instances, args.output)
    logger.info("Saved %d FaithEval instances to %s", len(instances), args.output)


if __name__ == "__main__":
    main()
