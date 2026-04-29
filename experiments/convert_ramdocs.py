#!/usr/bin/env python3
"""Convert RAMDocs from Hugging Face into the project's JSON schema."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import Document, QueryInstance, save_dataset

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "HanNight/RAMDocs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert RAMDocs from Hugging Face to local JSON.")
    p.add_argument(
        "--split",
        default="test",
        help="HF split to convert (RAMDocs currently provides test).",
    )
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--limit", type=int, default=None, help="Optional max examples to convert.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def _load_hf_dataset(split: str):
    datasets_mod = importlib.import_module("datasets")
    return datasets_mod.load_dataset(HF_DATASET_NAME, split=split)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _pick_correct_answer(example: dict[str, Any], docs: list[dict[str, Any]]) -> str:
    gold_answers = example.get("gold_answers", [])
    for ans in gold_answers:
        x = _norm(ans)
        if x:
            return x

    for doc in docs:
        if _norm(doc.get("type", "")).lower() == "correct":
            ans = _norm(doc.get("answer", ""))
            if ans and ans.lower() != "unknown":
                return ans

    for doc in docs:
        ans = _norm(doc.get("answer", ""))
        if ans and ans.lower() != "unknown":
            return ans
    return ""


def _doc_credibility(doc_type: str) -> str:
    t = _norm(doc_type).lower()
    if t == "correct":
        return "high"
    if t == "misinfo":
        return "low"
    return "unknown"


def _doc_tone(doc_type: str) -> str:
    t = _norm(doc_type).lower()
    if t == "misinfo":
        return "assertive"
    if t == "correct":
        return "neutral"
    return "original"


def _convert_example(example: dict[str, Any], idx: int, split: str) -> QueryInstance:
    docs_raw = list(example.get("documents", []) or [])
    if not docs_raw:
        raise ValueError("missing documents")

    correct_answer = _pick_correct_answer(example, docs_raw)
    documents: list[Document] = []
    for i, doc in enumerate(docs_raw):
        doc_type = _norm(doc.get("type", ""))
        answer = _norm(doc.get("answer", ""))
        if not answer or answer.lower() == "unknown":
            answer = ""
        documents.append(
            Document(
                id=f"ramdocs_{idx}_doc_{i+1}",
                content=_norm(doc.get("text", "")),
                source=f"RAMDocs:{doc_type or 'unknown'}",
                credibility=_doc_credibility(doc_type),
                supports_answer=answer,
                tone=_doc_tone(doc_type),
                metadata={
                    "ramdocs_doc_type": doc_type,
                    "ramdocs_doc_answer": _norm(doc.get("answer", "")),
                },
            )
        )

    if not correct_answer:
        raise ValueError("missing correct answer")

    qid = _norm(example.get("id", "")) or f"ramdocs_{split}_{idx+1}"
    return QueryInstance(
        id=qid,
        question=_norm(example.get("question", "")),
        documents=documents,
        correct_answer=correct_answer,
        metadata={
            "source_benchmark": "ramdocs",
            "hf_dataset": HF_DATASET_NAME,
            "hf_split": split,
            "disambig_entity": list(example.get("disambig_entity", []) or []),
            "gold_answers": list(example.get("gold_answers", []) or []),
            "wrong_answers": list(example.get("wrong_answers", []) or []),
        },
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    logger.info("Loading %s split=%s", HF_DATASET_NAME, args.split)
    dataset = _load_hf_dataset(args.split)

    instances: list[QueryInstance] = []
    for i, row in enumerate(dataset):
        if args.limit is not None and i >= args.limit:
            break
        try:
            instances.append(_convert_example(dict(row), i, args.split))
        except Exception as exc:
            logger.warning("Skipping RAMDocs row %d: %s", i, exc)

    save_dataset(instances, args.output)
    logger.info("Saved %d RAMDocs instances to %s", len(instances), args.output)


if __name__ == "__main__":
    main()
