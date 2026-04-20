#!/usr/bin/env python3
"""Convert AmbigDocs from Hugging Face into the project's JSON schema.

This script loads the public dataset directly from Hugging Face with the
`datasets` library, converts each example into the local `QueryInstance`
format, and saves a JSON file that can be used by the experiment runner.

By default, the first document is treated as the gold document for
`correct_answer`. You can override this with `--gold-doc-index`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import Document, QueryInstance, save_dataset

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "yoonsanglee/AmbigDocs"


def _split_file(split: str) -> str:
    mapping = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }
    return mapping[split]


def _load_split_rows(split: str) -> list[dict[str, Any]]:
    """Load AmbigDocs rows for a split.

    Preferred path is `datasets.load_dataset`; if that fails due upstream
    schema incompatibilities, fall back to downloading the split JSON file
    directly from the dataset repo.
    """
    datasets_mod = importlib.import_module("datasets")
    try:
        raw = datasets_mod.load_dataset(HF_DATASET_NAME, split=split)
        return [dict(row) for row in raw]
    except Exception as exc:
        logger.warning(
            "datasets.load_dataset failed (%s). Falling back to direct JSON download.",
            exc,
        )

    hf_hub = importlib.import_module("huggingface_hub")
    split_path = hf_hub.hf_hub_download(
        repo_id=HF_DATASET_NAME,
        filename=_split_file(split),
        repo_type="dataset",
    )
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Unexpected AmbigDocs payload type: {type(payload).__name__}")
    return [dict(row) for row in payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AmbigDocs from Hugging Face to local JSON."
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Which Hugging Face split to convert.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--gold-doc-index",
        type=int,
        default=0,
        help="Document index to treat as the correct answer source.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to convert.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _convert_document(doc: dict[str, Any], index: int) -> Document:
    return Document(
        id=str(doc.get("pid", f"doc_{index + 1}")),
        content=str(doc.get("text", "")),
        source=str(doc.get("title", "")),
        credibility="unknown",
        supports_answer=str(doc.get("answer", "")),
        tone="original",
        metadata={
            "hf_pid": doc.get("pid"),
            "hf_title": doc.get("title"),
        },
    )


def _convert_example(example: dict[str, Any], gold_doc_index: int) -> QueryInstance:
    docs = example.get("documents", [])
    documents = [_convert_document(doc, i) for i, doc in enumerate(docs)]

    if not documents:
        raise ValueError(f"Example {example.get('qid')} has no documents")

    if gold_doc_index < 0 or gold_doc_index >= len(documents):
        gold_doc_index = 0

    gold_doc = documents[gold_doc_index]

    return QueryInstance(
        id=str(example.get("qid", "")),
        question=str(example.get("question", "")),
        documents=documents,
        correct_answer=gold_doc.supports_answer,
        metadata={
            "source_benchmark": "ambigdocs",
            "hf_dataset": HF_DATASET_NAME,
            "hf_split": example.get("_split", ""),
            "ambiguous_entity": example.get("ambiguous_entity", ""),
            "gold_doc_index": gold_doc_index,
            "gold_doc_id": gold_doc.id,
        },
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    logger.info("Loading %s split=%s", HF_DATASET_NAME, args.split)
    raw_rows = _load_split_rows(args.split)

    instances: list[QueryInstance] = []
    for i, example in enumerate(raw_rows):
        if args.limit is not None and i >= args.limit:
            break
        item = dict(example)
        item["_split"] = args.split
        try:
            instances.append(_convert_example(item, args.gold_doc_index))
        except Exception as exc:
            logger.warning("Skipping example %s: %s", example.get("qid"), exc)

    save_dataset(instances, args.output)
    logger.info("Saved %d converted examples to %s", len(instances), args.output)


if __name__ == "__main__":
    main()
