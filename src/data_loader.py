"""
Data models and loaders for RAG tone-bias benchmarks.

Supports loading from:
  - AmbigDocs (ambiguous-interpretation QA pairs)
  - FaithEval (faithfulness evaluation with contradictory docs)
  - Tone-Controlled Benchmark (TCB) derived from the above

All loaders produce a uniform list of ``QueryInstance`` objects.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Core data classes
# -----------------------------------------------------------------------

@dataclass
class Document:
    """A single retrieved document associated with a query."""

    id: str
    content: str
    source: str = ""
    credibility: str = "unknown"       # "high", "medium", "low", "unknown"
    supports_answer: str = ""           # the answer this document supports
    tone: str = "original"              # "assertive", "neutral", "original"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryInstance:
    """One evaluation instance: a question together with its retrieved
    documents and a ground-truth correct answer."""

    id: str
    question: str
    documents: List[Document]
    correct_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def correct_documents(self) -> List[Document]:
        """Documents whose ``supports_answer`` matches the correct answer."""
        return [d for d in self.documents if d.supports_answer == self.correct_answer]

    @property
    def incorrect_documents(self) -> List[Document]:
        """Documents supporting an answer other than the correct one."""
        return [
            d for d in self.documents
            if d.supports_answer and d.supports_answer != self.correct_answer
        ]


# -----------------------------------------------------------------------
# Serialisation helpers
# -----------------------------------------------------------------------

def _doc_from_dict(d: dict) -> Document:
    return Document(
        id=d["id"],
        content=d["content"],
        source=d.get("source", ""),
        credibility=d.get("credibility", "unknown"),
        supports_answer=d.get("supports_answer", ""),
        tone=d.get("tone", "original"),
        metadata=d.get("metadata", {}),
    )


def _instance_from_dict(d: dict) -> QueryInstance:
    return QueryInstance(
        id=d["id"],
        question=d["question"],
        documents=[_doc_from_dict(doc) for doc in d["documents"]],
        correct_answer=d["correct_answer"],
        metadata=d.get("metadata", {}),
    )


def instance_to_dict(qi: QueryInstance) -> dict:
    """Serialise a ``QueryInstance`` back to a plain dict (for JSON output)."""
    return {
        "id": qi.id,
        "question": qi.question,
        "documents": [
            {
                "id": doc.id,
                "content": doc.content,
                "source": doc.source,
                "credibility": doc.credibility,
                "supports_answer": doc.supports_answer,
                "tone": doc.tone,
                "metadata": doc.metadata,
            }
            for doc in qi.documents
        ],
        "correct_answer": qi.correct_answer,
        "metadata": qi.metadata,
    }


# -----------------------------------------------------------------------
# Generic JSON loader
# -----------------------------------------------------------------------

def load_dataset(path: str) -> List[QueryInstance]:
    """Load a dataset from a JSON file.

    Expected format – a JSON array of objects, each with at minimum::

        {
            "id": "...",
            "question": "...",
            "documents": [ { "id": "...", "content": "..." }, ... ],
            "correct_answer": "..."
        }
    """
    logger.info("Loading dataset from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "instances" in raw:
        raw = raw["instances"]

    instances = [_instance_from_dict(item) for item in raw]
    logger.info("Loaded %d instances from %s", len(instances), path)
    return instances


def save_dataset(instances: List[QueryInstance], path: str) -> None:
    """Write a list of ``QueryInstance`` objects to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([instance_to_dict(qi) for qi in instances], f, indent=2)
    logger.info("Saved %d instances to %s", len(instances), path)


# -----------------------------------------------------------------------
# Benchmark-specific loaders (thin wrappers for now)
# -----------------------------------------------------------------------

def load_ambigdocs(path: str) -> List[QueryInstance]:
    """Load the AmbigDocs benchmark.

    Applies any AmbigDocs-specific normalisation on top of the generic
    loader. Currently the generic format suffices.
    """
    return load_dataset(path)


def load_faitheval(
    path: str, require_correct_doc: bool = True
) -> List[QueryInstance]:
    """Load a FaithEval subset.

    Parameters
    ----------
    require_correct_doc:
        If *True* (default), filter to instances where at least one
        document supports the correct answer — matching the paper's
        requirement that the task is selecting the right claim, not
        detecting its absence.
    """
    instances = load_dataset(path)
    if require_correct_doc:
        before = len(instances)
        instances = [qi for qi in instances if qi.correct_documents]
        logger.info(
            "FaithEval: kept %d / %d instances with ≥1 correct document",
            len(instances),
            before,
        )
    return instances


def load_tcb(path: str) -> List[QueryInstance]:
    """Load a Tone-Controlled Benchmark (TCB) dataset.

    TCB is treated as a standard dataset file with tone-controlled
    instances already encoded in the JSON. The loader is a named alias to
    make experiment code self-documenting.
    """
    return load_dataset(path)


def load_benchmark(path: str, benchmark: str) -> List[QueryInstance]:
    """Load one of the supported benchmark families by name."""
    benchmark = benchmark.lower()
    if benchmark == "ambigdocs":
        return load_ambigdocs(path)
    if benchmark == "faitheval":
        return load_faitheval(path, require_correct_doc=True)
    if benchmark == "tcb":
        return load_tcb(path)
    return load_dataset(path)
