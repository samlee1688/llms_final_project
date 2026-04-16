"""Tone-Controlled Benchmark (TCB) construction utilities.

TCB is built by taking instances with conflicting retrieved documents and
rewriting the relevant documents into neutral vs assertive registers while
preserving meaning. If Gemini is available, it is used for the rewrite step;
otherwise a deterministic local fallback is used so the pipeline still works.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from src.data_loader import Document, QueryInstance, instance_to_dict, save_dataset
from src.prompts import TCB_REWRITE_SYSTEM, TCB_REWRITE_USER

logger = logging.getLogger(__name__)


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _credibility_rank(value: str) -> int:
    order = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    return order.get((value or "unknown").lower(), 0)


def _best_correct_doc(instance: QueryInstance) -> Optional[Document]:
    docs = instance.correct_documents
    if not docs:
        return None
    return sorted(docs, key=lambda d: _credibility_rank(d.credibility), reverse=True)[0]


def _best_incorrect_doc(instance: QueryInstance) -> Optional[Document]:
    docs = instance.incorrect_documents
    if not docs:
        return None
    return sorted(docs, key=lambda d: _credibility_rank(d.credibility))[0]


def _local_rewrite(content: str, tone: str) -> str:
    content = _normalise_text(content)
    if tone == "assertive":
        return (
            "This is clear and well-established: "
            + content.replace("may", "does").replace("might", "does").replace("suggests", "shows")
        )
    # neutral
    return (
        "Available evidence suggests that "
        + content.replace("well-established", "supported")
        .replace("clearly", "likely")
        .replace("definitely", "likely")
    )


class ToneRewriter:
    """Rewrite factual text into a requested rhetorical tone."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = config.GEMINI_MODEL,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or config.GEMINI_API_KEY
        self._client = None

        if provider == "gemini" and self.api_key:
            try:
                from google import genai  # type: ignore

                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Gemini unavailable (%s); falling back to local rewrites.", exc)
                self._client = None

    def rewrite(self, content: str, tone: str) -> str:
        if self._client is None:
            return _local_rewrite(content, tone)

        prompt = TCB_REWRITE_USER.format(content=content, tone=tone)
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": TCB_REWRITE_SYSTEM,
                "temperature": 0.0,
                "max_output_tokens": 1024,
            },
        )
        data = _extract_json(getattr(response, "text", "") or "")
        rewritten = str(data.get("rewritten_content", "")).strip()
        return rewritten or _local_rewrite(content, tone)


@dataclass
class TCBInstanceMetadata:
    source_benchmark: str
    original_instance_id: str
    correct_doc_id: str
    incorrect_doc_id: str
    tone_target: str = "neutral_vs_assertive"
    meaning_preserved: bool = True
    rewrite_model: str = config.GEMINI_MODEL
    rewrite_provider: str = "gemini"


class ToneControlledBenchmarkBuilder:
    """Construct a TCB dataset from an existing benchmark."""

    def __init__(self, rewriter: Optional[ToneRewriter] = None):
        self.rewriter = rewriter or ToneRewriter()

    def build_instance(
        self,
        instance: QueryInstance,
        source_benchmark: str = "unknown",
    ) -> Optional[QueryInstance]:
        correct_doc = _best_correct_doc(instance)
        incorrect_doc = _best_incorrect_doc(instance)
        if not correct_doc or not incorrect_doc:
            return None

        neutral_content = self.rewriter.rewrite(correct_doc.content, "neutral")
        assertive_content = self.rewriter.rewrite(incorrect_doc.content, "assertive")

        docs = [
            Document(
                id=f"{instance.id}_correct_neutral",
                content=neutral_content,
                source=correct_doc.source,
                credibility=correct_doc.credibility or "high",
                supports_answer=correct_doc.supports_answer,
                tone="neutral",
                metadata={**correct_doc.metadata, "tone_controlled": True, "original_doc_id": correct_doc.id},
            ),
            Document(
                id=f"{instance.id}_incorrect_assertive",
                content=assertive_content,
                source=incorrect_doc.source,
                credibility=incorrect_doc.credibility or "low",
                supports_answer=incorrect_doc.supports_answer,
                tone="assertive",
                metadata={**incorrect_doc.metadata, "tone_controlled": True, "original_doc_id": incorrect_doc.id},
            ),
        ]

        metadata = {
            **instance.metadata,
            "constructed_from": instance.id,
            "source_benchmark": source_benchmark,
            "tone_controlled": True,
            "correct_doc_id": correct_doc.id,
            "incorrect_doc_id": incorrect_doc.id,
            "rewrite_provider": self.rewriter.provider,
            "rewrite_model": self.rewriter.model,
        }
        return QueryInstance(
            id=f"tcb_{instance.id}",
            question=instance.question,
            documents=docs,
            correct_answer=instance.correct_answer,
            metadata=metadata,
        )

    def build_dataset(
        self,
        instances: List[QueryInstance],
        source_benchmark: str = "unknown",
    ) -> List[QueryInstance]:
        built: List[QueryInstance] = []
        for instance in instances:
            created = self.build_instance(instance, source_benchmark=source_benchmark)
            if created is not None:
                built.append(created)
        return built

    def build_and_save(
        self,
        instances: List[QueryInstance],
        output_path: str,
        source_benchmark: str = "unknown",
    ) -> List[QueryInstance]:
        built = self.build_dataset(instances, source_benchmark=source_benchmark)
        save_dataset(built, output_path)
        logger.info("Saved %d TCB instances to %s", len(built), output_path)
        return built


__all__ = ["ToneRewriter", "ToneControlledBenchmarkBuilder", "TCBInstanceMetadata"]