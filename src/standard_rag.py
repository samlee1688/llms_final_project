"""
Standard RAG pipeline.

Given a query and a set of retrieved documents, concatenate them into a
single context block and prompt the LLM to produce an answer.  This serves
as the baseline system in the tone-bias experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data_loader import Document, QueryInstance
from src.models import LLMClient, LLMResponse
from src.prompts import (
    STANDARD_RAG_SYSTEM,
    STANDARD_RAG_USER,
    format_documents,
)

logger = logging.getLogger(__name__)


@dataclass
class StandardRAGOutput:
    """Result of running a single query through the standard RAG pipeline."""

    query_id: str
    final_answer: str
    raw_response: str
    model: str
    usage: dict = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StandardRAG:
    """Single-pass retrieve-then-generate RAG pipeline.

    Parameters
    ----------
    llm_client:
        An initialised ``LLMClient`` (OpenAI or Anthropic).
    include_source:
        Whether to show source names in the document context provided to
        the model.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        include_source: bool = True,
    ):
        self.llm = llm_client
        self.include_source = include_source

    def answer(
        self,
        instance: QueryInstance,
        documents: Optional[List[Document]] = None,
    ) -> StandardRAGOutput:
        """Run the pipeline for a single query instance.

        Parameters
        ----------
        instance:
            The evaluation instance (query + documents + correct answer).
        documents:
            If provided, use these documents instead of ``instance.documents``.
            Useful for running on tone-rewritten variants.

        Returns
        -------
        StandardRAGOutput
        """
        docs = documents if documents is not None else instance.documents
        formatted_docs = format_documents(docs, include_source=self.include_source)

        user_prompt = STANDARD_RAG_USER.format(
            formatted_documents=formatted_docs,
            question=instance.question,
        )

        logger.debug("Standard RAG prompt for %s:\n%s", instance.id, user_prompt)
        response: LLMResponse = self.llm.generate(STANDARD_RAG_SYSTEM, user_prompt)

        return StandardRAGOutput(
            query_id=instance.id,
            final_answer=response.text,
            raw_response=response.text,
            model=response.model,
            usage=response.usage,
        )

    def run_batch(
        self,
        instances: List[QueryInstance],
    ) -> List[StandardRAGOutput]:
        """Run the pipeline over a list of instances, returning results in
        the same order."""
        results: List[StandardRAGOutput] = []
        for inst in instances:
            logger.info("Standard RAG – processing %s", inst.id)
            results.append(self.answer(inst))
        return results
