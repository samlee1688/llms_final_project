"""Evidence-Anchored Judge Prompting (EAJP) wrapper.

EAJP is implemented as MADAM-RAG with an evidence-anchored judge that
forces a structured scorecard before reading the full debate arguments.
"""

from __future__ import annotations

from typing import Optional

from src.madam_rag import EvidenceAnchoredAggregator, MadamRAG
from src.models import LLMClient


class EAJPRAG(MadamRAG):
    """MADAM-RAG configured with the EAJP judge."""

    def __init__(
        self,
        agent_llm: LLMClient,
        judge_llm: Optional[LLMClient] = None,
        num_rounds: Optional[int] = None,
        privilege_scorecard: bool = True,
    ):
        judge_llm = judge_llm or agent_llm
        super().__init__(
            agent_llm=agent_llm,
            judge_llm=judge_llm,
            num_rounds=num_rounds,
            judge=EvidenceAnchoredAggregator(judge_llm, privilege_scorecard),
        )


__all__ = ["EAJPRAG"]
