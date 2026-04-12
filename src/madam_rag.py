"""
MADAM-RAG: Multi-Agent Debate-Augmented Retrieval-Augmented Generation.

Implements the debate framework described by Wang et al.:

1. **Agent initialisation** – Each retrieved document is assigned a
   dedicated *document agent* backed by an LLM.
2. **Initial response** – Every agent independently produces an answer
   grounded in its assigned document.
3. **Debate rounds** – For *R* rounds each agent sees the other agents'
   most recent arguments and produces an updated response.  Agents are
   encouraged to maintain their position unless opposing evidence is
   clearly stronger.
4. **Aggregation** – A separate *judge* agent reviews every agent's final
   argument and selects the best-supported answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from src.data_loader import Document, QueryInstance
from src.models import LLMClient, LLMResponse
from src.prompts import (
    MADAM_AGENT_SYSTEM,
    MADAM_AGENT_INITIAL_USER,
    MADAM_DEBATE_SYSTEM,
    MADAM_DEBATE_USER,
    MADAM_AGGREGATOR_SYSTEM,
    MADAM_AGGREGATOR_USER,
    format_other_arguments,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data classes for intermediate / final outputs
# -----------------------------------------------------------------------

@dataclass
class AgentResponse:
    """A single response from a document agent (one round)."""

    agent_id: str
    round_num: int          # 0 = initial, 1..R = debate rounds
    text: str
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class DebateHistory:
    """Full debate transcript for one query."""

    query_id: str
    rounds: List[List[AgentResponse]] = field(default_factory=list)
    # rounds[r] contains one AgentResponse per agent for round r


@dataclass
class MadamRAGOutput:
    """Result of running a single query through MADAM-RAG."""

    query_id: str
    final_answer: str
    judge_reasoning: str
    debate_history: DebateHistory
    model: str = ""
    usage: dict = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------
# Document agent
# -----------------------------------------------------------------------

class DocumentAgent:
    """LLM-backed agent that argues for its assigned document's answer."""

    def __init__(self, agent_id: str, document: Document, llm: LLMClient):
        self.agent_id = agent_id
        self.document = document
        self.llm = llm
        self.responses: List[AgentResponse] = []

    @property
    def latest_response(self) -> Optional[AgentResponse]:
        return self.responses[-1] if self.responses else None

    # ----- initial round ------------------------------------------------

    def generate_initial(self, question: str) -> AgentResponse:
        """Produce the agent's first answer using only its own document."""
        user_prompt = MADAM_AGENT_INITIAL_USER.format(
            doc_id=self.agent_id,
            source=self.document.source or "Unknown",
            content=self.document.content,
            question=question,
        )
        resp: LLMResponse = self.llm.generate(MADAM_AGENT_SYSTEM, user_prompt)
        ar = AgentResponse(
            agent_id=self.agent_id,
            round_num=0,
            text=resp.text,
            model=resp.model,
            usage=resp.usage,
        )
        self.responses.append(ar)
        return ar

    # ----- debate round -------------------------------------------------

    def generate_debate(
        self,
        question: str,
        round_num: int,
        other_arguments: Dict[str, str],
    ) -> AgentResponse:
        """Produce an updated argument after seeing the other agents."""
        formatted_others = format_other_arguments(other_arguments, self.agent_id)
        user_prompt = MADAM_DEBATE_USER.format(
            doc_id=self.agent_id,
            source=self.document.source or "Unknown",
            content=self.document.content,
            previous_argument=self.latest_response.text if self.latest_response else "",
            other_arguments=formatted_others,
            question=question,
        )
        resp: LLMResponse = self.llm.generate(MADAM_DEBATE_SYSTEM, user_prompt)
        ar = AgentResponse(
            agent_id=self.agent_id,
            round_num=round_num,
            text=resp.text,
            model=resp.model,
            usage=resp.usage,
        )
        self.responses.append(ar)
        return ar


# -----------------------------------------------------------------------
# Aggregator (judge)
# -----------------------------------------------------------------------

class Aggregator:
    """Judge that reviews all agents' final arguments and decides on the
    best-supported answer."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def aggregate(
        self,
        question: str,
        final_arguments: Dict[str, str],
        num_rounds: int,
    ) -> LLMResponse:
        formatted = "\n\n".join(
            f"[Analyst {aid}]:\n{arg}" for aid, arg in final_arguments.items()
        )
        user_prompt = MADAM_AGGREGATOR_USER.format(
            question=question,
            num_rounds=num_rounds,
            final_arguments=formatted,
        )
        return self.llm.generate(MADAM_AGGREGATOR_SYSTEM, user_prompt)


# -----------------------------------------------------------------------
# Full MADAM-RAG pipeline
# -----------------------------------------------------------------------

class MadamRAG:
    """Orchestrates the full MADAM-RAG debate pipeline.

    Parameters
    ----------
    agent_llm:
        LLM client used for document agents (may be a cheaper / faster
        model).
    judge_llm:
        LLM client used for the final aggregator.  If *None*, defaults
        to ``agent_llm``.
    num_rounds:
        Number of debate rounds *after* the initial response round.
    """

    def __init__(
        self,
        agent_llm: LLMClient,
        judge_llm: Optional[LLMClient] = None,
        num_rounds: Optional[int] = None,
    ):
        self.agent_llm = agent_llm
        self.judge_llm = judge_llm or agent_llm
        self.num_rounds = num_rounds if num_rounds is not None else config.DEBATE_ROUNDS

    # ----- single instance ----------------------------------------------

    def answer(
        self,
        instance: QueryInstance,
        documents: Optional[List[Document]] = None,
    ) -> MadamRAGOutput:
        """Run the full debate pipeline for one query.

        Parameters
        ----------
        instance:
            The evaluation instance.
        documents:
            Optional override for the documents to use.
        """
        docs = documents if documents is not None else instance.documents
        question = instance.question

        # 1. Create document agents
        agents: List[DocumentAgent] = []
        for i, doc in enumerate(docs):
            agent_id = doc.id or f"doc_{i + 1}"
            agents.append(DocumentAgent(agent_id, doc, self.agent_llm))

        debate_history = DebateHistory(query_id=instance.id)

        # 2. Initial response round (round 0)
        logger.info(
            "MADAM-RAG [%s] – initial round with %d agents",
            instance.id, len(agents),
        )
        round_0: List[AgentResponse] = []
        for agent in agents:
            resp = agent.generate_initial(question)
            round_0.append(resp)
            logger.debug(
                "  Agent %s initial answer (first 120 chars): %s",
                agent.agent_id, resp.text[:120],
            )
        debate_history.rounds.append(round_0)

        # 3. Debate rounds
        for r in range(1, self.num_rounds + 1):
            logger.info("MADAM-RAG [%s] – debate round %d/%d", instance.id, r, self.num_rounds)
            # Collect latest responses from all agents
            current_responses: Dict[str, str] = {
                a.agent_id: a.latest_response.text for a in agents
            }
            round_r: List[AgentResponse] = []
            for agent in agents:
                resp = agent.generate_debate(question, r, current_responses)
                round_r.append(resp)
                logger.debug(
                    "  Agent %s round-%d (first 120 chars): %s",
                    agent.agent_id, r, resp.text[:120],
                )
            debate_history.rounds.append(round_r)

        # 4. Aggregation
        logger.info("MADAM-RAG [%s] – aggregating", instance.id)
        final_args: Dict[str, str] = {
            a.agent_id: a.latest_response.text for a in agents
        }
        aggregator = Aggregator(self.judge_llm)
        judge_resp = aggregator.aggregate(question, final_args, self.num_rounds)

        return MadamRAGOutput(
            query_id=instance.id,
            final_answer=judge_resp.text,
            judge_reasoning=judge_resp.text,
            debate_history=debate_history,
            model=judge_resp.model,
            usage=judge_resp.usage,
        )

    # ----- batch --------------------------------------------------------

    def run_batch(
        self,
        instances: List[QueryInstance],
    ) -> List[MadamRAGOutput]:
        """Run the pipeline over many instances sequentially."""
        results: List[MadamRAGOutput] = []
        for inst in instances:
            logger.info("MADAM-RAG – processing %s", inst.id)
            results.append(self.answer(inst))
        return results
