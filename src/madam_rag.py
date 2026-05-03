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

from concurrent.futures import ThreadPoolExecutor
import logging
import random
from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import config
from src.data_loader import Document, QueryInstance
from src.models import LLMClient, LLMResponse
from src.prompts import (
    EAJP_FINAL_SYSTEM,
    EAJP_FINAL_USER,
    EAJP_SCORECARD_SYSTEM,
    EAJP_SCORECARD_USER,
    MADAM_AGENT_SYSTEM,
    MADAM_AGENT_INITIAL_USER,
    MADAM_DEBATE_SYSTEM,
    MADAM_DEBATE_USER,
    MADAM_AGGREGATOR_SYSTEM,
    MADAM_AGGREGATOR_USER,
    MADAM_AGGREGATOR_CREDIBILITY_HINT_SYSTEM,
    MADAM_AGGREGATOR_CREDIBILITY_HINT_USER,
    MADAM_RA_RAG_SYSTEM,
    MADAM_RA_RAG_USER,
    MADAM_COT_JUDGE_SYSTEM,
    MADAM_COT_JUDGE_USER,
    format_other_arguments,
)

logger = logging.getLogger(__name__)


def _norm_claim(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."


def _sum_usage(rows: List[dict]) -> dict:
    merged: dict[str, float] = {}
    for row in rows:
        for k, v in (row or {}).items():
            if isinstance(v, (int, float)):
                merged[k] = float(merged.get(k, 0.0)) + float(v)
    return merged


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


@dataclass
class EvidenceScoreEntry:
    """Structured evidence score for one agent-document pair."""

    agent_id: str
    document_id: str
    credibility_tier: str
    agent_claim: str
    span_checks: List[Dict[str, str]] = field(default_factory=list)
    document_support_assessment: str = "mixed"
    cross_document_consistency: str = "unknown"
    notes: str = ""


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
        include_full_document: bool,
        compact_doc_chars: int,
    ) -> AgentResponse:
        """Produce an updated argument after seeing the other agents."""
        formatted_others = format_other_arguments(other_arguments, self.agent_id)
        if include_full_document:
            doc_context = self.document.content
        else:
            # After round 0, compact to an excerpt to avoid repeatedly paying
            # for full-document prompt tokens every debate turn.
            doc_context = _clip(self.document.content, compact_doc_chars)
        user_prompt = MADAM_DEBATE_USER.format(
            doc_id=self.agent_id,
            source=self.document.source or "Unknown",
            content=doc_context,
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
        documents: Optional[List[Document]] = None,
        agent_claims: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        # Shuffle agent order to eliminate position bias before presenting to
        # the judge — matches the reference implementation (Wang et al. 2025).
        items = list(final_arguments.items())
        random.shuffle(items)
        formatted = "\n\n".join(
            f"[Analyst {aid}]:\n{arg}" for aid, arg in items
        )
        user_prompt = MADAM_AGGREGATOR_USER.format(
            question=question,
            num_rounds=num_rounds,
            final_arguments=formatted,
        )
        return self.llm.generate(MADAM_AGGREGATOR_SYSTEM, user_prompt)


class CredibilityHintAggregator(Aggregator):
    """MADAM aggregator that injects per-document credibility tiers into the
    judge prompt before arguments are read.

    Isolates the contribution of a simple credibility signal without any
    structured scorecard or privilege override — a lightweight ablation
    baseline between plain MADAM-RAG and full EAJP.
    """

    def aggregate(
        self,
        question: str,
        final_arguments: Dict[str, str],
        num_rounds: int,
        documents: Optional[List[Document]] = None,
        agent_claims: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        items = list(final_arguments.items())
        random.shuffle(items)
        formatted = "\n\n".join(
            f"[Analyst {aid}]:\n{arg}" for aid, arg in items
        )
        doc_map = {d.id: d for d in (documents or [])}
        cred_lines = [
            f"- Analyst {aid}: {(doc_map[aid].credibility if aid in doc_map and doc_map[aid].credibility else 'unknown')}-credibility"
            for aid, _ in items
        ]
        credibility_summary = (
            "\n".join(cred_lines)
            if cred_lines
            else "(credibility information not available)"
        )
        user_prompt = MADAM_AGGREGATOR_CREDIBILITY_HINT_USER.format(
            question=question,
            credibility_summary=credibility_summary,
            num_rounds=num_rounds,
            final_arguments=formatted,
        )
        return self.llm.generate(MADAM_AGGREGATOR_CREDIBILITY_HINT_SYSTEM, user_prompt)


class ReliabilityAwareAggregator(Aggregator):
    """MADAM aggregator that estimates source reliability via cross-document
    consistency before aggregating — a judge-only reimplementation of the
    RA-RAG approach (Tang et al.).

    Presents the judge with per-document excerpts and credibility tiers
    alongside the debate arguments, then asks for an explicit consistency
    assessment before committing to a final answer.
    """

    def __init__(self, llm: LLMClient, doc_excerpt_chars: int = 500):
        super().__init__(llm)
        self.doc_excerpt_chars = max(100, doc_excerpt_chars)

    def aggregate(
        self,
        question: str,
        final_arguments: Dict[str, str],
        num_rounds: int,
        documents: Optional[List[Document]] = None,
        agent_claims: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        items = list(final_arguments.items())
        random.shuffle(items)
        doc_map = {d.id: d for d in (documents or [])}

        evidence_parts: List[str] = []
        for aid, arg in items:
            doc = doc_map.get(aid)
            excerpt = _clip(doc.content if doc else "", self.doc_excerpt_chars)
            cred = (doc.credibility if doc else None) or "unknown"
            evidence_parts.append(
                f"[Analyst {aid}] (credibility: {cred})\n"
                f"Document excerpt: {excerpt}\n"
                f"Analyst argument: {arg}"
            )

        evidence_summary = "\n\n".join(evidence_parts)
        formatted = "\n\n".join(f"[Analyst {aid}]:\n{arg}" for aid, arg in items)
        user_prompt = MADAM_RA_RAG_USER.format(
            question=question,
            num_rounds=num_rounds,
            evidence_summary=evidence_summary,
            final_arguments=formatted,
        )
        return self.llm.generate(MADAM_RA_RAG_SYSTEM, user_prompt)


class ChainOfThoughtAggregator(Aggregator):
    """MADAM aggregator that prompts the judge to reason step-by-step about
    evidence quality before committing to a final answer.

    Tests whether the benefit of EAJP's structured scorecard comes from
    explicit evidence reasoning vs. the scorecard format itself.
    """

    def aggregate(
        self,
        question: str,
        final_arguments: Dict[str, str],
        num_rounds: int,
        documents: Optional[List[Document]] = None,
        agent_claims: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        items = list(final_arguments.items())
        random.shuffle(items)
        formatted = "\n\n".join(
            f"[Analyst {aid}]:\n{arg}" for aid, arg in items
        )
        user_prompt = MADAM_COT_JUDGE_USER.format(
            question=question,
            num_rounds=num_rounds,
            final_arguments=formatted,
        )
        return self.llm.generate(MADAM_COT_JUDGE_SYSTEM, user_prompt)


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from model output."""
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


def _claim_from_text(text: str) -> str:
    """Extract a concise claim from an agent response."""
    if not text.strip():
        return ""
    first_line = text.strip().splitlines()[0].strip()
    return first_line[:1000]


class EvidenceAnchoredAggregator(Aggregator):
    """EAJP judge that scores evidence before reading debate arguments."""

    def __init__(
        self,
        llm: LLMClient,
        privilege_scorecard: bool = True,
        compact_doc_chars: int = 10000,
        short_circuit_on_confident_scorecard: bool = True,
        force_scorecard_only: bool = False,
    ):
        super().__init__(llm)
        self.privilege_scorecard = privilege_scorecard
        self.compact_doc_chars = max(200, compact_doc_chars)
        self.short_circuit_on_confident_scorecard = short_circuit_on_confident_scorecard
        self.force_scorecard_only = force_scorecard_only

    def _build_evidence_bundle(
        self,
        documents: List[Document],
        agent_claims: Dict[str, str],
    ) -> str:
        # Shuffle document order to prevent position bias in the preliminary
        # verdict — TCB instances always place the correct doc first, so a
        # fixed order would systematically favour it in the scorecard phase.
        doc_list = list(documents)
        random.shuffle(doc_list)
        parts: List[str] = []
        for i, doc in enumerate(doc_list):
            agent_id = doc.id or f"doc_{i + 1}"
            parts.append(
                "\n".join(
                    [
                        f"[Analyst {agent_id}]",
                        f"Document ID: {doc.id or agent_id}",
                        f"Source: {doc.source or 'Unknown'}",
                        f"Credibility tier: {doc.credibility or 'unknown'}",
                        f"Agent claim: {agent_claims.get(agent_id, '') or '(not provided)'}",
                        "Document excerpt:",
                        _clip(doc.content, self.compact_doc_chars),
                    ]
                )
            )
        return "\n\n".join(parts)

    def _confident_preliminary(self, entries: List[EvidenceScoreEntry], winner_id: str) -> bool:
        if not winner_id or not entries:
            return False
        lookup = {e.agent_id: e for e in entries}
        winner = lookup.get(winner_id)
        if winner is None:
            return False
        if winner.document_support_assessment not in {"strong_support", "mixed"}:
            return False
        weak_others = 0
        for e in entries:
            if e.agent_id == winner_id:
                continue
            if e.document_support_assessment in {"weak_support", "contradicted"}:
                weak_others += 1
        return weak_others >= max(1, len(entries) - 1)

    def aggregate(
        self,
        question: str,
        final_arguments: Dict[str, str],
        num_rounds: int,
        documents: Optional[List[Document]] = None,
        agent_claims: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        documents = documents or []
        claims = agent_claims or {
            aid: _claim_from_text(arg) for aid, arg in final_arguments.items()
        }

        scorecard_bundle = self._build_evidence_bundle(documents, claims)
        scorecard_prompt = EAJP_SCORECARD_USER.format(
            question=question,
            evidence_bundle=scorecard_bundle,
        )
        scorecard_resp = self.llm.generate(EAJP_SCORECARD_SYSTEM, scorecard_prompt)
        scorecard_data = _extract_json_block(scorecard_resp.text)

        scorecard_entries = scorecard_data.get("scorecard", [])
        typed_entries: List[EvidenceScoreEntry] = []
        for entry in scorecard_entries:
            typed_entries.append(
                EvidenceScoreEntry(
                    agent_id=str(entry.get("agent_id", "")),
                    document_id=str(entry.get("document_id", "")),
                    credibility_tier=str(entry.get("credibility_tier", "unknown")),
                    agent_claim=str(entry.get("agent_claim", "")),
                    span_checks=list(entry.get("span_checks", [])),
                    document_support_assessment=str(
                        entry.get("document_support_assessment", "mixed")
                    ),
                    cross_document_consistency=str(
                        entry.get("cross_document_consistency", "unknown")
                    ),
                    notes=str(entry.get("notes", "")),
                )
            )
        scorecard_lookup = {
            entry.agent_id: entry for entry in typed_entries
        }
        preliminary_winner = str(scorecard_data.get("preliminary_winner_id", "")).strip()
        preliminary_verdict = str(scorecard_data.get("preliminary_verdict", "")).strip()

        scorecard_struct = {
            "scorecard": [
                {
                    "agent_id": e.agent_id,
                    "document_id": e.document_id,
                    "credibility_tier": e.credibility_tier,
                    "agent_claim": e.agent_claim,
                    "span_checks": e.span_checks,
                    "document_support_assessment": e.document_support_assessment,
                    "cross_document_consistency": e.cross_document_consistency,
                    "notes": e.notes,
                }
                for e in typed_entries
            ],
            "preliminary_winner_id": preliminary_winner,
            "preliminary_verdict": preliminary_verdict,
        }

        if self.force_scorecard_only or (
            self.short_circuit_on_confident_scorecard
            and self._confident_preliminary(typed_entries, preliminary_winner)
        ):
            winner_entry = scorecard_lookup.get(preliminary_winner)
            chosen_claim = (
                claims.get(preliminary_winner)
                or (winner_entry.agent_claim if winner_entry is not None else "")
                or _claim_from_text(final_arguments.get(preliminary_winner, ""))
            )
            return LLMResponse(
                text=f"{chosen_claim}\n\n{preliminary_verdict}".strip(),
                model=scorecard_resp.model,
                usage=dict(scorecard_resp.usage or {}),
                metadata={
                    "judge_mode": "eajp",
                    "scorecard": scorecard_struct,
                    "short_circuit_used": not self.force_scorecard_only,
                    "scorecard_only_used": self.force_scorecard_only,
                    "scorecard_privilege_applied": False,
                    "preliminary_winner_id": preliminary_winner,
                    "final_winner_id_before_privilege": preliminary_winner,
                    "final_winner_id": preliminary_winner,
                    "final_verdict": preliminary_verdict,
                },
            )

        scorecard_json = json.dumps(scorecard_struct, indent=2)
        # Shuffle agent order before final judge to eliminate position bias —
        # consistent with the base Aggregator and the MADAM-RAG reference impl.
        final_items = list(final_arguments.items())
        random.shuffle(final_items)
        final_prompt = EAJP_FINAL_USER.format(
            question=question,
            scorecard_json=scorecard_json,
            final_arguments="\n\n".join(
                f"[Analyst {aid}]:\n{arg}" for aid, arg in final_items
            ),
        )
        final_resp = self.llm.generate(EAJP_FINAL_SYSTEM, final_prompt)
        final_data = _extract_json_block(final_resp.text)
        final_winner = str(final_data.get("final_winner_id", "")).strip()
        final_winner_before_privilege = final_winner
        final_answer = str(final_data.get("final_answer", "")).strip()
        final_verdict = str(final_data.get("final_verdict", "")).strip()

        privilege_applied = False
        if self.privilege_scorecard and preliminary_winner:
            if final_winner and final_winner != preliminary_winner:
                privilege_applied = True
                final_winner = preliminary_winner
            elif not final_winner:
                privilege_applied = True
                final_winner = preliminary_winner

        if not final_winner and final_arguments:
            final_winner = next(iter(final_arguments.keys()))

        chosen_claim = (
            claims.get(final_winner)
            or ((scorecard_lookup.get(final_winner).agent_claim) if scorecard_lookup.get(final_winner) else "")
            or final_answer
            or _claim_from_text(final_arguments.get(final_winner, ""))
        )

        reasoning = final_verdict or str(scorecard_data.get("preliminary_verdict", ""))
        if self.privilege_scorecard and preliminary_winner and final_data.get("final_winner_id"):
            if str(final_data.get("final_winner_id", "")).strip() != preliminary_winner:
                reasoning = (
                    reasoning
                    + "\n\nEAJP rule: final verdict was checked against the scorecard and "
                    + "the scorecard verdict was privileged when they conflicted."
                )

        merged_usage = {}
        merged_usage.update(scorecard_resp.usage or {})
        for key, value in (final_resp.usage or {}).items():
            merged_usage[f"final_{key}"] = value

        return LLMResponse(
            text=f"{chosen_claim}\n\n{reasoning}".strip(),
            model=final_resp.model,
            usage=merged_usage,
            metadata={
                "judge_mode": "eajp",
                "scorecard": scorecard_struct,
                "short_circuit_used": False,
                "scorecard_only_used": False,
                "scorecard_privilege_applied": privilege_applied,
                "preliminary_winner_id": preliminary_winner,
                "final_winner_id_before_privilege": final_winner_before_privilege,
                "final_winner_id": final_winner,
                "final_verdict": final_verdict,
            },
        )


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
        judge: Optional[Aggregator] = None,
        max_parallel_agents: Optional[int] = None,
        compact_doc_chars: int = 10000,
        argument_summary_chars: int = 1000,
        early_stop_patience: int = 1,
        include_full_document_in_debate: bool = False,
    ):
        self.agent_llm = agent_llm
        self.judge_llm = judge_llm or agent_llm
        self.num_rounds = num_rounds if num_rounds is not None else config.DEBATE_ROUNDS
        self.judge = judge or Aggregator(self.judge_llm)
        if max_parallel_agents is None:
            max_parallel_agents = int(os.getenv("MADAM_MAX_AGENT_PARALLEL", "4"))
        self.max_parallel_agents = max(1, max_parallel_agents)
        self.compact_doc_chars = max(300, compact_doc_chars)
        self.argument_summary_chars = max(80, argument_summary_chars)
        self.early_stop_patience = max(0, early_stop_patience)
        self.include_full_document_in_debate = include_full_document_in_debate

    def _summarize_arguments(self, responses: Dict[str, str]) -> Dict[str, str]:
        # Pass the full response (claim + update + evidence) to opponent agents,
        # not just the first-line claim. Matches the reference code which forwards
        # complete peer responses so agents can engage with the evidence, not just
        # the headline assertion.
        return {
            aid: _clip(text, self.argument_summary_chars)
            for aid, text in responses.items()
        }

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
        t0 = time.perf_counter()
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
        if len(agents) <= 1 or self.max_parallel_agents <= 1:
            for agent in agents:
                resp = agent.generate_initial(question)
                round_0.append(resp)
                logger.debug(
                    "  Agent %s initial answer (first 120 chars): %s",
                    agent.agent_id, resp.text[:120],
                )
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_parallel_agents, len(agents))) as pool:
                responses = list(pool.map(lambda a: a.generate_initial(question), agents))
            round_0.extend(responses)
        debate_history.rounds.append(round_0)

        # 3. Debate rounds
        stable_rounds = 0
        prev_agent_answers: Dict[str, str] = {}
        early_stop_triggered = False
        for r in range(1, self.num_rounds + 1):
            logger.info("MADAM-RAG [%s] – debate round %d/%d", instance.id, r, self.num_rounds)
            # Collect latest responses from all agents
            current_responses: Dict[str, str] = {
                a.agent_id: a.latest_response.text for a in agents
            }
            summarized_responses = self._summarize_arguments(current_responses)
            round_r: List[AgentResponse] = []
            if len(agents) <= 1 or self.max_parallel_agents <= 1:
                for agent in agents:
                    resp = agent.generate_debate(
                        question,
                        r,
                        summarized_responses,
                        include_full_document=self.include_full_document_in_debate,
                        compact_doc_chars=self.compact_doc_chars,
                    )
                    round_r.append(resp)
                    logger.debug(
                        "  Agent %s round-%d (first 120 chars): %s",
                        agent.agent_id, r, resp.text[:120],
                    )
            else:
                with ThreadPoolExecutor(max_workers=min(self.max_parallel_agents, len(agents))) as pool:
                    responses = list(
                        pool.map(
                            lambda a: a.generate_debate(
                                question,
                                r,
                                summarized_responses,
                                include_full_document=self.include_full_document_in_debate,
                                compact_doc_chars=self.compact_doc_chars,
                            ),
                            agents,
                        )
                    )
                round_r.extend(responses)
            debate_history.rounds.append(round_r)

            # Early stop: each agent's answer stable (unchanged) from previous
            # round — mirrors the reference code which stops when no agent
            # changes its answer, regardless of whether agents agree with each other.
            current_agent_answers = {
                a.agent_id: _norm_claim(_claim_from_text(a.latest_response.text))
                for a in agents
            }
            if prev_agent_answers and current_agent_answers == prev_agent_answers:
                stable_rounds += 1
            else:
                stable_rounds = 0
            prev_agent_answers = current_agent_answers
            if stable_rounds >= self.early_stop_patience:
                early_stop_triggered = True
                logger.info("MADAM-RAG [%s] – early stop at round %d (all agents stable)", instance.id, r)
                break

        # 4. Aggregation
        logger.info("MADAM-RAG [%s] – aggregating", instance.id)
        final_args: Dict[str, str] = {
            a.agent_id: a.latest_response.text for a in agents
        }
        agent_claims: Dict[str, str] = {
            a.agent_id: _claim_from_text(a.responses[0].text if a.responses else "")
            for a in agents
        }
        judge_resp = self.judge.aggregate(
            question,
            final_args,
            max(0, len(debate_history.rounds) - 1),
            documents=docs,
            agent_claims=agent_claims,
        )

        all_agent_usage = [resp.usage for rnd in debate_history.rounds for resp in rnd]
        total_usage = _sum_usage(all_agent_usage + [judge_resp.usage or {}])
        elapsed = time.perf_counter() - t0
        rounds_executed = max(0, len(debate_history.rounds) - 1)

        return MadamRAGOutput(
            query_id=instance.id,
            final_answer=judge_resp.text,
            judge_reasoning=judge_resp.text,
            debate_history=debate_history,
            model=judge_resp.model,
            usage=total_usage,
            metadata={
                "runtime_seconds": round(elapsed, 4),
                "rounds_configured": self.num_rounds,
                "rounds_executed": rounds_executed,
                "early_stop_triggered": early_stop_triggered,
                "num_agents": len(agents),
                "agent_calls": len(all_agent_usage),
                "judge_calls": 1,
                "parallel_agent_workers": min(self.max_parallel_agents, max(1, len(agents))),
                "judge_metadata": dict(getattr(judge_resp, "metadata", {}) or {}),
            },
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
