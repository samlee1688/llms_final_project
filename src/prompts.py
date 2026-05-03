"""
Prompt templates for every stage of the three RAG systems.

Each template is a plain Python string with ``{placeholders}`` that are
filled at call-time via ``str.format`` / ``str.format_map``.
"""

# ======================================================================
# Standard RAG
# ======================================================================

STANDARD_RAG_SYSTEM = (
    "You are a knowledgeable assistant. Answer the question based on the "
    "provided documents. If the documents contain conflicting information, "
    "use your best judgment to determine the most well-supported answer. "
    "Provide a concise answer followed by a brief explanation."
)

STANDARD_RAG_USER = """\
Documents:
{formatted_documents}

Question: {question}

Provide your answer based on the documents above. First state your answer \
clearly, then briefly explain your reasoning."""

# ======================================================================
# MADAM-RAG  –  Document Agent
# ======================================================================

MADAM_AGENT_SYSTEM = (
    "You are an expert analyst assigned to a specific document. Your role "
    "is to carefully analyse the document and argue for the answer it "
    "supports. Use direct evidence from the document to support your "
    "position. Be concise and avoid repetition."
)

MADAM_AGENT_INITIAL_USER = """\
You have been assigned the following document:

[Document — {doc_id}]
Source: {source}
{content}

Question: {question}

Based on this document, provide your answer to the question. Support your \
answer with specific evidence from the document.

Keep the response under 150 words using exactly:
Claim: <one sentence>
Evidence: <1-2 short citations or paraphrases>"""

# ======================================================================
# MADAM-RAG  –  Debate rounds
# ======================================================================

MADAM_DEBATE_SYSTEM = (
    "You are an expert analyst participating in a structured debate. You "
    "have been assigned a specific document and have previously argued for "
    "an answer. Other analysts have presented their positions based on "
    "different documents. Consider their arguments carefully, but continue "
    "to defend your position if your document's evidence is strong. You "
    "may update your answer only if the opposing evidence is clearly more "
    "compelling. Keep output concise and evidence-centered."
)

MADAM_DEBATE_USER = """\
Your assigned document:

[Document — {doc_id}]
Source: {source}
{content}

Your previous argument:
{previous_argument}

Arguments from other analysts:
{other_arguments}

Question: {question}

Provide your updated position. If you change your answer, explain why the \
other evidence was more compelling. If you maintain your answer, explain \
why your document's evidence is stronger.

Keep the response under 150 words using exactly:
Claim: <one sentence>
Update: <changed|unchanged + short reason>
Evidence: <1-2 short citations or paraphrases>"""

# ======================================================================
# MADAM-RAG  –  Aggregator / Judge
# ======================================================================

MADAM_AGGREGATOR_SYSTEM = (
    "You are an impartial judge. Multiple expert analysts have debated a "
    "question, each drawing on a different source document. Your task is "
    "to review their final arguments and determine the best-supported "
    "answer. Base your decision on the strength of the evidence presented, "
    "not on the style or confidence of the arguments. Keep your rationale brief."
)

MADAM_AGGREGATOR_USER = """\
Question: {question}

Below are the final arguments from each analyst after {num_rounds} rounds \
of debate:

{final_arguments}

Based on the arguments above, determine the best-supported answer. First \
state your answer clearly, then briefly explain which evidence was most \
convincing and why in at most 100 words."""


# ======================================================================
# MADAM-RAG  –  Credibility-Hint Aggregator
# ======================================================================

MADAM_AGGREGATOR_CREDIBILITY_HINT_SYSTEM = (
    "You are an impartial judge. Multiple expert analysts have debated a "
    "question, each drawing on a different source document. Each document "
    "was independently assigned a credibility tier (high, medium, or low) "
    "based on source reliability. Your task is to review their final arguments "
    "and determine the best-supported answer. When arguments conflict, prefer "
    "claims from higher-credibility documents unless lower-credibility documents "
    "present clearly stronger direct evidence. Base your decision on evidence "
    "strength and source reliability, not on argument style or confidence. "
    "Keep your rationale brief."
)

MADAM_AGGREGATOR_CREDIBILITY_HINT_USER = """\
Question: {question}

Document credibility tiers (assigned before the debate):
{credibility_summary}

Below are the final arguments from each analyst after {num_rounds} rounds \
of debate:

{final_arguments}

Determine the best-supported answer. When claims conflict, prefer the analyst \
whose document has the higher credibility tier unless a lower-credibility \
document provides clearly stronger direct evidence. First state your answer \
clearly, then briefly explain your reasoning in at most 100 words."""


# ======================================================================
# MADAM-RAG  –  RA-RAG (Reliability-Aware) Aggregator
# ======================================================================

MADAM_RA_RAG_SYSTEM = (
    "You are an evidence reliability assessor and impartial judge. "
    "Multiple expert analysts have debated a question, each drawing on a "
    "different source document. Before deciding on the best answer, assess "
    "the cross-document consistency of each source: identify which answers "
    "are corroborated across multiple independent documents and which are "
    "contradicted. Weight your confidence by source reliability and "
    "corroboration strength. Base your decision on evidence, not argument style."
)

MADAM_RA_RAG_USER = """\
Question: {question}

Evidence summary (document excerpts with assigned credibility tiers):
{evidence_summary}

Below are the final arguments from each analyst after {num_rounds} rounds \
of debate:

{final_arguments}

Step 1: Assess cross-document consistency. Which claims are corroborated by \
multiple independent sources? Which are contradicted or isolated?
Step 2: Weight each claim by its corroboration strength and the credibility \
of its supporting sources.
Step 3: State the best-supported answer clearly, then briefly explain which \
evidence was most reliable and consistent. Keep your explanation under 100 words."""


# ======================================================================
# MADAM-RAG  –  Chain-of-Thought Judge Aggregator
# ======================================================================

MADAM_COT_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating competing expert analyses. "
    "Before deciding on the best answer, reason step-by-step about evidence "
    "quality: identify what each document directly states, assess how strongly "
    "the document supports each analyst's claim, and determine which answer "
    "is best supported by direct evidence. Ignore argument style, confidence, "
    "or rhetorical persuasiveness. Only then state your final conclusion."
)

MADAM_COT_JUDGE_USER = """\
Question: {question}

Below are the final arguments from each analyst after {num_rounds} rounds \
of debate:

{final_arguments}

Reason step-by-step:
1. What direct evidence does each analyst cite from their document?
2. How strongly does that evidence support their claim (strong, moderate, or weak)?
3. Are there factual contradictions between the analysts?
4. Which analyst has the most directly evidenced answer?

After your step-by-step reasoning, state your final answer clearly and \
briefly explain your conclusion in at most 80 words."""


# ======================================================================
# EAJP  –  Evidence-Anchored Judge Prompting
# ======================================================================

EAJP_SCORECARD_SYSTEM = (
    "You are an evidence auditor. Your task is to evaluate source evidence "
    "before reading any persuasive arguments. You must identify the key "
    "claim in each document, locate the supporting evidence, assess source "
    "credibility and cross-document consistency, and avoid using rhetorical "
    "style as a signal."
)

EAJP_SCORECARD_USER = """\
Question: {question}

You will review each analyst's document and claim. For each analyst:
1. Record the stated credibility tier exactly as provided.
2. Identify the most relevant claim supported by the document.
3. Quote 1-2 short spans from the document.
4. For each span, label support for the claim as one of:
    ["supports", "weak", "contradicts"].
5. Provide a document-level support assessment:
    ["strong_support", "mixed", "weak_support", "contradicted"].
6. State whether the claim is consistent with the other documents:
    ["consistent", "partially_consistent", "inconsistent"].

Input bundle:
{evidence_bundle}

Return strict JSON with this schema:
{{
    "scorecard": [
        {{
            "agent_id": "...",
            "document_id": "...",
            "credibility_tier": "high|medium|low|unknown",
            "agent_claim": "...",
            "span_checks": [
                        {{"span": "...", "label": "supports|weak|contradicts"}}
            ],
            "document_support_assessment": "strong_support|mixed|weak_support|contradicted",
                    "cross_document_consistency": "consistent|partially_consistent|inconsistent",
            "notes": "..."
        }}
    ],
    "preliminary_winner_id": "agent_id",
    "preliminary_verdict": "A short evidence-only verdict."
}}
"""

EAJP_FINAL_SYSTEM = (
    "You are an impartial judge. You are given an evidence scorecard and "
    "full analyst arguments. Decide which analyst has the best-supported "
    "answer. Ignore confidence and rhetoric. Favor verifiable evidence. If "
    "the final verdict conflicts with the evidence scorecard, the scorecard "
    "verdict must be privileged. Keep justification concise."
)

EAJP_FINAL_USER = """\
Question: {question}

Evidence scorecard (completed before reading arguments):
{scorecard_json}

Analyst final arguments:
{final_arguments}

If your final conclusion differs from the evidence scorecard, explain the
conflict briefly and still provide the evidence-first answer.

Return strict JSON:
{{
    "final_winner_id": "agent_id",
    "final_answer": "The winning claim as a concise answer.",
    "final_verdict": "Brief rationale grounded in evidence."
}}
"""


# ======================================================================
# Tone-Controlled Benchmark construction
# ======================================================================

TCB_REWRITE_SYSTEM = (
    "You are a controlled rewriting engine for factual documents. "
    "Your job is to preserve meaning exactly while changing only rhetorical tone. "
    "You may adjust style, hedging strength, sentence structure, and register, but "
    "you must not add, remove, reorder, or alter factual claims, entities, "
    "relationships, chronology, or numeric values. "
    "Preserve all named entities, quantities, dates, percentages, measurements, "
    "and explicit comparisons. "
    "Do not introduce new evidence, conclusions, or speculation. "
    "Before finalizing, perform an internal lexical sanity check to verify: "
    "(1) all factual claims in the source remain present, "
    "(2) all numeric and named details are unchanged, "
    "(3) no source content is omitted, "
    "(4) only tone or rhetorical framing changed. "
    "If exact preservation cannot be guaranteed, return a minimally edited rewrite "
    "that keeps all source facts intact."
)

TCB_REWRITE_USER = """\
Original document:
{content}

Target tone: {tone}

Rewrite instructions:
1. Preserve factual meaning exactly.
2. Keep all entities, numbers, dates, and factual relations unchanged.
3. Do not add new claims, remove claims, or merge or split claims in ways
   that lose information.
4. Change only tone, register, and rhetorical framing.
5. Keep output close in informational content and coverage to the original.

Self-check before returning:
- All factual claims from the original are still present.
- No numeric value, named entity, date, or unit was changed.
- No claim was dropped or newly introduced.
- Tone now matches target tone: {tone}.

Return strict JSON with:
{{
    "rewritten_content": "...",
    "meaning_preserved": true,
    "tone": "{tone}",
    "preservation_checks": {{
        "all_claims_retained": true,
        "numeric_data_unchanged": true,
        "entities_unchanged": true,
        "no_new_claims_added": true,
        "no_claims_removed": true
    }}
}}
"""


# ======================================================================
# Helpers
# ======================================================================

def format_documents(documents, include_source: bool = True) -> str:
    """Format a list of ``Document`` objects into a numbered block suitable
    for insertion into a prompt."""
    parts = []
    for i, doc in enumerate(documents, 1):
        header = f"[Document {i}]"
        if include_source and doc.source:
            header += f" (Source: {doc.source})"
        parts.append(f"{header}\n{doc.content}")
    return "\n\n".join(parts)


def format_other_arguments(
    agents_responses: dict[str, str], exclude_agent: str
) -> str:
    """Format other agents' responses for a debate round, excluding the
    current agent."""
    parts = []
    for agent_id, response in agents_responses.items():
        if agent_id != exclude_agent:
            parts.append(f"[Analyst {agent_id}]:\n{response}")
    return "\n\n".join(parts) if parts else "(no other arguments yet)"
