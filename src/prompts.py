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
    "supports. Be thorough and use direct evidence from the document to "
    "support your position."
)

MADAM_AGENT_INITIAL_USER = """\
You have been assigned the following document:

[Document — {doc_id}]
Source: {source}
{content}

Question: {question}

Based on this document, provide your answer to the question. Support your \
answer with specific evidence from the document."""

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
    "compelling."
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
why your document's evidence is stronger."""

# ======================================================================
# MADAM-RAG  –  Aggregator / Judge
# ======================================================================

MADAM_AGGREGATOR_SYSTEM = (
    "You are an impartial judge. Multiple expert analysts have debated a "
    "question, each drawing on a different source document. Your task is "
    "to review their final arguments and determine the best-supported "
    "answer. Base your decision on the strength of the evidence presented, "
    "not on the style or confidence of the arguments."
)

MADAM_AGGREGATOR_USER = """\
Question: {question}

Below are the final arguments from each analyst after {num_rounds} rounds \
of debate:

{final_arguments}

Based on the arguments above, determine the best-supported answer. First \
state your answer clearly, then briefly explain which evidence was most \
convincing and why."""


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
