from __future__ import annotations

from vid2kb.kb.store import query_documents


def query_knowledge(question: str, top_k: int = 5) -> list[dict]:
    """Return the most relevant document chunks for ``question``.

    This is retrieval-only for now; RAG-grounded chat over the knowledge base
    is wired up in a later phase.
    """
    return query_documents(question, top_k=top_k)
