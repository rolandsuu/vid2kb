from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class AgentState(TypedDict, total=False):
    run_id: str
    user_prompt: str
    source: str
    video_path: Optional[str]
    transcript: Optional[str]
    transcript_language: Optional[str]
    timeline: Optional[dict]
    doc_spec: Optional[dict]
    document: Optional[dict]
    markdown: Optional[str]
    pdf_path: Optional[str]
    kb_doc_id: Optional[str]
    kb_node_count: Optional[int]
    errors: list[str]
    steps: Annotated[list[str], add_messages]
    next: Optional[str]
    iterations: int
    final_report: Optional[dict]
