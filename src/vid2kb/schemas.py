from __future__ import annotations
from pydantic import BaseModel, Field

class FrameDescription(BaseModel):
    index: int = Field(description='frame index from sampling')
    timestamp_seconds: float
    description: str
    visible_text: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

class VisualTimeline(BaseModel):
    summary: str
    frames: list[FrameDescription]
    warnings: list[str] = Field(default_factory=list)

class DocumentSection(BaseModel):
    heading: str
    body_md: str
    source_timestamps: list[float] = Field(default_factory=list, description='video seconds this content came from')

class KnowledgeDocument(BaseModel):
    title: str
    doc_type: str = Field(description='tutorial|summary|meeting_notes|notes|qa')
    audience: str
    summary: str
    sections: list[DocumentSection]
    key_points: list[str]
    glossary: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
