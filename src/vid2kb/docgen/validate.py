from __future__ import annotations

from vid2kb.docgen.planner import DocSpec
from vid2kb.schemas import KnowledgeDocument

ALLOWED_DOC_TYPES = {'tutorial', 'summary', 'meeting_notes', 'notes', 'qa'}

MAX_BODY_CHARS = 2000
MAX_SECTIONS = 12
TIMESTAMP_TOLERANCE = 2.0


def validate_document(
    doc: KnowledgeDocument,
    spec: DocSpec,
    transcript: str,
    max_timestamps: set[float],
) -> list[str]:
    problems: list[str] = []

    if not doc.title:
        problems.append('title is empty')

    if doc.doc_type not in ALLOWED_DOC_TYPES:
        problems.append(f'invalid doc_type: {doc.doc_type}')

    headings = {section.heading for section in doc.sections}
    for heading in spec.outline:
        if heading not in headings:
            problems.append(f'missing outline section: {heading}')

    for section in doc.sections:
        for ts in section.source_timestamps:
            if not any(abs(ts - value) <= TIMESTAMP_TOLERANCE for value in max_timestamps):
                problems.append(f'source_timestamp {ts} not near any known timestamp')

        if len(section.body_md) > MAX_BODY_CHARS:
            problems.append(f'body_md too long in section: {section.heading}')

    if len(doc.sections) > MAX_SECTIONS:
        problems.append(f'too many sections: {len(doc.sections)}')

    if not doc.summary:
        problems.append('summary is empty')

    return problems
