from __future__ import annotations

import pytest
from pydantic import ValidationError

from vid2kb.schemas import (
    DocumentSection,
    FrameDescription,
    KnowledgeDocument,
    VisualTimeline,
)


def test_valid_knowledge_document_parses():
    doc = KnowledgeDocument.model_validate({
        'title': 'Test Doc',
        'doc_type': 'tutorial',
        'audience': 'developers',
        'summary': 'A summary.',
        'sections': [
            {'heading': 'Intro', 'body_md': 'hello', 'source_timestamps': [0.0, 1.5]},
        ],
        'key_points': ['point one', 'point two'],
        'glossary': {'term': 'definition'},
        'warnings': [],
    })
    assert doc.title == 'Test Doc'
    assert doc.doc_type == 'tutorial'
    assert doc.sections[0].heading == 'Intro'
    assert doc.sections[0].source_timestamps == [0.0, 1.5]
    assert doc.key_points == ['point one', 'point two']


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate({
            'doc_type': 'tutorial',
            'audience': 'developers',
            'summary': 'A summary.',
            'sections': [],
            'key_points': [],
        })


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        FrameDescription(
            index=0,
            timestamp_seconds=0.0,
            description='x',
            confidence=1.5,
        )


def test_visual_timeline_roundtrip():
    tl = VisualTimeline(
        summary='s',
        frames=[
            FrameDescription(
                index=0,
                timestamp_seconds=0.0,
                description='a',
                confidence=0.5,
            ),
            FrameDescription(
                index=1,
                timestamp_seconds=2.0,
                description='b',
                visible_text=['hi'],
                actions=['run'],
                confidence=0.9,
            ),
        ],
        warnings=['w'],
    )
    roundtripped = VisualTimeline.model_validate_json(tl.model_dump_json())
    assert roundtripped == tl
