from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vid2kb.docgen.composer import compose_document
from vid2kb.docgen.planner import DocSpec, plan_document
from vid2kb.docgen.render import render_markdown, render_pdf
from vid2kb.docgen.validate import validate_document
from vid2kb.schemas import (
    FrameDescription,
    KnowledgeDocument,
    VisualTimeline,
)


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def __getattr__(self, name):
        return self

    def create(self, **kwargs):
        self.calls += 1
        return _resp(self._contents.pop(0))


DOC_JSON = {
    'title': 'Test Tutorial',
    'doc_type': 'tutorial',
    'audience': 'developers',
    'summary': 'A summary',
    'sections': [
        {'heading': 'Introduction', 'body_md': 'hello world', 'source_timestamps': [12.0]},
    ],
    'key_points': ['key point one'],
    'glossary': {'term': 'meaning'},
    'warnings': [],
}


def _spec() -> DocSpec:
    return DocSpec(
        doc_type='tutorial',
        title='Test Tutorial',
        audience='developers',
        outline=['Introduction'],
        focus_instructions='focus',
    )


def _timeline() -> VisualTimeline:
    return VisualTimeline(
        summary='tl summary',
        frames=[
            FrameDescription(index=0, timestamp_seconds=10.0, description='a', confidence=0.5),
            FrameDescription(index=1, timestamp_seconds=12.0, description='b', confidence=0.5),
        ],
    )


def test_plan_document_parses(monkeypatch):
    fake = FakeClient([
        json.dumps({
            'doc_type': 'tutorial',
            'title': 'My Tutorial',
            'audience': 'beginners',
            'language': 'zh-CN',
            'outline': ['Intro', 'Setup', 'Conclusion'],
            'focus_instructions': 'focus on setup',
        })
    ])
    monkeypatch.setattr('vid2kb.llm.deepseek_client', lambda: fake)

    spec = plan_document('write a tutorial', 'transcript text', 'timeline summary')

    assert isinstance(spec, DocSpec)
    assert spec.doc_type == 'tutorial'
    assert spec.title == 'My Tutorial'
    assert spec.audience == 'beginners'
    assert spec.language == 'zh-CN'
    assert spec.outline == ['Intro', 'Setup', 'Conclusion']
    assert spec.focus_instructions == 'focus on setup'


def test_plan_document_invalid_json_raises(monkeypatch):
    fake = FakeClient(['{not json'])
    monkeypatch.setattr('vid2kb.llm.deepseek_client', lambda: fake)

    with pytest.raises(ValueError):
        plan_document('p', 't', 'tl')


def test_compose_document_parses(monkeypatch):
    fake = FakeClient([json.dumps(DOC_JSON)])
    monkeypatch.setattr('vid2kb.llm.deepseek_client', lambda: fake)

    doc = compose_document(_spec(), 'transcript', _timeline(), 'user prompt')

    assert isinstance(doc, KnowledgeDocument)
    assert doc.sections[0].heading == 'Introduction'


def test_compose_document_validation_retry(monkeypatch):
    invalid = json.dumps({k: v for k, v in DOC_JSON.items() if k != 'title'})
    fake = FakeClient([invalid, json.dumps(DOC_JSON)])
    monkeypatch.setattr('vid2kb.llm.deepseek_client', lambda: fake)

    doc = compose_document(_spec(), 'transcript', _timeline(), 'user prompt')

    assert doc.title == 'Test Tutorial'
    assert fake.calls == 2


def test_validate_document_clean():
    doc = KnowledgeDocument.model_validate(DOC_JSON)

    problems = validate_document(doc, _spec(), 'transcript', {10.0, 12.0})

    assert problems == []


def test_validate_document_problems():
    doc = KnowledgeDocument.model_validate({
        'title': '',
        'doc_type': 'tutorial',
        'audience': 'developers',
        'summary': 'summary',
        'sections': [
            {'heading': 'Wrong Heading', 'body_md': 'x', 'source_timestamps': [999.0]},
        ],
        'key_points': [],
    })
    spec = DocSpec(
        doc_type='tutorial',
        title='',
        audience='developers',
        outline=['Missing Heading'],
        focus_instructions='',
    )

    problems = validate_document(doc, spec, 'transcript', {10.0, 12.0})
    joined = '\n'.join(problems)

    assert problems
    assert 'title' in joined
    assert 'Missing Heading' in joined
    assert '999.0' in joined


def test_render_markdown_contains_sections():
    doc = KnowledgeDocument.model_validate(DOC_JSON)

    md = render_markdown(doc)

    assert '# ' in md
    assert 'Test Tutorial' in md
    assert '## Introduction' in md
    assert '[00:12]' in md
    assert 'key point one' in md


def test_render_pdf_writes_file(tmp_path):
    out = tmp_path / 'out.pdf'

    result = render_pdf('# 测试\n\n正文', out)

    assert result == out
    assert out.exists()
    assert out.read_bytes()[:4] == b'%PDF'
    assert out.stat().st_size > 500
