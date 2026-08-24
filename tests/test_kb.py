from __future__ import annotations

import urllib.request
from uuid import uuid4

import psycopg
import pytest

from vid2kb.config import settings
from vid2kb.kb.embed import embed_texts
from vid2kb.kb.query import query_knowledge
from vid2kb.kb.store import ingest_document, query_documents


def _pg_conn_url() -> str:
    return settings.pgvector_database_url.replace('+psycopg', '')


def _pg_available() -> bool:
    try:
        with psycopg.connect(_pg_conn_url()) as conn:
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        return True
    except Exception:
        return False


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=3):
            return True
    except Exception:
        return False


_integration = pytest.mark.skipif(
    not (_pg_available() and _ollama_available()),
    reason='needs postgres+ollama',
)


def _drop_table(table_name: str) -> None:
    try:
        with psycopg.connect(_pg_conn_url()) as conn:
            conn.execute(f'DROP TABLE IF EXISTS {table_name}')
            conn.execute(f'DROP TABLE IF EXISTS data_{table_name}')
    except Exception:
        pass


@_integration
def test_embed_dims():
    vectors = embed_texts(['你好世界'])

    assert len(vectors) == 1
    assert len(vectors[0]) == settings.embed_dims
    assert all(isinstance(x, float) for x in vectors[0])


@_integration
def test_ingest_and_query_roundtrip():
    table_name = f'vid2kb_test_{uuid4().hex[:8]}'
    try:
        md_text = (
            '# 集成测试文档\n\n'
            '## 向量检索\n'
            '本文介绍 pgvector向量检索 的基本原理，用于语义相似度召回。\n\n'
            '## 索引构建\n'
            '使用 HNSW 索引可以加速近似最近邻搜索。\n\n'
            '## 应用场景\n'
            '在知识库问答中，向量检索负责召回相关片段。\n'
        )
        metadata = {
            'run_id': 'test',
            'doc_title': '集成测试',
            'doc_type': 'summary',
            'source_url': '',
            'timestamp_marks': [],
        }

        count = ingest_document(md_text, metadata, 'doc-integration-1', table_name=table_name)
        assert count >= 2

        results = query_documents('pgvector向量检索是什么', top_k=2, table_name=table_name)
        assert results
        assert any('pgvector向量检索' in r['text'] for r in results)
        assert all(r['metadata'].get('doc_title') == '集成测试' for r in results)
    finally:
        _drop_table(table_name)


@_integration
def test_query_empty_knowledge():
    table_name = f'vid2kb_test_{uuid4().hex[:8]}'
    try:
        results = query_documents('anything', top_k=5, table_name=table_name)
        assert results == []
    finally:
        _drop_table(table_name)


def test_query_knowledge_wrapper_uses_default_table(monkeypatch):
    captured = {}

    def fake_query_documents(question, top_k=5, table_name='vid2kb_nodes'):
        captured['question'] = question
        captured['top_k'] = top_k
        captured['table_name'] = table_name
        return [{'node_id': 'n1', 'score': 0.9, 'text': 'x', 'metadata': {}}]

    monkeypatch.setattr('vid2kb.kb.query.query_documents', fake_query_documents)

    result = query_knowledge('hello', top_k=3)

    assert result == [{'node_id': 'n1', 'score': 0.9, 'text': 'x', 'metadata': {}}]
    assert captured == {'question': 'hello', 'top_k': 3, 'table_name': 'vid2kb_nodes'}
