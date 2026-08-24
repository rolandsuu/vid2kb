from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from vid2kb.api.main import app


@pytest.fixture(autouse=True)
def _fake_run_agent(monkeypatch):
    def _fake(run_id, prompt, source):
        from vid2kb.jobs import db

        db.update_run(
            run_id,
            status='done',
            result_json=json.dumps({'final_report': {'run_id': run_id, 'title': 'x'}}),
        )

    monkeypatch.setattr('vid2kb.api.runs.run_agent', _fake)


@asynccontextmanager
async def _client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        yield client


async def _poll_status(client, run_id, target='done', timeout=5.0):
    deadline = time.time() + timeout
    data = None
    while time.time() < deadline:
        resp = await client.get(f'/runs/{run_id}')
        assert resp.status_code == 200
        data = resp.json()
        if data['status'] == target:
            return data
        time.sleep(0.05)
    return data


@pytest.mark.asyncio
async def test_health():
    async with _client() as client:
        resp = await client.get('/health')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok'}


@pytest.mark.asyncio
async def test_create_run_json_and_poll():
    async with _client() as client:
        resp = await client.post(
            '/runs',
            json={'prompt': '整理成教程', 'source_url': 'https://example.com/v.mp4'},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body['status'] == 'queued'
        assert 'run_id' in body

        data = await _poll_status(client, body['run_id'])

    assert data['status'] == 'done'
    assert data['result']['final_report']['title'] == 'x'


@pytest.mark.asyncio
async def test_create_run_upload():
    uploads_dir = Path('data') / 'uploads'
    async with _client() as client:
        resp = await client.post(
            '/runs/upload',
            files={'video': ('t.mp4', b'fakevideodata', 'video/mp4')},
            data={'prompt': 'x'},
        )
        assert resp.status_code == 202
        run_id = resp.json()['run_id']

        uploaded = uploads_dir / f'{run_id}.mp4'
        assert uploaded.exists()
        try:
            data = await _poll_status(client, run_id)
        finally:
            uploaded.unlink(missing_ok=True)

    assert data['status'] == 'done'


@pytest.mark.asyncio
async def test_artifact_missing_returns_404():
    async with _client() as client:
        resp = await client.get('/runs/some-run-id/artifact/nope.txt')
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_query_endpoint(monkeypatch):
    monkeypatch.setattr(
        'vid2kb.api.query.query_knowledge',
        lambda question, top_k=5: [{'node_id': 'n1', 'score': 0.9, 'text': '内容', 'metadata': {}}],
    )
    async with _client() as client:
        resp = await client.post('/query', json={'question': '测试'})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body['results']) == 1
    assert body['results'][0]['node_id'] == 'n1'
