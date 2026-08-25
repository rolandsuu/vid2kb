from __future__ import annotations

import json
import tempfile
from email.parser import BytesParser
from email.policy import default as default_policy
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vid2kb.config import settings
from vid2kb.jobs import db
from vid2kb.jobs.worker import run_agent
from vid2kb.media.store import ArtifactStore

router = APIRouter()

UPLOADS_DIR = Path('data') / 'uploads'
ARTIFACTS = {'document.md', 'document.pdf'}


class CreateRunRequest(BaseModel):
    prompt: str
    source_url: str


def _parse_multipart(body: bytes, content_type: str) -> dict:
    raw = f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + body
    msg = BytesParser(policy=default_policy).parsebytes(raw)
    result: dict = {}
    if not msg.is_multipart():
        return result
    for part in msg.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name is None:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b''
        if filename is not None:
            result['filename'] = filename
            result[name] = payload
        else:
            result[name] = payload.decode('utf-8', errors='replace')
    return result


async def _run_temporal(run_id: str, prompt: str, source: str) -> None:
    from vid2kb.temporal import client as temporal_client

    try:
        await temporal_client.start_run(run_id, prompt, source)
        db.update_run(run_id, status='running')
        result = await temporal_client.get_run_result(run_id)
        if result.get('status') == 'failed':
            db.update_run(
                run_id,
                status='failed',
                result_json=json.dumps(result, ensure_ascii=False),
                error='; '.join(result.get('errors', [])),
            )
        else:
            db.update_run(run_id, status='done', result_json=json.dumps(result, ensure_ascii=False))
    except Exception as e:
        db.update_run(run_id, status='failed', error=str(e))


def _queue(run_id: str, prompt: str, source: str, background_tasks: BackgroundTasks) -> dict:
    db.create_run(prompt=prompt, source=source, run_id=run_id)
    if settings.run_driver == 'temporal':
        background_tasks.add_task(_run_temporal, run_id, prompt, source)
    else:
        background_tasks.add_task(run_agent, run_id, prompt, source)
    return {'run_id': run_id, 'status': 'queued'}


@router.post('/runs', status_code=202)
def create_run_json(body: CreateRunRequest, background_tasks: BackgroundTasks) -> dict:
    run_id = uuid4().hex[:12]
    return _queue(run_id, body.prompt, body.source_url, background_tasks)


@router.post('/runs/upload', status_code=202)
async def create_run_upload(request: Request, background_tasks: BackgroundTasks) -> dict:
    body = await request.body()
    parsed = _parse_multipart(body, request.headers.get('content-type', ''))
    prompt = parsed.get('prompt', '')
    filename = parsed.get('filename', '')
    run_id = uuid4().hex[:12]
    suffix = Path(filename).suffix or '.bin'
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = UPLOADS_DIR / f'{run_id}{suffix}'
    local_path.write_bytes(parsed.get('video', b''))
    return _queue(run_id, prompt, str(local_path), background_tasks)


def _get_run(run_id: str):
    db.init_db()
    with db.SessionLocal() as session:
        return session.get(db.Run, run_id)


@router.get('/runs/{run_id}')
def get_run(run_id: str) -> dict:
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail='run not found')
    return {
        'run_id': run.id,
        'status': run.status,
        'prompt': run.prompt,
        'source': run.source,
        'error': run.error,
        'result': json.loads(run.result_json) if run.result_json else None,
        'created_at': run.created_at,
        'updated_at': run.updated_at,
    }


@router.get('/runs/{run_id}/artifact/{name}')
def get_artifact(run_id: str, name: str) -> FileResponse:
    if name not in ARTIFACTS:
        raise HTTPException(status_code=404, detail='unknown artifact')
    store = ArtifactStore(run_id)
    if settings.artifact_store == 's3':
        if not store.exists(f'out/{name}'):
            raise HTTPException(status_code=404, detail='artifact not found')
        path = store.get_file(f'out/{name}', Path(tempfile.gettempdir()) / f'vid2kb-{run_id}-{name}')
    else:
        path = store.out / name
        if not path.exists():
            raise HTTPException(status_code=404, detail='artifact not found')
    return FileResponse(path)
