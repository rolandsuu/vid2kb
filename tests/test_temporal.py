from __future__ import annotations

import pytest
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from vid2kb.temporal.client import TASK_QUEUE
from vid2kb.temporal.workflow import RunVideoJobArgs, RunVideoJobWorkflow, _run_pipeline


def _stub_results() -> dict:
    return {
        'ingest': {'video_path': '/tmp/v.mp4', 'steps': ['ingest: ok']},
        'transcribe': {'transcript': 'hello', 'transcript_language': 'zh', 'steps': ['transcribe: ok']},
        'visual': {'timeline': {'frames': [], 'summary': 's'}, 'steps': ['visual: ok']},
        'compose': {'document': {'title': 't', 'doc_type': 'tutorial'}, 'doc_spec': {}, 'steps': ['compose: ok']},
        'render': {'markdown': '# hi', 'pdf_path': '/tmp/o.pdf', 'steps': ['render: ok']},
        'ingest_kb': {'kb_doc_id': 'r1', 'kb_node_count': 3, 'steps': ['ingest_kb: ok']},
    }


@pytest.mark.asyncio
async def test_pipeline_happy_path_unit():
    calls: list[str] = []

    async def runner(name, fn, arg, retry_policy):
        calls.append(name)
        return _stub_results()[name]

    out = await _run_pipeline(RunVideoJobArgs('r1', 'p', 'src'), runner)

    assert out['status'] == 'done'
    assert out['errors'] == []
    assert out['stages']['render']['markdown'] == '# hi'
    assert out['stages']['ingest_kb']['kb_node_count'] == 3
    assert calls == ['ingest', 'transcribe', 'visual', 'compose', 'render', 'ingest_kb']


@pytest.mark.asyncio
async def test_pipeline_failure_unit():
    calls: list[str] = []

    async def runner(name, fn, arg, retry_policy):
        calls.append(name)
        if name == 'ingest':
            raise RuntimeError('boom')
        return {}

    out = await _run_pipeline(RunVideoJobArgs('r1', 'p', 'src'), runner)

    assert out['status'] == 'failed'
    assert any('ingest' in e for e in out['errors'])
    assert calls == ['ingest']


@activity.defn(name='ingest')
async def _stub_ingest(args):
    return {'video_path': '/tmp/v.mp4'}


@activity.defn(name='transcribe')
async def _stub_transcribe(args):
    return {'transcript': 'hello', 'transcript_language': 'zh'}


@activity.defn(name='visual')
async def _stub_visual(args):
    return {'timeline': {'frames': [], 'summary': 's'}}


@activity.defn(name='compose')
async def _stub_compose(args):
    return {'document': {'title': 't', 'doc_type': 'tutorial'}, 'doc_spec': {}}


@activity.defn(name='render')
async def _stub_render(args):
    return {'markdown': '# hi', 'pdf_path': '/tmp/o.pdf'}


@activity.defn(name='ingest_kb')
async def _stub_ingest_kb(args):
    return {'kb_doc_id': 'r1', 'kb_node_count': 3}


_STUBS = [_stub_ingest, _stub_transcribe, _stub_visual, _stub_compose, _stub_render, _stub_ingest_kb]


async def _run_workflow_local(run_id: str, stubs):
    try:
        env = await WorkflowEnvironment.start_local()
    except Exception as exc:
        pytest.skip(f'temporal test server unavailable: {exc}')
    async with env:
        client = env.client
        async with Worker(client, task_queue=TASK_QUEUE, workflows=[RunVideoJobWorkflow], activities=stubs):
            handle = await client.start_workflow(
                RunVideoJobWorkflow.run,
                RunVideoJobArgs(run_id, 'prompt', 'src'),
                id=run_id,
                task_queue=TASK_QUEUE,
            )
            return await handle.result()


@pytest.mark.asyncio
async def test_workflow_happy_path_local():
    out = await _run_workflow_local('run-happy', _STUBS)
    assert out['status'] == 'done'
    assert out['stages']['render']['markdown'] == '# hi'
    assert out['stages']['ingest_kb']['kb_node_count'] == 3


@pytest.mark.asyncio
async def test_workflow_retry_local():
    attempts = {'n': 0}

    @activity.defn(name='transcribe')
    async def flaky_transcribe(args):
        attempts['n'] += 1
        if attempts['n'] == 1:
            raise ApplicationError('boom once')
        return {'transcript': 'hello', 'transcript_language': 'zh'}

    stubs = [_stub_ingest, flaky_transcribe, _stub_visual, _stub_compose, _stub_render, _stub_ingest_kb]

    out = await _run_workflow_local('run-retry', stubs)
    assert out['status'] == 'done'
    assert out['stages']['transcribe']['transcript'] == 'hello'
    assert attempts['n'] == 2
