from __future__ import annotations

from temporalio.client import Client

from vid2kb.temporal.workflow import RunVideoJobArgs, RunVideoJobWorkflow

TASK_QUEUE = 'vid2kb-tasks'


async def _connect() -> Client:
    from vid2kb.config import settings

    return await Client.connect(settings.temporal_address)


async def start_run(run_id: str, prompt: str, source: str) -> str:
    client = await _connect()
    handle = await client.start_workflow(
        RunVideoJobWorkflow.run,
        RunVideoJobArgs(run_id=run_id, prompt=prompt, source=source),
        id=run_id,
        task_queue=TASK_QUEUE,
    )
    return handle.id


async def get_run_status(run_id: str) -> dict:
    client = await _connect()
    handle = client.get_workflow_handle(run_id)
    desc = await handle.describe()
    return {'workflow_id': run_id, 'status': str(desc.status)}


async def get_run_result(run_id: str) -> dict:
    client = await _connect()
    handle = client.get_workflow_handle(run_id)
    return await handle.result()
