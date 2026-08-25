from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from vid2kb.temporal.activities import (
    compose_activity,
    ingest_activity,
    ingest_kb_activity,
    render_activity,
    transcribe_activity,
    visual_activity,
)
from vid2kb.temporal.client import TASK_QUEUE
from vid2kb.temporal.workflow import RunVideoJobWorkflow


async def main() -> None:
    from vid2kb.config import settings

    client = await Client.connect(settings.temporal_address)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunVideoJobWorkflow],
        activities=[
            ingest_activity,
            transcribe_activity,
            visual_activity,
            compose_activity,
            render_activity,
            ingest_kb_activity,
        ],
    )
    await worker.run()


if __name__ == '__main__':
    asyncio.run(main())
