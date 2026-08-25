from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable

from temporalio import workflow
from temporalio.common import RetryPolicy

from vid2kb.temporal.activities import (
    ComposeInput,
    IngestInput,
    IngestKbInput,
    RenderInput,
    TranscribeInput,
    VisualInput,
    compose_activity,
    ingest_activity,
    ingest_kb_activity,
    render_activity,
    transcribe_activity,
    visual_activity,
)


@dataclass
class RunVideoJobArgs:
    run_id: str
    prompt: str
    source: str


RETRY_STANDARD = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
)

RETRY_LIGHT = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=2,
)

StageRunner = Callable[[str, Callable, object, RetryPolicy], Awaitable[dict]]


async def _run_pipeline(args: RunVideoJobArgs, run_stage: StageRunner) -> dict:
    result: dict = {'run_id': args.run_id, 'status': 'done', 'stages': {}, 'errors': []}

    def _fail(stage: str, exc: Exception) -> dict:
        result['errors'].append(f'{stage}: {exc}')
        result['status'] = 'failed'
        return result

    try:
        ingest = await run_stage(
            'ingest',
            ingest_activity,
            IngestInput(run_id=args.run_id, source=args.source),
            RETRY_LIGHT,
        )
        result['stages']['ingest'] = ingest
    except Exception as exc:
        return _fail('ingest', exc)

    try:
        transcribe = await run_stage(
            'transcribe',
            transcribe_activity,
            TranscribeInput(run_id=args.run_id, video_path=ingest['video_path']),
            RETRY_STANDARD,
        )
        result['stages']['transcribe'] = transcribe
    except Exception as exc:
        return _fail('transcribe', exc)

    try:
        visual = await run_stage(
            'visual',
            visual_activity,
            VisualInput(
                run_id=args.run_id,
                video_path=ingest['video_path'],
                transcript=transcribe.get('transcript', ''),
                prompt=args.prompt,
            ),
            RETRY_STANDARD,
        )
        result['stages']['visual'] = visual
    except Exception as exc:
        return _fail('visual', exc)

    try:
        compose = await run_stage(
            'compose',
            compose_activity,
            ComposeInput(
                run_id=args.run_id,
                transcript=transcribe.get('transcript', ''),
                timeline=visual['timeline'],
                prompt=args.prompt,
            ),
            RETRY_STANDARD,
        )
        result['stages']['compose'] = compose
    except Exception as exc:
        return _fail('compose', exc)

    try:
        render = await run_stage(
            'render',
            render_activity,
            RenderInput(run_id=args.run_id, document=compose['document']),
            RETRY_LIGHT,
        )
        result['stages']['render'] = render
    except Exception as exc:
        return _fail('render', exc)

    try:
        kb = await run_stage(
            'ingest_kb',
            ingest_kb_activity,
            IngestKbInput(
                run_id=args.run_id,
                markdown=render['markdown'],
                document=compose['document'],
                source=args.source,
            ),
            RETRY_LIGHT,
        )
        result['stages']['ingest_kb'] = kb
    except Exception as exc:
        return _fail('ingest_kb', exc)

    return result


@workflow.defn
class RunVideoJobWorkflow:
    @workflow.run
    async def run(self, args: RunVideoJobArgs) -> dict:
        async def run_stage(
            name: str,
            fn: Callable,
            arg: object,
            retry_policy: RetryPolicy,
        ) -> dict:
            return await workflow.execute_activity(
                fn,
                arg,
                retry_policy=retry_policy,
                start_to_close_timeout=timedelta(minutes=30),
            )

        return await _run_pipeline(args, run_stage)
