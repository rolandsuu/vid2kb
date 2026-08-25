from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError


@dataclass
class IngestInput:
    run_id: str
    source: str


@dataclass
class TranscribeInput:
    run_id: str
    video_path: str


@dataclass
class VisualInput:
    run_id: str
    video_path: str
    transcript: str
    prompt: str


@dataclass
class ComposeInput:
    run_id: str
    transcript: str
    timeline: dict
    prompt: str


@dataclass
class RenderInput:
    run_id: str
    document: dict


@dataclass
class IngestKbInput:
    run_id: str
    markdown: str
    document: dict
    source: str


def _state(**fields: Any) -> dict:
    state = {'run_id': fields.pop('run_id'), 'errors': [], 'steps': []}
    state.update(fields)
    return state


def _raise_if_failed(result: dict) -> dict:
    if result.get('errors'):
        raise ApplicationError('; '.join(result['errors']))
    return result


@activity.defn(name='ingest')
async def ingest_activity(args: IngestInput) -> dict:
    from vid2kb.agent.tools import tool_ingest

    return _raise_if_failed(tool_ingest(_state(run_id=args.run_id, source=args.source)))


@activity.defn(name='transcribe')
async def transcribe_activity(args: TranscribeInput) -> dict:
    from vid2kb.agent.tools import tool_transcribe

    return _raise_if_failed(
        tool_transcribe(_state(run_id=args.run_id, video_path=args.video_path))
    )


@activity.defn(name='visual')
async def visual_activity(args: VisualInput) -> dict:
    from vid2kb.agent.tools import tool_visual

    return _raise_if_failed(
        tool_visual(
            _state(
                run_id=args.run_id,
                video_path=args.video_path,
                transcript=args.transcript,
                user_prompt=args.prompt,
            )
        )
    )


@activity.defn(name='compose')
async def compose_activity(args: ComposeInput) -> dict:
    from vid2kb.agent.tools import tool_compose

    return _raise_if_failed(
        tool_compose(
            _state(
                run_id=args.run_id,
                transcript=args.transcript,
                timeline=args.timeline,
                user_prompt=args.prompt,
            )
        )
    )


@activity.defn(name='render')
async def render_activity(args: RenderInput) -> dict:
    from vid2kb.agent.tools import tool_render

    return _raise_if_failed(tool_render(_state(run_id=args.run_id, document=args.document)))


@activity.defn(name='ingest_kb')
async def ingest_kb_activity(args: IngestKbInput) -> dict:
    from vid2kb.agent.tools import tool_ingest_kb

    return _raise_if_failed(
        tool_ingest_kb(
            _state(
                run_id=args.run_id,
                markdown=args.markdown,
                document=args.document,
                source=args.source,
            )
        )
    )
