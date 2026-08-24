from __future__ import annotations

import base64
import json
from pathlib import Path

from vid2kb.config import settings
from vid2kb.schemas import FrameDescription, VisualTimeline

BATCH_SIZE = 5

MAX_EXCERPT_CHARS = 4000


def _frame_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def analyze_frames(
    frames: list[tuple[int, float, Path]],
    user_prompt: str,
    transcript_excerpt: str,
) -> VisualTimeline:
    from vid2kb.llm import dashscope_client

    system_prompt = (
        '你是一个视频画面分析助手。请输出结构化的 JSON，格式与 VisualTimeline 一致：'
        '包含 summary（字符串）、frames（数组，每个元素包含 index、timestamp_seconds、'
        'description、visible_text、actions、confidence 字段）以及 warnings（字符串数组）。'
    )

    excerpt = transcript_excerpt[:MAX_EXCERPT_CHARS]

    batches = [frames[i:i + BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]

    merged_frames: list[FrameDescription] = []
    warnings: list[str] = []
    summary = ''
    successful = 0

    for batch_num, batch in enumerate(batches, start=1):
        content_parts: list[dict] = [
            {
                'type': 'text',
                'text': (
                    'You must output a JSON object. Respond with valid JSON only.\n\n'
                    f'User prompt:\n{user_prompt}\n\n'
                    f'Transcript excerpt:\n{excerpt}\n\n'
                    'For each frame image below, describe its visual content, any visible text, '
                    'and the actions being performed. Provide a confidence score between 0 and 1. '
                    'Use the provided frame index as "index" and the provided timestamp as '
                    '"timestamp_seconds". Keep frame order as given.'
                ),
            }
        ]
        for index, timestamp, path in batch:
            content_parts.append(
                {'type': 'image_url', 'image_url': {'url': _frame_to_data_url(path)}}
            )
            content_parts.append(
                {'type': 'text', 'text': f'frame index={index}, timestamp_seconds={timestamp}'}
            )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content_parts},
        ]

        try:
            client = dashscope_client()
            response = client.chat.completions.create(
                model=settings.vision_model,
                messages=messages,
                response_format={'type': 'json_object'},
            )
            raw = response.choices[0].message.content
            partial = VisualTimeline.model_validate(json.loads(raw))
            merged_frames.extend(partial.frames)
            warnings.extend(partial.warnings)
            if not summary:
                summary = partial.summary
            successful += 1
        except Exception as e:
            warnings.append(f'batch {batch_num} failed: {e}')

    if successful == 0:
        raise RuntimeError('vision analysis failed')

    return VisualTimeline(summary=summary, frames=merged_frames, warnings=warnings)
