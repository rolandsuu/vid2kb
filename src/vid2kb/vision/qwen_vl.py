from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from vid2kb.config import settings
from vid2kb.schemas import FrameDescription, VisualTimeline

BATCH_SIZE = 5

MAX_EXCERPT_CHARS = 4000


def _frame_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def _parse_json(raw: str) -> dict:
    text = (raw or '').strip()
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f'no JSON object found; raw preview: {text[:400]!r}')
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f'JSON decode failed: {e}; raw preview: {text[:400]!r}') from e


def _build_timeline(obj: dict, batch: list[tuple[int, float, Path]]) -> VisualTimeline:
    summary = obj.get('summary', '') if isinstance(obj.get('summary'), str) else ''
    warnings = list(obj.get('warnings') or [])

    if 'frames' in obj and obj['frames'] is not None:
        frames = [FrameDescription.model_validate(f) for f in obj['frames']]
        if len(frames) < len(batch):
            warnings.append(f'expected {len(batch)} frames, got {len(frames)}')
    elif 'description' in obj:
        index, timestamp, _ = batch[0]
        frames = [
            FrameDescription(
                index=index,
                timestamp_seconds=timestamp,
                description=obj['description'],
                confidence=0.5,
            )
        ]
        warnings.append('model returned single description; mapped to first frame')
    else:
        raise ValueError(f'model output missing frames/description: {json.dumps(obj, ensure_ascii=False)[:400]!r}')

    return VisualTimeline(summary=summary, frames=frames, warnings=warnings)


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
                    f'你会收到 {len(batch)} 张视频帧图片。'
                    '每张图片我按顺序发送，请为每一帧输出一个条目。'
                    '每个条目必须包含以下字段：index、timestamp_seconds、description、'
                    'visible_text、actions、confidence。'
                    '最后输出 JSON 对象：{"summary": "...", "frames": [N 个条目], "warnings": [...]}。'
                    f'frames 数组必须恰好包含 {len(batch)} 个条目，index 必须是给定的帧序号。'
                    '请输出 json。\n\n'
                    f'用户指令：\n{user_prompt}\n\n'
                    f'字幕摘要：\n{excerpt}'
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
            obj = _parse_json(raw)
            partial = _build_timeline(obj, batch)
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
