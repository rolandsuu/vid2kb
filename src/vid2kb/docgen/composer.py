from __future__ import annotations

import json

from vid2kb.config import settings
from vid2kb.docgen.planner import DocSpec
from vid2kb.schemas import KnowledgeDocument, VisualTimeline

MAX_TRANSCRIPT_CHARS = 60000
_SENTENCE_ENDERS = '。！？!?.\n'


def _truncate_transcript(transcript: str) -> tuple[str, bool]:
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript, False
    cut = transcript[:MAX_TRANSCRIPT_CHARS]
    last = -1
    for ch in _SENTENCE_ENDERS:
        idx = cut.rfind(ch)
        if idx > last:
            last = idx
    if last > 0:
        cut = cut[: last + 1]
    return cut, True


def _parse(raw: str) -> KnowledgeDocument:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(raw) from e
    return KnowledgeDocument.model_validate(data)


def compose_document(
    spec: DocSpec,
    transcript: str,
    timeline: VisualTimeline,
    user_prompt: str,
) -> KnowledgeDocument:
    from vid2kb.llm import deepseek_client, record_usage

    system_prompt = (
        '你是一个知识文档撰写助手。严格遵循以下规则：'
        '1. 文档最多 12 个章节（sections）。'
        '2. 每个章节的 body_md 不超过 600 字。'
        '3. 每个章节的 source_timestamps 必须引用真实出现在转录文本段落或时间线帧中的视频秒数。'
        '4. 文档标题必须使用给定标题。'
        '5. 使用指定的语言撰写。'
        '输出一个符合 KnowledgeDocument 结构的 json 对象：title、doc_type、audience、summary、'
        'sections（数组，每个元素含 heading、body_md、source_timestamps 数组）、'
        'key_points（字符串数组）、glossary（对象）、warnings（字符串数组）。'
    )

    transcript_text, truncated = _truncate_transcript(transcript)
    if truncated:
        transcript_text += '\n\n[注意：转录文本过长，已在此截断。]'

    frames_text = '\n'.join(
        f'- {f.timestamp_seconds:.1f}s: {f.description}' for f in timeline.frames[:20]
    )

    user_message = (
        'You must output a valid json object. Respond with json only.\n\n'
        f'标题（title）：{spec.title}\n'
        f'文档类型（doc_type）：{spec.doc_type}\n'
        f'目标读者（audience）：{spec.audience}\n'
        f'语言（language）：{spec.language}\n'
        f'大纲（outline）：\n' + '\n'.join(f'- {h}' for h in spec.outline) + '\n\n'
        f'重点说明（focus_instructions）：{spec.focus_instructions}\n\n'
        f'用户需求（user_prompt）：{user_prompt}\n\n'
        f'转录文本（已截断到最多 {MAX_TRANSCRIPT_CHARS} 字符）：\n{transcript_text}\n\n'
        f'时间线摘要：{timeline.summary}\n\n'
        f'前 20 帧：\n{frames_text}\n'
    )

    client = deepseek_client()
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_message},
    ]

    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        response_format={'type': 'json_object'},
    )
    record_usage('deepseek', response)
    raw = response.choices[0].message.content

    try:
        return _parse(raw)
    except ValueError as e:
        messages.append({'role': 'assistant', 'content': raw})
        messages.append(
            {'role': 'user', 'content': f'你的输出无效，请修复错误后重新输出 json：{e}'}
        )
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            response_format={'type': 'json_object'},
        )
        record_usage('deepseek', response)
        raw2 = response.choices[0].message.content
        try:
            return _parse(raw2)
        except ValueError as e2:
            raise ValueError(raw2) from e2
