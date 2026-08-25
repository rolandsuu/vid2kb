from __future__ import annotations

import json

from pydantic import BaseModel, Field

from vid2kb.config import settings

MAX_EXCERPT_CHARS = 6000


class DocSpec(BaseModel):
    doc_type: str = Field(description='tutorial|summary|meeting_notes|notes|qa')
    title: str
    audience: str
    language: str = 'zh-CN'
    outline: list[str]
    focus_instructions: str


def plan_document(user_prompt: str, transcript: str, timeline_summary: str) -> DocSpec:
    from vid2kb.llm import deepseek_client, record_usage

    system_prompt = (
        '你是一个文档规划助手。请根据用户需求和内容信号，推断文档类型'
        '（tutorial、summary、meeting_notes、notes 或 qa）、标题、目标读者、语言'
        '（默认 zh-CN）、大纲（章节标题列表）以及重点说明。必须输出一个 json 对象，'
        '字段与 DocSpec 一致：doc_type、title、audience、language、outline（字符串数组）、'
        'focus_instructions。'
    )

    excerpt = transcript[:MAX_EXCERPT_CHARS]

    user_message = (
        'You must output a valid json object. Respond with json only.\n\n'
        f'用户需求（user_prompt）：\n{user_prompt}\n\n'
        f'时间线摘要（timeline_summary）：\n{timeline_summary}\n\n'
        f'转录文本摘录（已截断到 {MAX_EXCERPT_CHARS} 字符）：\n{excerpt}\n'
    )

    client = deepseek_client()
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ],
        response_format={'type': 'json_object'},
    )
    record_usage('deepseek', response)
    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
        return DocSpec.model_validate(data)
    except Exception as e:
        raise ValueError(raw) from e
