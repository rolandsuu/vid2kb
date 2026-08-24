from __future__ import annotations
from openai import OpenAI

def deepseek_client() -> OpenAI:
    from vid2kb.config import settings
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

def dashscope_client() -> OpenAI:
    from vid2kb.config import settings
    return OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
