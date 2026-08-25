from __future__ import annotations

import threading

from openai import OpenAI

_local = threading.local()


def deepseek_client() -> OpenAI:
    from vid2kb.config import settings
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def dashscope_client() -> OpenAI:
    from vid2kb.config import settings
    return OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)


def record_usage(vendor: str, response) -> None:
    """Record token usage from an OpenAI-compatible completion response.

    Usage is accumulated on the current thread and drained by the agent tools
    when they write their per-stage audit records, so the eval harness can
    attribute tokens/cost to each pipeline stage.
    """
    usage = getattr(response, 'usage', None)
    if usage is None:
        return
    log = getattr(_local, 'usage_log', None)
    if log is None:
        log = []
        _local.usage_log = log
    log.append(
        {
            'vendor': vendor,
            'input': getattr(usage, 'prompt_tokens', None),
            'output': getattr(usage, 'completion_tokens', None),
        }
    )


def drain_usage() -> dict:
    """Return token usage accumulated since the last drain, keyed by vendor."""
    log = getattr(_local, 'usage_log', None)
    if not log:
        return {}
    _local.usage_log = []
    totals: dict = {}
    for entry in log:
        bucket = totals.setdefault(entry['vendor'], {'input': 0, 'output': 0})
        for key in ('input', 'output'):
            value = entry[key]
            if isinstance(value, (int, float)):
                bucket[key] += int(value)
    return totals
