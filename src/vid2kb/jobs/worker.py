from __future__ import annotations

import json
from datetime import date, datetime

from vid2kb.agent.checkpointer import build_checkpointer
from vid2kb.agent.graph import build_graph
from vid2kb.jobs import db


def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'content'):
        return str(obj.content)
    return str(obj)


def run_agent(run_id: str, prompt: str, source: str) -> dict:
    db.init_db()
    db.update_run(run_id, status='running')
    graph = build_graph(checkpointer=build_checkpointer())
    try:
        result = graph.invoke(
            {
                'run_id': run_id,
                'user_prompt': prompt,
                'source': source,
                'errors': [],
                'steps': [],
                'iterations': 0,
            },
            config={'configurable': {'thread_id': run_id}},
        )
        db.update_run(run_id, status='done', result_json=json.dumps(_json_safe(result), ensure_ascii=False))
        return result
    except Exception as e:
        db.update_run(run_id, status='failed', error=str(e))
        raise
