from __future__ import annotations

import json

from vid2kb.agent.checkpointer import build_checkpointer
from vid2kb.agent.graph import build_graph
from vid2kb.jobs import db


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
    except Exception as e:
        db.update_run(run_id, status='failed', error=str(e))
        raise
    db.update_run(run_id, status='done', result_json=json.dumps(result))
    return result
