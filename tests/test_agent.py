from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

from vid2kb.agent import graph as graph_mod
from vid2kb.agent import tools
from vid2kb.jobs.worker import _json_safe


def _fake_llm_client(monkeypatch, content: str):
    class FakeMessage:
        pass

    class FakeChoice:
        pass

    class FakeResponse:
        pass

    class FakeCompletions:
        def create(self, **kwargs):
            msg = FakeMessage()
            msg.content = content
            choice = FakeChoice()
            choice.message = msg
            resp = FakeResponse()
            resp.choices = [choice]
            return resp

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr('vid2kb.llm.deepseek_client', lambda: FakeClient())


def _thread():
    return {'configurable': {'thread_id': f'test-{uuid4().hex[:8]}'}}


def _initial(run_id='run'):
    return {
        'run_id': run_id,
        'user_prompt': 'summarize this video',
        'source': '',
        'errors': [],
        'steps': [],
        'iterations': 0,
    }


def test_graph_compiles():
    g = graph_mod.build_graph()
    assert g is not None
    assert 'planner' in g.get_graph().nodes


def test_planner_llm_output_parsed(monkeypatch):
    _fake_llm_client(monkeypatch, '{"next": "ingest", "reason": "no video yet"}')

    g = graph_mod.build_graph()
    result = g.invoke(_initial(), config=_thread(), interrupt_before=['ingest'])

    assert result['next'] == 'ingest'
    assert result['iterations'] == 1


def test_planner_invalid_next_defaults_to_report(monkeypatch):
    _fake_llm_client(monkeypatch, '{"next": "bogus", "reason": "nope"}')

    g = graph_mod.build_graph()
    result = g.invoke(_initial(), config=_thread())

    assert result['next'] == 'report'
    assert result['final_report'] is not None


def test_happy_path_full_pipeline(monkeypatch):
    nexts = iter(['ingest', 'transcribe', 'visual', 'compose', 'render', 'ingest_kb', 'report'])

    def fake_planner(state):
        return {'next': next(nexts, 'report'), 'iterations': state.get('iterations', 0) + 1}

    monkeypatch.setattr(tools, 'tool_planner', fake_planner)
    monkeypatch.setattr(tools, 'tool_ingest', lambda s: {'video_path': '/tmp/video.mp4'})
    monkeypatch.setattr(
        tools, 'tool_transcribe', lambda s: {'transcript': 'hello world', 'transcript_language': 'en'}
    )
    monkeypatch.setattr(
        tools, 'tool_visual', lambda s: {'timeline': {'summary': 's', 'frames': [], 'warnings': []}}
    )
    monkeypatch.setattr(
        tools,
        'tool_compose',
        lambda s: {
            'document': {'title': 'T', 'doc_type': 'summary', 'warnings': []},
            'doc_spec': {'doc_type': 'summary', 'title': 'T'},
        },
    )
    monkeypatch.setattr(
        tools, 'tool_render', lambda s: {'markdown': '# T', 'pdf_path': '/tmp/out/document.pdf'}
    )
    monkeypatch.setattr(
        tools, 'tool_ingest_kb', lambda s: {'kb_doc_id': 'doc-1', 'kb_node_count': 5}
    )

    g = graph_mod.build_graph()
    result = g.invoke(_initial('happy'), config=_thread())

    assert result['final_report']['kb_doc_id'] == 'doc-1'
    assert result['final_report']['kb_node_count'] == 5
    assert result['markdown'] == '# T'


def test_transcribe_error_then_retry(monkeypatch):
    nexts = iter(['ingest', 'transcribe', 'transcribe', 'report'])

    def fake_planner(state):
        return {'next': next(nexts, 'report'), 'iterations': state.get('iterations', 0) + 1}

    calls = {'n': 0}

    def fake_transcribe(state):
        calls['n'] += 1
        if calls['n'] == 1:
            return {'errors': state.get('errors', []) + ['asr engine failure']}
        return {'transcript': 'hello world', 'transcript_language': 'en'}

    monkeypatch.setattr(tools, 'tool_planner', fake_planner)
    monkeypatch.setattr(tools, 'tool_ingest', lambda s: {'video_path': '/tmp/video.mp4'})
    monkeypatch.setattr(tools, 'tool_transcribe', fake_transcribe)

    g = graph_mod.build_graph()
    result = g.invoke(_initial('retry'), config=_thread())

    assert result['transcript'] == 'hello world'
    assert any('asr engine failure' in e for e in result['errors'])
    assert calls['n'] == 2


def test_loop_guard(monkeypatch):
    _fake_llm_client(monkeypatch, '{"next": "transcribe", "reason": "keep going"}')
    monkeypatch.setattr(tools, 'tool_transcribe', lambda s: {})

    g = graph_mod.build_graph()
    result = g.invoke(_initial('loop'), config=_thread())

    assert any('loop guard' in e for e in result['errors'])
    assert result['iterations'] <= 13


def test_json_safe_handles_message_like_objects():
    result = {'steps': [SimpleNamespace(content='hi')]}
    out = json.dumps(_json_safe(result), ensure_ascii=False)
    assert 'hi' in out


def test_planner_gives_up_after_two_failures(monkeypatch):
    _fake_llm_client(monkeypatch, '{"next": "transcribe", "reason": "retry"}')

    g = graph_mod.build_graph()
    state = _initial('planner-cap')
    state['errors'] = [
        'transcribe failed: asr funasr failed: x',
        'transcribe failed: asr whisper failed: y',
    ]
    result = g.invoke(state, config=_thread())

    assert result['next'] == 'report'
    assert result['final_report'] is not None
