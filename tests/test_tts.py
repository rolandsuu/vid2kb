from __future__ import annotations

import os
from pathlib import Path

import pytest

from vid2kb.agent import tools
from vid2kb.tts.cosyvoice import CHUNK_CHARS, synthesize_speech

FAKE_MP3 = b'ID3\x04\x00\x00\x00\x00\x00\x00' + b'\xff\xfb\x90\x00' * 100


def _fake_post_tts(payload, api_key):
    return FAKE_MP3


def test_synthesize_speech_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr('vid2kb.tts.cosyvoice._post_tts', _fake_post_tts)
    out = tmp_path / 'speech.mp3'
    result = synthesize_speech('你好世界', 'longxiaochun', out)
    assert result == out
    assert out.exists()
    assert out.read_bytes() == FAKE_MP3


def test_synthesize_speech_chunks_long_text(monkeypatch, tmp_path):
    calls = []

    def record(payload, api_key):
        calls.append(payload)
        return FAKE_MP3

    monkeypatch.setattr('vid2kb.tts.cosyvoice._post_tts', record)
    long_text = '字' * (CHUNK_CHARS * 2 + 10)
    out = synthesize_speech(long_text, 'longxiaochun', tmp_path / 'x.mp3')

    assert len(calls) == 3
    assert out.exists()
    assert out.read_bytes() == FAKE_MP3 * 3
    assert all(c['model'] == 'cosyvoice-v2' for c in calls)
    assert all(c['input']['format'] == 'mp3' for c in calls)


def test_synthesize_speech_raises_on_error(monkeypatch, tmp_path):
    def boom(payload, api_key):
        raise RuntimeError('cosyvoice tts failed: HTTP 400: invalid voice')

    monkeypatch.setattr('vid2kb.tts.cosyvoice._post_tts', boom)
    with pytest.raises(RuntimeError, match='invalid voice'):
        synthesize_speech('hi', 'nope', tmp_path / 'x.mp3')


def _fake_store_class(tmp_path):
    class FakeStore:
        def __init__(self, run_id):
            self.out = tmp_path / 'out'
            self.out.mkdir(parents=True, exist_ok=True)

    return FakeStore


def test_tool_voiceover_builds_files_and_state(monkeypatch, tmp_path):
    from vid2kb.media import store as store_mod

    monkeypatch.setattr(store_mod, 'ArtifactStore', _fake_store_class(tmp_path))

    def fake_synth(text, voice, out_path, model='cosyvoice-v1', sample_rate=48000):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(FAKE_MP3)
        return out_path

    monkeypatch.setattr('vid2kb.tts.cosyvoice.synthesize_speech', fake_synth)
    monkeypatch.setattr('vid2kb.media.ffmpeg.probe_duration', lambda p: 1.5)

    state = {
        'run_id': 'run-vo',
        'document': {
            'title': 'LangChain 入门',
            'summary': '介绍 LangChain 的核心概念。',
            'key_points': ['Agent', '工具调用'],
            'sections': [
                {'heading': '什么是 Agent', 'body_md': 'Agent 是自主推理的智能体。'},
                {'heading': '工具调用', 'body_md': '通过工具调用扩展能力。'},
            ],
        },
        'errors': [],
        'steps': [],
    }

    result = tools.tool_voiceover(state)

    assert 'voiceover' in result
    vo = result['voiceover']
    assert vo['total_seconds'] == 3 * 1.5
    assert len(vo['files']) == 3
    assert vo['files'][0]['path'].endswith('section_01.mp3')
    assert vo['files'][2]['path'].endswith('section_03.mp3')
    assert all(Path(f['path']).exists() for f in vo['files'])


def test_tool_voiceover_empty_document_returns_error(monkeypatch, tmp_path):
    from vid2kb.media import store as store_mod

    monkeypatch.setattr(store_mod, 'ArtifactStore', _fake_store_class(tmp_path))
    state = {'run_id': 'run-empty', 'document': {}, 'errors': [], 'steps': []}
    result = tools.tool_voiceover(state)
    assert result.get('errors')
    assert 'voiceover' not in result


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get('VID2KB_RUN_SMOKE'),
    reason='set VID2KB_RUN_SMOKE=1 to run a real DashScope TTS call',
)
def test_real_tts_smoke(tmp_path):
    out = synthesize_speech('你好，这是 vid2kb 的配音测试。', 'longxiaochun_v2', tmp_path / 'smoke.mp3')
    assert out.exists()
    assert out.stat().st_size > 1000
