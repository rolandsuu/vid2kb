from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vid2kb.vision.qwen_vl import analyze_frames


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeClient:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def __getattr__(self, name):
        return self

    def create(self, **kwargs):
        self.calls += 1
        content = self._contents.pop(0)
        return _resp(content)


class RaisingClient:
    def __init__(self, fail_on_call: int):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __getattr__(self, name):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError('boom')
        return _resp(
            '{"summary": "s", "frames": ['
            '{"index": 0, "timestamp_seconds": 0.0, "description": "a", "confidence": 0.5}, '
            '{"index": 1, "timestamp_seconds": 2.0, "description": "b", "confidence": 0.5}'
            '], "warnings": []}'
        )


@pytest.fixture
def dummy_frames(tmp_path):
    frames = []
    for i in range(3):
        p = tmp_path / f'frame_{i:04d}.jpg'
        p.write_bytes(b'\xff\xd8\xff\xe0fake')
        frames.append((i, float(i * 2), p))
    return frames


def test_analyze_frames_merges_batches(dummy_frames, monkeypatch):
    batch1 = (
        '{"summary": "merged", "frames": ['
        '{"index": 0, "timestamp_seconds": 0.0, "description": "a", "confidence": 0.9}, '
        '{"index": 1, "timestamp_seconds": 2.0, "description": "b", "confidence": 0.8}'
        '], "warnings": ["warn a"]}'
    )
    batch2 = (
        '{"summary": "merged", "frames": ['
        '{"index": 2, "timestamp_seconds": 4.0, "description": "c", "confidence": 0.7}'
        '], "warnings": ["warn b"]}'
    )
    fake = FakeClient([batch1, batch2])
    monkeypatch.setattr('vid2kb.llm.dashscope_client', lambda: fake)
    monkeypatch.setattr('vid2kb.vision.qwen_vl.BATCH_SIZE', 2)

    result = analyze_frames(dummy_frames, 'describe these', 'hello transcript')

    assert result.summary == 'merged'
    assert len(result.frames) == 3
    assert [f.index for f in result.frames] == [0, 1, 2]
    assert [f.timestamp_seconds for f in result.frames] == [0.0, 2.0, 4.0]
    assert result.warnings == ['warn a', 'warn b']


def test_analyze_frames_failed_batch_records_warning(dummy_frames, monkeypatch):
    fake = RaisingClient(fail_on_call=2)
    monkeypatch.setattr('vid2kb.llm.dashscope_client', lambda: fake)
    monkeypatch.setattr('vid2kb.vision.qwen_vl.BATCH_SIZE', 2)

    result = analyze_frames(dummy_frames, 'describe these', 'hello transcript')

    assert len(result.frames) == 2
    assert any('failed' in w for w in result.warnings)


def test_analyze_frames_all_fail_raises(dummy_frames, monkeypatch):
    fake = RaisingClient(fail_on_call=1)
    monkeypatch.setattr('vid2kb.llm.dashscope_client', lambda: fake)

    with pytest.raises(RuntimeError, match='vision analysis failed'):
        analyze_frames(dummy_frames, 'describe these', 'hello transcript')


def test_parse_json_handles_string_lists(monkeypatch):
    """Qwen-VL returns visible_text/actions as '' strings; must coerce to lists."""
    from vid2kb.vision.qwen_vl import _build_timeline
    import tempfile
    p = Path(tempfile.mkdtemp()) / 'f.jpg'
    p.write_bytes(b'x')
    obj = {
        'summary': 's',
        'frames': [{
            'index': 0, 'timestamp_seconds': 0.0, 'description': 'd',
            'visible_text': '', 'actions': 'something', 'confidence': '0.9',
        }],
        'warnings': [],
    }
    tl = _build_timeline(obj, [(0, 0.0, p)])
    assert tl.frames[0].visible_text == []
    assert tl.frames[0].actions == ['something']
    assert tl.frames[0].confidence == 0.9


def test_analyze_frames_single_description_fallback(dummy_frames, monkeypatch):
    fake = FakeClient(['{"description": "测试画面"}'])
    monkeypatch.setattr('vid2kb.llm.dashscope_client', lambda: fake)

    result = analyze_frames(dummy_frames, 'describe these', 'hello transcript')

    assert len(result.frames) == 1
    assert result.frames[0].description == '测试画面'
    assert any('single description' in w for w in result.warnings)


def test_analyze_frames_json_fence_stripped(dummy_frames, monkeypatch):
    content = (
        '```json\n'
        '{"summary": "s", "frames": [{"index": 0, "timestamp_seconds": 0.0, '
        '"description": "d", "visible_text": [], "actions": [], "confidence": 0.9}], '
        '"warnings": []}\n'
        '```'
    )
    fake = FakeClient([content])
    monkeypatch.setattr('vid2kb.llm.dashscope_client', lambda: fake)

    result = analyze_frames(dummy_frames, 'describe these', 'hello transcript')

    assert len(result.frames) == 1
    assert result.frames[0].description == 'd'
    assert result.summary == 's'
