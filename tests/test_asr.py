from __future__ import annotations

from pathlib import Path

import pytest

from vid2kb.asr.base import ASREngine, Segment, Transcript
from vid2kb.asr.funasr_engine import FunASREngine
from vid2kb.asr.whisper_engine import WhisperEngine


def test_transcript_text_joins_segments():
    t = Transcript(
        language='en',
        segments=[
            Segment(0.0, 1.0, 'hello'),
            Segment(1.0, 2.0, 'world'),
        ],
    )
    assert t.text == 'hello\nworld'


def test_whisper_engine_mocked():
    class FakeModel:
        def transcribe(self, audio, word_timestamps=False):
            return {
                'language': 'en',
                'segments': [
                    {'start': 0.0, 'end': 1.0, 'text': 'hello'},
                    {'start': 1.0, 'end': 2.0, 'text': ''},
                ],
            }

    class FakeWhisper(WhisperEngine):
        def _model_factory(self):
            return FakeModel()

    tr = FakeWhisper().transcribe(Path('x.wav'))
    assert tr.language == 'en'
    assert len(tr.segments) == 1
    assert tr.text == 'hello'


def test_funasr_engine_mocked():
    class FakeModel:
        def generate(self, input, language='auto'):
            return [{'text': '你好'}]

    class FakeFunASR(FunASREngine):
        def _ensure_model(self):
            return FakeModel()

    tr = FakeFunASR().transcribe(Path('x.wav'))
    assert tr.text == '你好'


def test_engine_base_raises():
    with pytest.raises(NotImplementedError):
        ASREngine().transcribe(Path('x'))
