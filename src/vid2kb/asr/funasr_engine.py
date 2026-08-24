from __future__ import annotations

from pathlib import Path

from vid2kb.asr.base import ASREngine, Segment, Transcript


class FunASREngine(ASREngine):
    _model = None

    def _ensure_model(self):
        if self._model is None:
            from funasr import AutoModel
            self._model = AutoModel(model='iic/SenseVoiceSmall', device='cpu')
        return self._model

    def transcribe(self, audio: Path) -> Transcript:
        model = self._ensure_model()
        res = model.generate(input=str(audio), language='auto')
        text = res[0]['text']
        return Transcript(language='zh', segments=[Segment(0.0, 0.0, text)])
