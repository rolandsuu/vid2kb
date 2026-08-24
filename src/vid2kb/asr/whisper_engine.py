from __future__ import annotations

from pathlib import Path

from vid2kb.asr.base import ASREngine, Segment, Transcript


class WhisperEngine(ASREngine):
    def __init__(self, model_size: str = 'small', device: str = 'cpu') -> None:
        self.model_size = model_size
        self.device = device

    def _model_factory(self):
        from faster_whisper import WhisperModel
        return WhisperModel(self.model_size, device=self.device)

    def transcribe(self, audio: Path) -> Transcript:
        segments_iter, info = self._model_factory().transcribe(str(audio), word_timestamps=False)
        segments = [
            Segment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments_iter
            if seg.text.strip()
        ]
        return Transcript(language=info.language, segments=segments)
