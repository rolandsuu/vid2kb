from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return chr(10).join(s.text for s in self.segments)


class ASREngine:
    def transcribe(self, audio: Path) -> Transcript:
        raise NotImplementedError
