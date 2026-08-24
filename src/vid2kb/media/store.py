from __future__ import annotations

import shutil
from pathlib import Path

from vid2kb.config import settings


class ArtifactStore:
    def __init__(self, run_id: str) -> None:
        self.root = Path(settings.data_dir) / 'runs' / run_id
        self.raw = self.root / 'raw'
        self.audio = self.root / 'audio'
        self.frames = self.root / 'frames'
        self.out = self.root / 'out'
        for d in (self.raw, self.audio, self.frames, self.out):
            d.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
