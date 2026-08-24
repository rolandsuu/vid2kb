from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vid2kb.media.store import ArtifactStore


@pytest.fixture
def make_test_video(tmp_path):
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg not found')
    out = tmp_path / 'out.mp4'
    subprocess.run(
        [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=6:size=320x240:rate=10',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture
def store(run_id='test-run'):
    return ArtifactStore(run_id=run_id)
