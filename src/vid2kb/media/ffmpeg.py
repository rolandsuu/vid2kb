from __future__ import annotations

import subprocess
from pathlib import Path


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(video)],
        check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def extract_audio(video: Path, dest: Path) -> Path:
    subprocess.run(['ffmpeg', '-y', '-i', str(video), '-vn',
                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    str(dest)], check=True, capture_output=True)
    return dest


def sample_frames(video: Path, dest_dir: Path, interval_s: int,
                  max_frames: int) -> list[tuple[int, float, Path]]:
    subprocess.run(['ffmpeg', '-y', '-i', str(video),
                    '-vf', f'fps=1/{interval_s},scale=1280:-2',
                    '-q:v', '2', str(dest_dir / 'frame_%04d.jpg')],
                   check=True, capture_output=True)
    frames = sorted(dest_dir.glob('frame_*.jpg'))[:max_frames]
    return [(i, i * interval_s, p) for i, p in enumerate(frames)]


def cut_clip(video: Path, start_s: float, end_s: float, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['ffmpeg', '-y', '-ss', str(start_s), '-to', str(end_s),
         '-i', str(video), '-c', 'copy', str(dest)],
        check=True, capture_output=True)
    return dest
