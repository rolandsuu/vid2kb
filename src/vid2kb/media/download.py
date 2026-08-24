from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def download_video(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', str(dest_dir / 'source.%(ext)s'),
            url,
        ],
        check=True,
        timeout=1800,
    )
    return next(dest_dir.glob('source.*'))


def copy_local_file(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f'source{src.suffix}'
    return Path(shutil.copy2(src, dest))
