from __future__ import annotations

from pathlib import Path

import vid2kb.media.download as dl


def test_download_video_command(monkeypatch, tmp_path):
    dest = tmp_path / 'downloads'
    captured = {}

    def fake_run(args, **kwargs):
        captured['args'] = list(args)
        captured['kwargs'] = kwargs
        (dest / 'source.mp4').write_bytes(b'x')

    monkeypatch.setattr(dl.subprocess, 'run', fake_run)

    result = dl.download_video('https://example.com/video.mp4', dest)

    args = captured['args']
    assert 'yt-dlp' in args
    assert args[0] == 'yt-dlp'
    o_idx = args.index('-o')
    template = args[o_idx + 1]
    assert 'source' in template
    assert captured['kwargs']['check'] is True
    assert captured['kwargs']['timeout'] == 1800
    assert result == dest / 'source.mp4'


def test_copy_local_file(tmp_path):
    src = tmp_path / 'clip.mov'
    src.write_bytes(b'video-bytes')
    dest_dir = tmp_path / 'staging'
    result = dl.copy_local_file(src, dest_dir)
    assert result == dest_dir / 'source.mov'
    assert result.exists()
    assert result.read_bytes() == b'video-bytes'
