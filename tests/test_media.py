from __future__ import annotations

from pathlib import Path

from vid2kb.media.ffmpeg import extract_audio, probe_duration, sample_frames


def test_store_creates_dirs(store):
    assert store.root.exists()
    assert store.raw.exists()
    assert store.audio.exists()
    assert store.frames.exists()
    assert store.out.exists()


def test_store_cleanup(store):
    store.cleanup()
    assert not store.root.exists()


def test_probe_duration(make_test_video):
    d = probe_duration(make_test_video)
    assert 5.0 < d < 7.0


def test_extract_audio(make_test_video, tmp_path):
    wav = extract_audio(make_test_video, tmp_path / 'audio.wav')
    assert wav.exists()
    assert wav.stat().st_size > 1000


def test_sample_frames(make_test_video, tmp_path):
    dest = tmp_path / 'frames'
    dest.mkdir()
    frames = sample_frames(make_test_video, dest, interval_s=2, max_frames=5)
    assert len(frames) <= 5
    assert len(frames) in (3, 4)
    for i, ts, p in frames:
        assert ts == i * 2
        assert ts % 2 == 0
        assert p.exists()
