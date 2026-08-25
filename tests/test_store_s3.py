from __future__ import annotations

from pathlib import Path

import pytest
from moto import mock_aws

from vid2kb.config import settings
from vid2kb.media.store import ArtifactStore


@pytest.fixture
def s3_mode():
    original_endpoint = settings.s3_endpoint_url
    object.__setattr__(settings, 's3_endpoint_url', '')
    yield
    object.__setattr__(settings, 's3_endpoint_url', original_endpoint)


@mock_aws
def test_s3_put_get_list_roundtrip(tmp_path, s3_mode):
    store = ArtifactStore('run-s3')
    src = tmp_path / 'document.md'
    src.write_text('# hello world', encoding='utf-8')

    key = store.put_file('out/document.md', src)
    assert key == 'runs/run-s3/out/document.md'
    assert store.exists('out/document.md')

    dest = tmp_path / 'downloaded.md'
    got = store.get_file('out/document.md', dest)
    assert got.read_text(encoding='utf-8') == '# hello world'

    keys = store.list_artifacts()
    assert 'runs/run-s3/out/document.md' in keys


@mock_aws
def test_s3_exists_missing(tmp_path, s3_mode):
    store = ArtifactStore('run-s3')
    assert not store.exists('out/nope.md')
    assert store.list_objects('out/') == []


@mock_aws
def test_s3_nested_prefix(tmp_path, s3_mode):
    store = ArtifactStore('run-s3')
    src = tmp_path / 'frame.png'
    src.write_bytes(b'\x00\x01\x02')
    store.put_file('frames/000.png', src)
    store.put_file('out/document.md', src)

    keys = store.list_objects('frames/')
    assert 'runs/run-s3/frames/000.png' in keys
    assert 'runs/run-s3/out/document.md' not in keys
