from __future__ import annotations

import shutil
from pathlib import Path

from vid2kb.config import settings


class ArtifactStore:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.root = Path(settings.data_dir) / 'runs' / run_id
        self.raw = self.root / 'raw'
        self.audio = self.root / 'audio'
        self.frames = self.root / 'frames'
        self.out = self.root / 'out'
        for d in (self.raw, self.audio, self.frames, self.out):
            d.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def _prefix(self) -> str:
        return f'runs/{self.run_id}/'

    def _s3_client(self):
        if self._client is None:
            import boto3

            kwargs = {
                'aws_access_key_id': settings.s3_access_key,
                'aws_secret_access_key': settings.s3_secret_key,
                'region_name': settings.s3_region,
            }
            if settings.s3_endpoint_url:
                kwargs['endpoint_url'] = settings.s3_endpoint_url
            self._client = boto3.client('s3', **kwargs)
        return self._client

    def _ensure_bucket(self) -> None:
        client = self._s3_client()
        try:
            client.head_bucket(Bucket=settings.s3_bucket)
        except Exception:
            client.create_bucket(Bucket=settings.s3_bucket)

    def put_file(self, key: str, path: str | Path) -> str:
        self._ensure_bucket()
        full_key = f'{self._prefix}{key}'
        self._s3_client().upload_file(str(path), settings.s3_bucket, full_key)
        return full_key

    def get_file(self, key: str, dest: str | Path) -> Path:
        full_key = f'{self._prefix}{key}'
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3_client().download_file(settings.s3_bucket, full_key, str(dest))
        return dest

    def exists(self, key: str) -> bool:
        client = self._s3_client()
        try:
            client.head_object(Bucket=settings.s3_bucket, Key=f'{self._prefix}{key}')
            return True
        except Exception:
            return False

    def list_objects(self, prefix: str = '') -> list[str]:
        self._ensure_bucket()
        client = self._s3_client()
        keys: list[str] = []
        paginator = client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=f'{self._prefix}{prefix}'):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys

    def list_artifacts(self) -> list[str]:
        return self.list_objects('out/')

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
