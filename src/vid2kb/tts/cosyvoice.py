from __future__ import annotations

from pathlib import Path

TTS_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer'

CHUNK_CHARS = 500


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = (text or '').strip()
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _request_tts(payload: dict, api_key: str) -> dict:
    import httpx

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    resp = httpx.post(TTS_API_URL, json=payload, headers=headers, timeout=120.0)
    if resp.status_code != 200:
        raise RuntimeError(
            f'cosyvoice tts failed: HTTP {resp.status_code}: {resp.text[:1000]}'
        )
    return resp.json()


def _download(url: str) -> bytes:
    import httpx

    resp = httpx.get(url, timeout=120.0)
    if resp.status_code != 200:
        raise RuntimeError(f'failed to download audio: HTTP {resp.status_code}')
    return resp.content


def _post_tts(payload: dict, api_key: str) -> bytes:
    data = _request_tts(payload, api_key)
    url = (data.get('output', {}) or {}).get('audio', {}).get('url', '')
    if not url:
        raise RuntimeError(f'cosyvoice tts returned no audio url: {str(data)[:500]}')
    return _download(url)


def synthesize_speech(
    text: str,
    voice: str,
    out_path: str | Path,
    model: str = 'cosyvoice-v2',
    format: str = 'mp3',
    sample_rate: int = 48000,
    chunk_size: int = CHUNK_CHARS,
) -> Path:
    """Synthesize ``text`` to audio at ``out_path`` via the DashScope CosyVoice API.

    Long text is split client-side into ~``chunk_size``-char requests whose
    audio is concatenated into a single file. Raises ``RuntimeError`` (with the
    response body) on any non-200 response.
    """
    from vid2kb.config import settings

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[bytes] = []
    for chunk in _chunks(text, chunk_size):
        payload = {
            'model': model,
            'input': {
                'text': chunk,
                'voice': voice,
                'format': format,
                'sample_rate': sample_rate,
            },
        }
        parts.append(_post_tts(payload, settings.dashscope_api_key))

    if not parts:
        raise ValueError('no text to synthesize')

    out_path.write_bytes(b''.join(parts))
    return out_path


def audio_duration_seconds(audio: Path) -> float:
    from vid2kb.media.ffmpeg import probe_duration

    return probe_duration(audio)
