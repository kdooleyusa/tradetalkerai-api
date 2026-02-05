from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

from openai import AsyncOpenAI

# Keep filenames safe and short
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _safe_id(s: str) -> str:
    s = (s or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    return s[:80] if s else "unknown"


def _response_to_bytes(resp) -> bytes:
    """
    OpenAI SDK responses vary by version:
    - some have .content (bytes)
    - some are bytes-like and support bytes(resp)
    """
    if hasattr(resp, "content") and resp.content:
        return resp.content
    try:
        return bytes(resp)
    except Exception:
        # Last resort: try common attribute names
        for attr in ("data", "audio", "bytes"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val)
        raise RuntimeError("Could not extract audio bytes from TTS response.")


async def generate_tts_mp3(
    *,
    transcript: str,
    analysis_id: str,
    out_dir: Path,
    voice: str | None = None,
    model: str | None = None,
) -> Tuple[Path, str]:
    """
    Generate TTS audio and save it locally.
    Returns: (mp3_path, audio_url_path)

    NOTE: Your current OpenAI SDK on Railway does NOT accept `format=...`,
    so we omit it. Most deployments return MP3 by default; we save as .mp3.
    """

    if not transcript or not transcript.strip():
        raise ValueError("transcript is empty")

    # Allow overriding via env vars
    voice = voice or os.getenv("TTS_VOICE", "marin")
    model = model or os.getenv("TTS_MODEL", "gpt-4o-mini-tts")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_id(analysis_id)
    mp3_path = out_dir / f"{safe}.mp3"

    client = AsyncOpenAI()

    # IMPORTANT: no `format=` arg (your SDK errors if provided)
    response = await client.audio.speech.create(
    model=model,
    voice=voice,
    input=text,
    response_format="mp3",   # <-- use this
    speed=speed,
)


    audio_bytes = _response_to_bytes(resp)

    # Write to disk
    mp3_path.write_bytes(audio_bytes)

    # Sanity check
    if mp3_path.stat().st_size == 0:
        raise RuntimeError("TTS generated a 0-byte file (unexpected).")

    return mp3_path, f"/audio/{mp3_path.name}"
