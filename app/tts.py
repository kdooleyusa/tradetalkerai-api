from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, Optional

from openai import AsyncOpenAI

# Keep filenames safe and short
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _safe_id(s: str) -> str:
    s = (s or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    return s[:80] if s else "unknown"


def _response_to_bytes(resp) -> bytes:
    """
    Extract audio bytes from OpenAI SDK response objects across versions.

    In openai>=1.x, audio responses often provide a .read() method.
    Some variants provide .content or are bytes-like.
    """
    # Preferred: file-like read()
    if hasattr(resp, "read") and callable(resp.read):
        data = resp.read()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    # Common: .content
    if hasattr(resp, "content") and resp.content:
        if isinstance(resp.content, (bytes, bytearray)):
            return bytes(resp.content)

    # Bytes-like fallback
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
    voice: Optional[str] = None,
    model: Optional[str] = None,
    speed: Optional[float] = None,
) -> Tuple[Path, str]:
    """
    Generate TTS audio and save it locally.
    Returns: (mp3_path, audio_url_path)

    Uses OpenAI TTS. Avoids the unsupported `format=` argument.
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript is empty")

    # Allow overriding via env vars
    voice = voice or os.getenv("TTS_VOICE", "marin")
    model = model or os.getenv("TTS_MODEL", "gpt-4o-mini-tts")

    # Speed is optional; only pass it if set (some SDK/model combos may be strict)
    if speed is None:
        env_speed = os.getenv("TTS_SPEED", "").strip()
        speed = float(env_speed) if env_speed else None

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_id(analysis_id)
    mp3_path = out_dir / f"{safe}.mp3"

    client = AsyncOpenAI()

    # Build args carefully to avoid "unexpected keyword" issues
    create_kwargs = {
        "model": model,
        "voice": voice,
        "input": transcript,
        "response_format": "mp3",
    }
    if speed is not None:
        create_kwargs["speed"] = speed

    resp = await client.audio.speech.create(**create_kwargs)

    audio_bytes = _response_to_bytes(resp)

    # Write to disk
    mp3_path.write_bytes(audio_bytes)

    # Sanity check
    if mp3_path.stat().st_size == 0:
        raise RuntimeError("TTS generated a 0-byte file (unexpected).")

    return mp3_path, f"/audio/{mp3_path.name}"
