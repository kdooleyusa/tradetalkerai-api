from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

from openai import AsyncOpenAI

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _safe_id(s: str) -> str:
    s = (s or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    return s[:80] if s else "unknown"


def _response_to_bytes(resp) -> bytes:
    if hasattr(resp, "content") and resp.content:
        return resp.content
    try:
        return bytes(resp)
    except Exception:
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
    model: str | None = None,
    speed: float | None = None,
) -> Tuple[Path, str]:
    """Generate MP3 TTS audio and save it locally.

    Harold voice is forced to a male voice for now.
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript is empty")

    model = model or os.getenv("TTS_MODEL", "gpt-4o-mini-tts")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_id(analysis_id)
    mp3_path = out_dir / f"{safe}.mp3"

    client = AsyncOpenAI()

    # voice = "onyx"  # ED
    voice = "alloy"  # ED
    # voice = "verse"  # ED
    # voice = "echo"  # ED
    # voice = "nova"  # ED
    # voice = "marin"  # ED
    # voice = "verse"  # ED
    
    

    kwargs = {
        "model": model,
        "voice": voice,
        "input": transcript,
        "response_format": "mp3",
    }
    if speed is not None and speed > 0:
        kwargs["speed"] = speed

    resp = await client.audio.speech.create(**kwargs)

    audio_bytes = _response_to_bytes(resp)
    mp3_path.write_bytes(audio_bytes)

    if mp3_path.stat().st_size == 0:
        raise RuntimeError("TTS generated a 0-byte file (unexpected).")

    return mp3_path, f"/audio/{mp3_path.name}"
