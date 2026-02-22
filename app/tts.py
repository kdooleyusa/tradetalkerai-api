from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, Any

from openai import AsyncOpenAI

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]+")

# Voice map (1..6)
VOICE_MAP = {
    1: "onyx",   # M (default)
    2: "alloy",  # M
    3: "verse",  # M
    4: "echo",   # M
    5: "nova",   # F
    6: "marin",  # F
}


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




def _extract_usage_meta_from_resp(resp, *, model_fallback: str | None = None) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    model_name = getattr(resp, "model", None) or model_fallback
    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

def _resolve_voice(voice_id: int | None) -> str:
    env_voice = os.getenv("TTS_VOICE")
    if env_voice:
        return env_voice

    try:
        vid = int(voice_id or 1)
    except Exception:
        vid = 1

    return VOICE_MAP.get(vid, VOICE_MAP[1])


async def generate_tts_mp3(
    *,
    transcript: str,
    analysis_id: str,
    out_dir: Path,
    model: str | None = None,
    speed: float | None = None,
    voice: int | None = None,
    return_meta: bool = False,
) -> Tuple[Path, str] | Tuple[Path, str, dict[str, Any]]:
    """Generate MP3 TTS audio and save it locally.

    voice: integer 1..6
      1 onyx (default)
      2 alloy
      3 verse
      4 echo
      5 nova
      6 marin
    """

    if not transcript or not transcript.strip():
        raise ValueError("transcript is empty")

    model = model or os.getenv("TTS_MODEL", "gpt-4o-mini-tts")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_id(analysis_id)
    mp3_path = out_dir / f"{safe}.mp3"

    client = AsyncOpenAI()

    voice_name = _resolve_voice(voice)

    kwargs = {
        "model": model,
        "voice": voice_name,
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

    if return_meta:
        return mp3_path, f"/audio/{mp3_path.name}", _extract_usage_meta_from_resp(resp, model_fallback=model)

    return mp3_path, f"/audio/{mp3_path.name}"
