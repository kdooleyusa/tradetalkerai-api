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


def _uget(obj: Any, *names: str):
    for n in names:
        try:
            if obj is not None and hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
        except Exception:
            pass
    return None


def _usage_nested_token(usage: Any, detail_attr: str, key_names: tuple[str, ...]) -> Any:
    details = _uget(usage, detail_attr)
    if details is None:
        return None
    for k in key_names:
        try:
            if hasattr(details, k):
                v = getattr(details, k)
                if v is not None:
                    return v
        except Exception:
            pass
        try:
            if isinstance(details, dict) and k in details and details[k] is not None:
                return details[k]
        except Exception:
            pass
    return None


def _extract_usage_meta_from_resp(resp, *, model_fallback: str | None = None) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    prompt_tokens = _uget(usage, "prompt_tokens")
    completion_tokens = _uget(usage, "completion_tokens")
    total_tokens = _uget(usage, "total_tokens")
    model_name = getattr(resp, "model", None) or model_fallback

    tts_text_input_tokens = (
        _uget(usage, "text_input_tokens", "input_text_tokens")
        or _usage_nested_token(usage, "input_token_details", ("text_tokens", "input_text_tokens"))
    )
    tts_audio_output_tokens = (
        _uget(usage, "audio_output_tokens", "output_audio_tokens")
        or _usage_nested_token(usage, "output_token_details", ("audio_tokens", "output_audio_tokens"))
    )

    if model_name and "tts" in str(model_name).lower():
        if tts_text_input_tokens is None:
            tts_text_input_tokens = _uget(usage, "input_tokens", "prompt_tokens")
        if tts_audio_output_tokens is None:
            tts_audio_output_tokens = _uget(usage, "output_tokens", "completion_tokens")

    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tts_text_input_tokens": tts_text_input_tokens,
        "tts_audio_output_tokens": tts_audio_output_tokens,
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
