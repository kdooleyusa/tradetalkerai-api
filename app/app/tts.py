from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

from openai import AsyncOpenAI

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe_id(s: str) -> str:
    s = (s or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    return s[:80] if s else "an_unknown"


async def generate_tts_mp3(
    *,
    transcript: str,
    analysis_id: str,
    voice: str = "marin",
    speed: float = 1.1,
    instructions: Optional[str] = None,
    out_dir: str | Path = "./storage/audio",
    model: str = "gpt-4o-mini-tts",
) -> Tuple[Path, str]:
    """
    Generates an MP3 from transcript and saves it locally.
    Returns: (absolute_file_path, url_path)
    """
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("Transcript is empty")

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    safe_analysis_id = _safe_id(analysis_id)
    mp3_path = (out_path / f"{safe_analysis_id}.mp3").resolve()

    client = AsyncOpenAI()

    async with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=transcript,
        instructions=instructions or "",
        response_format="mp3",
        speed=speed,
    ) as response:
        response.stream_to_file(mp3_path)

    url_path = f"/audio/{mp3_path.name}"
    return mp3_path, url_path
