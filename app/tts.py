from pathlib import Path
from openai import AsyncOpenAI
import re

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe_id(s: str) -> str:
    s = (s or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    return s[:80] if s else "unknown"


async def generate_tts_mp3(
    *,
    transcript: str,
    analysis_id: str,
    voice: str = "marin",
    speed: float = 1.0,
    out_dir: Path,
    model: str = "gpt-4o-mini-tts",
):
    """
    Generates MP3 via OpenAI TTS and saves locally.
    Returns (Path, url_path)
    """

    client = AsyncOpenAI()

    out_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_id(analysis_id)
    mp3_path = out_dir / f"{safe}.mp3"

    response = await client.audio.speech.create(
        model=model,
        voice=voice,
        input=transcript,
        format="mp3",
        speed=speed,
    )

    audio_bytes = response.read()

    mp3_path.write_bytes(audio_bytes)

    return mp3_path, f"/audio/{mp3_path.name}"
