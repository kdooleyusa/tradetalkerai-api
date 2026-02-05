from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
import os
import uuid
from pathlib import Path

from tts import generate_tts_mp3

app = FastAPI()

# Make sure AUDIO_DIR is ABSOLUTE and used consistently
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./storage/audio")).expanduser().resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Serve audio files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


@app.get("/")
def root():
    return {"status": "TradeTalkerAI API running"}


@app.get("/debug/audio")
def debug_audio():
    """List audio files currently present on the server."""
    files = []
    for p in sorted(AUDIO_DIR.glob("*.mp3")):
        files.append({"name": p.name, "bytes": p.stat().st_size})
    return {"audio_dir": str(AUDIO_DIR), "count": len(files), "files": files}


@app.post("/v1/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    mode: str = Form("brief"),
):
    # For now we use a fake transcript just to test TTS
    transcript = "Trade Talker A I is running. This is your test audio."

    analysis_id = f"test_{uuid.uuid4().hex[:8]}"

    mp3_path, audio_url = await generate_tts_mp3(
        transcript=transcript,
        analysis_id=analysis_id,
        out_dir=AUDIO_DIR,
    )

    # Build a full URL you can click
    base = str(request.base_url).rstrip("/")
    audio_full_url = f"{base}{audio_url}"

    return {
        "transcript": transcript,
        "audio_url": audio_url,
        "audio_full_url": audio_full_url,
        "mp3_bytes": mp3_path.stat().st_size if mp3_path.exists() else 0,
        "audio_dir": str(AUDIO_DIR),
    }
