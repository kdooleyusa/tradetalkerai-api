from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles

# NOTE:
# This file assumes Railway Root Directory is set to "app"
# and the start command is: uvicorn main:app --host 0.0.0.0 --port $PORT

from tts import generate_tts_mp3
from vision.pipeline import analyze_chart_image_bytes

app = FastAPI(title="TradeTalkerAI API")

# AUDIO_DIR must be absolute and consistent.
# NOTE: On Railway, local disk is ephemeral between deploys/restarts.
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./storage/audio")).expanduser().resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Optional: store uploaded charts for debugging (set SAVE_UPLOADS=1)
SAVE_UPLOADS = os.getenv("SAVE_UPLOADS", "0") == "1"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./storage/uploads")).expanduser().resolve()
if SAVE_UPLOADS:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Optional: Vision timeout (seconds). If Vision hangs, return a clean 500 instead of Railway 502.
VISION_TIMEOUT_S = float(os.getenv("VISION_TIMEOUT_S", "25"))

# Serve audio files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


@app.get("/")
def root():
    return {"status": "TradeTalkerAI API running"}


@app.get("/health")
def health():
    return {"ok": True}


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
    # Optional overrides (nice for Swagger testing)
    voice: str | None = Form(None),
    model: str | None = Form(None),
    # Accept as string so Swagger empty values don't 422
    speed: str | None = Form(None),
):
    """
    Phase 2: Vision + TTS
    - Upload chart screenshot
    - Extract ChartFacts via OpenAI Vision
    - Convert to a spoken transcript
    - Generate MP3 via OpenAI TTS
    """

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # Optional: validate basic content-type (best-effort)
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type: {image.content_type}")

    # Swagger defaults often come through as literal "string"
    if voice in ("", "string"):
        voice = None
    if model in ("", "string"):
        model = None

    # Parse speed safely (Swagger may send "")
    speed_f: float | None = None
    if speed not in (None, "", "string"):
        try:
            speed_f = float(speed)
            # treat <=0 as "not set"
            if speed_f <= 0:
                speed_f = None
        except ValueError:
            speed_f = None

    analysis_id = f"chart_{uuid.uuid4().hex[:8]}"

    # Optional: save upload for debugging
    if SAVE_UPLOADS:
        # choose extension based on content type if possible
        ext = ".png"
        if image.content_type == "image/jpeg":
            ext = ".jpg"
        elif image.content_type == "image/webp":
            ext = ".webp"
        (UPLOAD_DIR / f"{analysis_id}{ext}").write_bytes(raw)

    # 1) Vision -> ChartFacts -> transcript (with timeout to avoid gateway 502)
    try:
        chart_facts, transcript = await asyncio.wait_for(
            analyze_chart_image_bytes(raw, mode=mode),
            timeout=VISION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=500,
            detail=f"Vision failed: TimeoutError after {VISION_TIMEOUT_S:.0f}s",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vision failed: {type(e).__name__}: {e}",
        )

    # 2) TTS -> MP3
    try:
        mp3_path, audio_url = await generate_tts_mp3(
            transcript=transcript,
            analysis_id=analysis_id,
            out_dir=AUDIO_DIR,
            voice=voice,
            model=model,
            speed=speed_f,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS failed: {type(e).__name__}: {e}",
        )

    # Build a full URL you can click
    base = str(request.base_url).rstrip("/")
    audio_full_url = f"{base}{audio_url}"

    return {
        "mode": mode,
        "chart_facts": chart_facts.model_dump(),
        "transcript": transcript,
        "audio_url": audio_url,
        "audio_full_url": audio_full_url,
        "mp3_bytes": mp3_path.stat().st_size if mp3_path.exists() else 0,
        "audio_dir": str(AUDIO_DIR),
        "filename": mp3_path.name,
        "vision_timeout_s": VISION_TIMEOUT_S,
        "saved_upload": SAVE_UPLOADS,
    }
