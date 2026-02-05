from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles

# Root Directory = "app" on Railway
# Start command example:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
from tts import generate_tts_mp3
from vision.pipeline import analyze_chart_image_bytes

app = FastAPI(title="TradeTalkerAI API")

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./storage/audio")).expanduser().resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CHART_DIR = Path(os.getenv("CHART_DIR", "./storage/charts")).expanduser().resolve()
CHART_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


@app.get("/")
def root():
    return {"status": "TradeTalkerAI API running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/audio")
def debug_audio():
    files = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(AUDIO_DIR.glob("*.mp3"))]
    return {"audio_dir": str(AUDIO_DIR), "count": len(files), "files": files}


@app.get("/debug/charts")
def debug_charts():
    files = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(CHART_DIR.glob("*.*"))]
    return {"chart_dir": str(CHART_DIR), "count": len(files), "files": files}


@app.post("/v1/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    mode: str = Form("brief"),
    model: str | None = Form(None),
    speed: str | None = Form(None),
    save_chart: bool = Form(False),
):
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type: {image.content_type}")

    if model in ("", "string"):
        model = None

    speed_f: float | None = None
    if speed not in (None, "", "string"):
        try:
            speed_f = float(speed)
            if speed_f <= 0:
                speed_f = None
        except ValueError:
            speed_f = None

    analysis_id = f"chart_{uuid.uuid4().hex[:8]}"

    chart_filename = None
    if save_chart:
        ext = "png"
        if image.filename and "." in image.filename:
            ext = image.filename.rsplit(".", 1)[-1].lower()[:6] or "png"
        chart_filename = f"{analysis_id}.{ext}"
        (CHART_DIR / chart_filename).write_bytes(raw)

    try:
        chart_facts, transcript = await asyncio.wait_for(
            analyze_chart_image_bytes(raw, mode=mode),
            timeout=float(os.getenv("VISION_TIMEOUT_SEC", "25")),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Vision timed out. Try a clearer/closer screenshot.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision failed: {type(e).__name__}: {e}")

    keep_looking = bool(
        (chart_facts.confidence is not None and chart_facts.confidence < 0.55)
        or (chart_facts.setup in (None, "unclear"))
        or (not chart_facts.symbol)
        or (not chart_facts.timeframe)
    )
    verdict = "keep_looking" if keep_looking else "actionable"

    try:
        mp3_path, audio_url = await asyncio.wait_for(
            generate_tts_mp3(
                transcript=transcript,
                analysis_id=analysis_id,
                out_dir=AUDIO_DIR,
                model=model,
                speed=speed_f,
            ),
            timeout=float(os.getenv("TTS_TIMEOUT_SEC", "25")),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="TTS timed out. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {type(e).__name__}: {e}")

    base = str(request.base_url).rstrip("/")
    audio_full_url = f"{base}{audio_url}"

    return {
        "mode": mode,
        "verdict": verdict,
        "keep_looking": keep_looking,
        "confidence": chart_facts.confidence,
        "chart_facts": chart_facts.model_dump(),
        "transcript": transcript,
        "audio_url": audio_url,
        "audio_full_url": audio_full_url,
        "mp3_bytes": mp3_path.stat().st_size if mp3_path.exists() else 0,
        "audio_dir": str(AUDIO_DIR),
        "filename": mp3_path.name,
        "saved_chart": chart_filename,
        "chart_dir": str(CHART_DIR),
        "vision_timeout_s": float(os.getenv("VISION_TIMEOUT_SEC", "25")),
    }
