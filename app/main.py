from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles

from tts import generate_tts_mp3
from vision.pipeline import analyze_chart_image_bytes
from vision.l2_pipeline import analyze_l2_image_bytes, compute_l2_delta, build_l2_commentary

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


@app.post("/v1/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    # Optional 2nd frame (send ~0.5s–1.5s later)
    image2: UploadFile | None = File(None),
    mode: str = Form("brief"),
    model: str | None = Form(None),
    speed: str | None = Form(None),
    # If True, save uploads in /storage/charts for debugging
    save_chart: bool = Form(False),
    # Optional metadata from helper app
    frame_delay_ms: int | None = Form(None),
):
    """
    Phase 2+: Vision + 2-frame Level2 snapshot inference + TTS

    Send:
      - image  (required)
      - image2 (optional; second screenshot ~500-1500ms later)
    """
    raw1 = await image.read()
    if not raw1:
        raise HTTPException(status_code=400, detail="Empty image upload (image)")

    raw2: bytes | None = None
    if image2 is not None:
        raw2 = await image2.read()
        if raw2 == b"":
            raw2 = None

    if model in ("", "string"):
        model = None

    # Parse speed safely (Swagger may send "")
    speed_f: float | None = None
    if speed not in (None, "", "string"):
        try:
            speed_f = float(speed)
            if speed_f <= 0:
                speed_f = None
        except ValueError:
            speed_f = None

    analysis_id = f"chart_{uuid.uuid4().hex[:8]}"

    saved = {"image": None, "image2": None}
    if save_chart:
        ext1 = "png"
        if image.filename and "." in image.filename:
            ext1 = image.filename.rsplit(".", 1)[-1].lower()[:6] or "png"
        fn1 = f"{analysis_id}_a.{ext1}"
        (CHART_DIR / fn1).write_bytes(raw1)
        saved["image"] = fn1

        if raw2 is not None:
            ext2 = "png"
            if image2 and image2.filename and "." in image2.filename:
                ext2 = image2.filename.rsplit(".", 1)[-1].lower()[:6] or "png"
            fn2 = f"{analysis_id}_b.{ext2}"
            (CHART_DIR / fn2).write_bytes(raw2)
            saved["image2"] = fn2

    # ---- 1) Chart facts from frame 1 ----
    try:
        chart_facts, transcript_core = await asyncio.wait_for(
            analyze_chart_image_bytes(raw1, mode=mode),
            timeout=float(os.getenv("VISION_TIMEOUT_SEC", "25")),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Chart vision timed out. Try a clearer/closer screenshot.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart vision failed: {type(e).__name__}: {e}")

    # ---- 2) Level 2 snapshot from frame 1 (and frame 2 if provided) ----
    l2_1 = None
    l2_2 = None
    l2_delta = None
    l2_comment = None

    try:
        l2_1 = await asyncio.wait_for(
            analyze_l2_image_bytes(raw1),
            timeout=float(os.getenv("L2_TIMEOUT_SEC", "25")),
        )
        if raw2 is not None:
            l2_2 = await asyncio.wait_for(
                analyze_l2_image_bytes(raw2),
                timeout=float(os.getenv("L2_TIMEOUT_SEC", "25")),
            )
            l2_delta = compute_l2_delta(l2_1, l2_2)
            l2_comment = build_l2_commentary(l2_1, l2_2, l2_delta)
        else:
            l2_comment = build_l2_commentary(l2_1, None, None)
    except asyncio.TimeoutError:
        # Don't fail the whole request — just omit L2
        l2_comment = "Level 2 read timed out."
    except Exception as e:
        l2_comment = f"Level 2 read failed: {type(e).__name__}: {e}"

    # ---- 3) Merge transcript ----
    transcript = transcript_core
    if l2_comment:
        transcript = transcript_core + " " + l2_comment

    keep_looking = bool(
        (chart_facts.confidence is not None and chart_facts.confidence < 0.55)
        or (chart_facts.setup in (None, "unclear"))
        or (not chart_facts.symbol)
        or (not chart_facts.timeframe)
    )
    verdict = "keep_looking" if keep_looking else "actionable"

    # ---- 4) TTS ----
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
        "l2_frame1": l2_1.model_dump() if l2_1 else None,
        "l2_frame2": l2_2.model_dump() if l2_2 else None,
        "l2_delta": l2_delta.model_dump() if l2_delta else None,
        "l2_comment": l2_comment,
        "transcript": transcript,
        "audio_url": audio_url,
        "audio_full_url": audio_full_url,
        "mp3_bytes": mp3_path.stat().st_size if mp3_path.exists() else 0,
        "audio_dir": str(AUDIO_DIR),
        "filename": mp3_path.name,
        "saved_upload": saved,
        "chart_dir": str(CHART_DIR),
        "vision_timeout_s": float(os.getenv("VISION_TIMEOUT_SEC", "25")),
        "l2_timeout_s": float(os.getenv("L2_TIMEOUT_SEC", "25")),
        "frame_delay_ms": frame_delay_ms,
    }
