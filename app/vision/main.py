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
from vision.trade_logic import build_trade_plan, build_trade_transcript

app = FastAPI(title="TradeTalkerAI API")

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./storage/audio")).expanduser().resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CHART_DIR = Path(os.getenv("CHART_DIR", "./storage/charts")).expanduser().resolve()
CHART_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


def l2_is_meaningful(l2_comment: str | None) -> bool:
    """Scanner gating: only include L2 narration when it's high-signal."""
    if not l2_comment:
        return False
    low = l2_comment.lower().strip()

    # Drop generic/filler/errors
    drop = ("not visible", "timed out", "failed", "captured")
    if any(d in low for d in drop):
        return False

    allow = (
        "asks strengthening",
        "bids strengthening",
        "asks thinning",
        "bids thinning",
        "asks pulling",
        "bids pulling",
        "asks stacked",
        "bids stacked",
        "while bids add",
        "while asks add",
        "asks add",
        "bids add",
    )
    return any(a in low for a in allow)



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
    image2: UploadFile | None = File(None),  # optional 2nd frame (~0.5–1.5s later)
    mode: str = Form("brief"),
    model: str | None = Form(None),
    speed: str | None = Form(None),
    save_chart: bool = Form(False),
    frame_delay_ms: int | None = Form(None),
):
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

    # 1) Chart facts
    try:
        chart_facts, _legacy = await asyncio.wait_for(
            analyze_chart_image_bytes(raw1, mode=mode),
            timeout=float(os.getenv("VISION_TIMEOUT_SEC", "25")),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Chart vision timed out. Try a clearer/closer screenshot.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart vision failed: {type(e).__name__}: {e}")

    # 2) L2
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
        l2_comment = "Level 2 read timed out."
    except Exception as e:
        l2_comment = f"Level 2 read failed: {type(e).__name__}: {e}"

    # 3) TradePlan
    trade_plan = build_trade_plan(chart_facts, l2_1, l2_2, l2_delta)

    # 4) Transcript (now mode actually changes output)
    transcript = build_trade_transcript(chart_facts, trade_plan, l2_comment=(l2_comment if l2_is_meaningful(l2_comment) else None), mode=mode)

    keep_looking = bool(
        (chart_facts.confidence is not None and chart_facts.confidence < 0.55)
        or (chart_facts.setup in (None, "unclear", "range"))
        or (not chart_facts.symbol)
        or (not chart_facts.timeframe)
        or bool(trade_plan.step_aside)
        or (trade_plan.quality in ("D", "F"))
    )
    verdict = "keep_looking" if keep_looking else "actionable"



    # 5) TTS
    try:
        mp3_path, audio_url = await asyncio.wait_for(
            generate_tts_mp3(
                transcript=transcript,
                analysis_id=analysis_id,
                out_dir=Path(os.getenv("AUDIO_DIR", "./storage/audio")).expanduser().resolve(),
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
        "trade_plan": trade_plan.model_dump(),
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
