from __future__ import annotations

import asyncio
import os
import uuid
import time
import secrets
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Header
from fastapi.staticfiles import StaticFiles

from tts import generate_tts_mp3
from vision.pipeline import analyze_chart_image_bytes
from vision.l2_pipeline import analyze_l2_image_bytes, compute_l2_delta, build_l2_commentary
from vision.trade_logic import build_trade_plan, build_trade_transcript
from vision.news import fetch_finviz_news
from vision.full_mode import build_full_transcript_llm

app = FastAPI(title="TradeTalkerAI API")


# --- Postgres subscriber gate + usage logging (psycopg3 async) ---
_DB_POOL: AsyncConnectionPool | None = None

def _new_request_id() -> str:
    return secrets.token_hex(12)

def _get_header_int(request: Request, key: str, default: int = 0) -> int:
    try:
        v = request.headers.get(key)
        return int(v) if v is not None and v != "" else default
    except Exception:
        return default

async def _db_pool() -> AsyncConnectionPool:
    global _DB_POOL
    if _DB_POOL is None:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL is not set")
        _DB_POOL = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=5, timeout=5, kwargs={'autocommit': True})
        await _DB_POOL.open()
    return _DB_POOL

@app.on_event("shutdown")
async def _close_db_pool():
    global _DB_POOL
    if _DB_POOL is not None:
        try:
            await _DB_POOL.close()
        except Exception:
            pass
        _DB_POOL = None

async def _check_subscriber(subscriber_id: str) -> tuple[bool, str]:
    """Returns (allowed, reason). reason: ALLOW | SUBSCRIPTION_REQUIRED | SUBSCRIPTION_DISABLED | DB_ERROR"""
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT enabled FROM subscribers WHERE subscriber_id = %s", (subscriber_id,))
                row = await cur.fetchone()
    except Exception:
        return False, "DB_ERROR"

    if not row:
        return False, "SUBSCRIPTION_REQUIRED"
    enabled = bool(row[0])
    if not enabled:
        return False, "SUBSCRIPTION_DISABLED"
    return True, "ALLOW"

async def _log_entitlement(
    request_id: str,
    subscriber_id: str | None,
    device_id: str | None,
    endpoint: str,
    decision: str,   # ALLOW | DENY | ERROR
    reason: str,
    ip: str | None,
    user_agent: str | None,
) -> None:
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO entitlement_events
                    (request_id, subscriber_id, device_id, endpoint, decision, reason, ip, user_agent)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (request_id, subscriber_id, device_id, endpoint, decision, reason, ip, user_agent),
                )
    except Exception:
        pass  # never block analysis on logging

async def _log_usage(
    request_id: str,
    subscriber_id: str | None,
    device_id: str | None,
    endpoint: str,
    mode: str | None,
    num_images: int,
    image_bytes: int,
    payload_bytes: int,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    api_cost_usd: float | None,
    latency_ms: int,
    status_code: int,
) -> None:
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO usage_events
                    (request_id, subscriber_id, device_id, endpoint, mode,
                     num_images, image_bytes, payload_bytes,
                     model, prompt_tokens, completion_tokens, total_tokens,
                     api_cost_usd, latency_ms, status_code)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        request_id, subscriber_id, device_id, endpoint, mode,
                        num_images, image_bytes, payload_bytes,
                        model, prompt_tokens, completion_tokens, total_tokens,
                        api_cost_usd, latency_ms, status_code
                    ),
                )
    except Exception:
        pass



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

@app.get("/v1/entitlement")
async def entitlement_check(
    request: Request,
    x_subscriber_id: str | None = Header(default=None, alias="X-Subscriber-Id"),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
):
    """
    Lightweight entitlement check (no image upload). Useful for debugging client headers.
    """
    request_id = _new_request_id()

    # Note: x_subscriber_id may be missing/blank; _check_subscriber will return an appropriate reason.
    allowed, reason = await _check_subscriber(x_subscriber_id)

    ip = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")

    await _log_entitlement(
        request_id=request_id,
        subscriber_id=x_subscriber_id,
        device_id=x_device_id,
        endpoint="/v1/entitlement",
        decision="ALLOW" if allowed else ("ERROR" if reason == "DB_ERROR" else "DENY"),
        reason=reason,
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "ok": True,
        "request_id": request_id,
        "subscriber_id": x_subscriber_id,
        "device_id": x_device_id,
        "allowed": allowed,
        "reason": reason,
    }



@app.post("/v1/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    image2: UploadFile | None = File(None),  # optional 2nd frame (~0.5–1.5s later)
    mode: str = Form("brief"),
    model: str | None = Form(None),
    speed: str | None = Form(None),
    # Voice selector (1..6). Default 1 = onyx
    voice: int = Form(1),
    save_chart: bool = Form(False),
    frame_delay_ms: int | None = Form(None),

):
    t0 = time.perf_counter()
    request_id = _new_request_id()

    subscriber_id = request.headers.get("X-Subscriber-Id")
    device_id = request.headers.get("X-Device-Id")

    # Optional client-sent metrics (tray app can supply)
    num_images_hdr = _get_header_int(request, "X-Num-Images", 0)
    image_bytes_hdr = _get_header_int(request, "X-Image-Bytes", 0)
    payload_bytes = _get_header_int(request, "X-Payload-Bytes", 0)

    ip = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    endpoint = str(request.url.path)

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

    
    # Validate voice id
    try:
        voice = int(voice)
    except Exception:
        raise HTTPException(status_code=400, detail="voice must be an integer 1..6")
    if voice < 1 or voice > 6:
        raise HTTPException(status_code=400, detail="voice must be in range 1..6")


    # If headers didn't provide image counts/bytes, infer from payload
    inferred_num_images = (1 + (1 if raw2 is not None else 0))
    num_images = num_images_hdr or inferred_num_images
    image_bytes = image_bytes_hdr or (len(raw1) + (len(raw2) if raw2 is not None else 0))

    # --- Subscriber gate ---
    if not subscriber_id:
        await _log_entitlement(
            request_id=request_id,
            subscriber_id=None,
            device_id=device_id,
            endpoint=endpoint,
            decision="DENY",
            reason="missing_subscriber_id",
            ip=ip,
            user_agent=user_agent,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        await _log_usage(
            request_id=request_id,
            subscriber_id=None,
            device_id=device_id,
            endpoint=endpoint,
            mode=mode,
            num_images=num_images,
            image_bytes=image_bytes,
            payload_bytes=payload_bytes,
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            api_cost_usd=None,
            latency_ms=latency,
            status_code=401,
        )
        raise HTTPException(status_code=401, detail="Subscription required")

    allowed, reason = await _check_subscriber(subscriber_id)

    if not allowed:
        # DB hiccup -> treat as network down
        status = 503 if reason == "DB_ERROR" else 403
        decision = "ERROR" if reason == "DB_ERROR" else "DENY"

        await _log_entitlement(
            request_id=request_id,
            subscriber_id=subscriber_id,
            device_id=device_id,
            endpoint=endpoint,
            decision=decision,
            reason=reason.lower(),
            ip=ip,
            user_agent=user_agent,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        await _log_usage(
            request_id=request_id,
            subscriber_id=subscriber_id,
            device_id=device_id,
            endpoint=endpoint,
            mode=mode,
            num_images=num_images,
            image_bytes=image_bytes,
            payload_bytes=payload_bytes,
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            api_cost_usd=None,
            latency_ms=latency,
            status_code=status,
        )

        if reason == "DB_ERROR":
            raise HTTPException(status_code=503, detail="Network down, try again in a few minutes")
        raise HTTPException(status_code=403, detail="No active subscription")

    # allow path
    await _log_entitlement(
        request_id=request_id,
        subscriber_id=subscriber_id,
        device_id=device_id,
        endpoint=endpoint,
        decision="ALLOW",
        reason="ok",
        ip=ip,
        user_agent=user_agent,
    )

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

    # 4) Transcript
    mode_norm = (mode or "brief").strip().lower()
    if mode_norm in ("full", "detail", "detailed"):
        # Best-effort catalyst scan (headlines). Never blocks analysis if it fails.
        news_items = []
        try:
            if chart_facts.symbol:
                news_items = fetch_finviz_news(chart_facts.symbol, limit=8, timeout_s=6)
        except Exception:
            news_items = []

        try:
            transcript = await asyncio.wait_for(
                build_full_transcript_llm(chart_facts, trade_plan, l2_comment=l2_comment, news=news_items),
                timeout=float(os.getenv("FULL_TIMEOUT_SEC", "18")),
            )
        except Exception:
            # Fallback to deterministic transcript
            transcript = build_trade_transcript(chart_facts, trade_plan, l2_comment=l2_comment, mode=mode)
    else:
        transcript = build_trade_transcript(chart_facts, trade_plan, l2_comment=l2_comment, mode=mode)

    keep_looking = bool(
        (chart_facts.confidence is not None and chart_facts.confidence < 0.55)
        or (chart_facts.setup in (None, "unclear"))
        or (not chart_facts.symbol)
        or (not chart_facts.timeframe)
        or (trade_plan.side == "none")
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
                voice=voice,
            ),
            timeout=float(os.getenv("TTS_TIMEOUT_SEC", "25")),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="TTS timed out. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {type(e).__name__}: {e}")

    base = str(request.base_url).rstrip("/")
    audio_full_url = f"{base}{audio_url}"


    latency = int((time.perf_counter() - t0) * 1000)
    await _log_usage(
        request_id=request_id,
        subscriber_id=subscriber_id,
        device_id=device_id,
        endpoint=endpoint,
        mode=mode,
        num_images=num_images,
        image_bytes=image_bytes,
        payload_bytes=payload_bytes,
        model=model,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        api_cost_usd=None,
        latency_ms=latency,
        status_code=200,
    )
    return {
        "mode": mode,
        "voice": voice,
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
