from __future__ import annotations

import asyncio
import os
import uuid
import time
import secrets
import json
from decimal import Decimal, ROUND_HALF_UP
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
        await _ensure_usage_events_columns()
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

async def _ensure_usage_events_columns() -> None:
    """Best-effort schema patch for newer usage tracking columns."""
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    ALTER TABLE usage_events
                    ADD COLUMN IF NOT EXISTS response_bytes bigint,
                    ADD COLUMN IF NOT EXISTS response_text_chars integer,
                    ADD COLUMN IF NOT EXISTS response_text_words integer,
                    ADD COLUMN IF NOT EXISTS error_type text,
                    ADD COLUMN IF NOT EXISTS error_message text,
                    ADD COLUMN IF NOT EXISTS pricing_version text,
                    ADD COLUMN IF NOT EXISTS tts_text_input_tokens integer,
                    ADD COLUMN IF NOT EXISTS tts_audio_output_tokens integer,
                    ADD COLUMN IF NOT EXISTS analysis_cost_usd numeric(10,6),
                    ADD COLUMN IF NOT EXISTS tts_cost_usd numeric(10,6),
                    ADD COLUMN IF NOT EXISTS credits_used integer,
                    ADD COLUMN IF NOT EXISTS ticker text
                """)
    except Exception:
        pass

def _safe_word_count(text: str | None) -> int:
    return len((text or "").split())

def _usage_int(v):
    try:
        return int(v) if v is not None else None
    except Exception:
        return None

def _merge_usage_meta(items: list[dict]) -> tuple[str | None, int | None, int | None, int | None, int | None, int | None]:
    """Aggregate usage while keeping TTS usage separate from main analysis token totals."""
    models: list[str] = []
    p = c = t = 0
    tts_text = tts_audio = 0
    any_main_tok = False
    any_tts_tok = False
    for it in items or []:
        if not isinstance(it, dict):
            continue
        m = str(it.get("model") or "")
        if m and m not in models:
            models.append(m)
        is_tts = "tts" in m.lower()
        if is_tts:
            tv = _usage_int(it.get("tts_text_input_tokens"))
            av = _usage_int(it.get("tts_audio_output_tokens"))
            if tv is not None:
                tts_text += tv
                any_tts_tok = True
            if av is not None:
                tts_audio += av
                any_tts_tok = True
            continue

        pv = _usage_int(it.get("prompt_tokens"))
        cv = _usage_int(it.get("completion_tokens"))
        tv2 = _usage_int(it.get("total_tokens"))
        if pv is not None:
            p += pv
            any_main_tok = True
        if cv is not None:
            c += cv
            any_main_tok = True
        if tv2 is not None:
            t += tv2
            any_main_tok = True

    prompt = p if any_main_tok else None
    completion = c if any_main_tok else None
    total = t if t else ((p + c) if any_main_tok else None)
    model_joined = ",".join(models[:6]) if models else None
    return model_joined, prompt, completion, total, (tts_text if any_tts_tok else None), (tts_audio if any_tts_tok else None)


def _pick_primary_model(model_names: str | None) -> str | None:
    parts = [p.strip() for p in str(model_names or "").split(",") if p.strip()]
    for p in parts:
        if "tts" not in p.lower():
            return p
    return parts[0] if parts else None


def _has_tts_model(model_names: str | None) -> bool:
    return any("tts" in p.strip().lower() for p in str(model_names or "").split(",") if p.strip())


async def _resolve_model_rate_row(model_name: str | None) -> dict | None:
    if not model_name:
        return None
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT model, kind, alias_to, input_per_1m, cached_input_per_1m, output_per_1m,
                           text_input_per_1m, audio_output_per_1m, pricing_version
                    FROM model_rate_table
                    WHERE model = %s
                    """,
                    (model_name,),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                keys = [
                    "model", "kind", "alias_to", "input_per_1m", "cached_input_per_1m", "output_per_1m",
                    "text_input_per_1m", "audio_output_per_1m", "pricing_version"
                ]
                return dict(zip(keys, row))
    except Exception:
        return None


def _q6(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def _estimate_tts_cost_from_chars(response_text_chars: int | None) -> float:
    chars = Decimal(int(response_text_chars or 0))
    est = chars * Decimal("0.00005")
    if est < Decimal("0.001"):
        est = Decimal("0.001")
    return float(_q6(est))


async def _estimate_costs_from_db(
    model_names: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    tts_text_input_tokens: int | None = None,
    tts_audio_output_tokens: int | None = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    analysis_cost = None
    tts_cost = None
    pricing_version = None

    primary = _pick_primary_model(model_names)
    rr = await _resolve_model_rate_row(primary)
    if rr and rr.get("kind") == "alias" and rr.get("alias_to"):
        rr = await _resolve_model_rate_row(rr.get("alias_to")) or rr
    if rr and rr.get("input_per_1m") is not None and rr.get("output_per_1m") is not None:
        in_rate = Decimal(str(rr["input_per_1m"]))
        out_rate = Decimal(str(rr["output_per_1m"]))
        p_tok = Decimal(int(prompt_tokens or 0))
        c_tok = Decimal(int(completion_tokens or 0))
        analysis_cost = _q6((p_tok / Decimal("1000000")) * in_rate + (c_tok / Decimal("1000000")) * out_rate)
        pricing_version = rr.get("pricing_version") or pricing_version

    if _has_tts_model(model_names) and (tts_text_input_tokens is not None or tts_audio_output_tokens is not None):
        tr = await _resolve_model_rate_row("gpt-4o-mini-tts")
        if tr and tr.get("text_input_per_1m") is not None and tr.get("audio_output_per_1m") is not None:
            ti_rate = Decimal(str(tr["text_input_per_1m"]))
            ao_rate = Decimal(str(tr["audio_output_per_1m"]))
            ti_tok = Decimal(int(tts_text_input_tokens or 0))
            ao_tok = Decimal(int(tts_audio_output_tokens or 0))
            tts_cost = _q6((ti_tok / Decimal("1000000")) * ti_rate + (ao_tok / Decimal("1000000")) * ao_rate)
            pricing_version = pricing_version or tr.get("pricing_version")

    total_cost = None
    if analysis_cost is not None or tts_cost is not None:
        total_cost = _q6((analysis_cost or Decimal("0")) + (tts_cost or Decimal("0")))

    return (
        float(analysis_cost) if analysis_cost is not None else None,
        float(tts_cost) if tts_cost is not None else None,
        float(total_cost) if total_cost is not None else None,
        pricing_version,
    )

def _round_to_penny_and_credits(api_cost_usd: float | None) -> tuple[float | None, int | None]:
    if api_cost_usd is None:
        return None, None
    amt = Decimal(str(api_cost_usd)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    credits = int((amt * Decimal("100")).to_integral_value(rounding=ROUND_HALF_UP))
    return float(amt), credits


async def _debit_credits_usage(
    subscriber_id: str | None,
    credits_used: int | None,
    usage_event_id: int | None,
    request_id: str | None,
    note: str | None = None,
) -> int | None:
    """Best-effort ledger debit. Returns remaining balance if credit_ledger exists."""
    if not subscriber_id or credits_used is None:
        return None
    if credits_used <= 0:
        return await _get_credit_balance(subscriber_id)
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    WITH bal AS (
                      SELECT COALESCE(SUM(credits_delta), 0)::integer AS b
                      FROM credit_ledger
                      WHERE subscriber_id = %s
                    ),
                    ins AS (
                      INSERT INTO credit_ledger
                      (subscriber_id, event_type, credits_delta, balance_after, usage_event_id, request_id, note, created_by)
                      SELECT %s, 'usage', %s, ((SELECT b FROM bal) + %s), %s, %s, %s, 'system'
                      RETURNING balance_after
                    )
                    SELECT balance_after FROM ins
                    """,
                    (subscriber_id, subscriber_id, -credits_used, -credits_used, usage_event_id, request_id, note or 'Analyze request credits'),
                )
                row = await cur.fetchone()
                return int(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


async def _get_credit_balance(subscriber_id: str | None) -> int | None:
    if not subscriber_id:
        return None
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COALESCE(SUM(credits_delta), 0)::integer FROM credit_ledger WHERE subscriber_id = %s",
                    (subscriber_id,),
                )
                row = await cur.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return None


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
    ticker: str | None,
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
    response_bytes: int | None = None,
    response_text_chars: int | None = None,
    response_text_words: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    pricing_version: str | None = None,
    tts_text_input_tokens: int | None = None,
    tts_audio_output_tokens: int | None = None,
    analysis_cost_usd: float | None = None,
    tts_cost_usd: float | None = None,
    credits_used: int | None = None,
) -> int | None:
    try:
        pool = await _db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO usage_events
                    (request_id, subscriber_id, device_id, endpoint, mode, ticker,
                     num_images, image_bytes, payload_bytes,
                     model, prompt_tokens, completion_tokens, total_tokens,
                     api_cost_usd, latency_ms, status_code,
                     response_bytes, response_text_chars, response_text_words,
                     error_type, error_message, pricing_version,
                     tts_text_input_tokens, tts_audio_output_tokens, analysis_cost_usd, tts_cost_usd, credits_used)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (
                        request_id, subscriber_id, device_id, endpoint, mode, ticker,
                        num_images, image_bytes, payload_bytes,
                        model, prompt_tokens, completion_tokens, total_tokens,
                        api_cost_usd, latency_ms, status_code,
                        response_bytes, response_text_chars, response_text_words,
                        error_type, error_message, pricing_version,
                        tts_text_input_tokens, tts_audio_output_tokens, analysis_cost_usd, tts_cost_usd, credits_used
                    ),
                )
                row = await cur.fetchone()
                return int(row[0]) if row and row[0] is not None else None
    except Exception:
        return None



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
    include_audio: bool = Form(True),

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

    usage_meta_items: list[dict] = []
    pricing_version = os.getenv("OPENAI_PRICING_VERSION") or "local_unset"

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
        chart_facts, _legacy, _chart_usage = await asyncio.wait_for(
            analyze_chart_image_bytes(raw1, mode=mode, return_meta=True),
            timeout=float(os.getenv("VISION_TIMEOUT_SEC", "25")),
        )
        usage_meta_items.append(_chart_usage)
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
        l2_1, _l2u1 = await asyncio.wait_for(
            analyze_l2_image_bytes(raw1, return_meta=True),
            timeout=float(os.getenv("L2_TIMEOUT_SEC", "25")),
        )
        usage_meta_items.append(_l2u1)
        if raw2 is not None:
            l2_2, _l2u2 = await asyncio.wait_for(
                analyze_l2_image_bytes(raw2, return_meta=True),
                timeout=float(os.getenv("L2_TIMEOUT_SEC", "25")),
            )
            usage_meta_items.append(_l2u2)
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
            transcript, _full_usage = await asyncio.wait_for(
                build_full_transcript_llm(chart_facts, trade_plan, l2_comment=l2_comment, news=news_items, return_meta=True),
                timeout=float(os.getenv("FULL_TIMEOUT_SEC", "18")),
            )
            usage_meta_items.append(_full_usage)
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

    # 5) TTS (optional; skipped when client requests text-only to save cost)
    mp3_path = None
    audio_url = None
    audio_full_url = None
    if include_audio:
        try:
            mp3_path, audio_url, _tts_usage = await asyncio.wait_for(
                generate_tts_mp3(
                    transcript=transcript,
                    analysis_id=analysis_id,
                    out_dir=Path(os.getenv("AUDIO_DIR", "./storage/audio")).expanduser().resolve(),
                    model=model,
                    speed=speed_f,
                    voice=voice,
                    return_meta=True,
                ),
                timeout=float(os.getenv("TTS_TIMEOUT_SEC", "25")),
            )
            usage_meta_items.append(_tts_usage)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="TTS timed out. Try again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS failed: {type(e).__name__}: {e}")

        base = str(request.base_url).rstrip("/")
        audio_full_url = f"{base}{audio_url}"

    response_obj = {
        "mode": mode,
        "voice": voice,
        "include_audio": include_audio,
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
        "mp3_bytes": (mp3_path.stat().st_size if (mp3_path and mp3_path.exists()) else 0),
        "audio_dir": (str(AUDIO_DIR) if include_audio else None),
        "filename": (mp3_path.name if mp3_path else None),
        "saved_upload": saved,
        "chart_dir": str(CHART_DIR),
        "vision_timeout_s": float(os.getenv("VISION_TIMEOUT_SEC", "25")),
        "l2_timeout_s": float(os.getenv("L2_TIMEOUT_SEC", "25")),
        "frame_delay_ms": frame_delay_ms,
    }

    response_bytes = len(json.dumps(response_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    response_text_chars = len(transcript or "")
    response_text_words = _safe_word_count(transcript)
    used_model, prompt_tokens, completion_tokens, total_tokens, tts_text_input_tokens, tts_audio_output_tokens = _merge_usage_meta(usage_meta_items)
    analysis_cost_usd, tts_cost_usd, api_cost_usd, pricing_version_db = await _estimate_costs_from_db(
        used_model or model, prompt_tokens, completion_tokens, tts_text_input_tokens, tts_audio_output_tokens
    )
    # Fallback TTS cost estimate when SDK does not expose TTS token usage
    if (tts_cost_usd is None) and _has_tts_model(used_model or model):
        tts_cost_usd = _estimate_tts_cost_from_chars(response_text_chars)
        base_analysis = Decimal(str(analysis_cost_usd or 0))
        api_cost_usd = float(_q6(base_analysis + Decimal(str(tts_cost_usd))))
        if pricing_version_db:
            pricing_version = f"{pricing_version_db}+ttschars0.00005_floor0.001"
        elif pricing_version:
            pricing_version = f"{pricing_version}+ttschars0.00005_floor0.001"
        else:
            pricing_version = "ttschars0.00005_floor0.001"
    elif pricing_version_db:
        pricing_version = pricing_version_db

    api_cost_usd, credits_used = _round_to_penny_and_credits(api_cost_usd)

    credits_remaining = None
    credit_line = None

    latency = int((time.perf_counter() - t0) * 1000)
    usage_event_id = await _log_usage(
        request_id=request_id,
        subscriber_id=subscriber_id,
        device_id=device_id,
        endpoint=endpoint,
        mode=mode,
        ticker=(chart_facts.symbol if chart_facts else None),
        num_images=num_images,
        image_bytes=image_bytes,
        payload_bytes=payload_bytes,
        model=used_model or model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        api_cost_usd=api_cost_usd,
        latency_ms=latency,
        status_code=200,
        response_bytes=response_bytes,
        response_text_chars=response_text_chars,
        response_text_words=response_text_words,
        pricing_version=pricing_version,
        tts_text_input_tokens=tts_text_input_tokens,
        tts_audio_output_tokens=tts_audio_output_tokens,
        analysis_cost_usd=analysis_cost_usd,
        tts_cost_usd=tts_cost_usd,
        credits_used=credits_used,
    )

    credits_remaining = await _debit_credits_usage(
        subscriber_id=subscriber_id,
        credits_used=credits_used,
        usage_event_id=usage_event_id,
        request_id=request_id,
        note=f"Analyze request ({mode})",
    )

    if credits_used is not None:
        if credits_remaining is not None:
            credit_line = f"Analysis used {credits_used} credits. You have {credits_remaining} credits left."
        else:
            credit_line = f"Analysis used {credits_used} credits."
        transcript_with_credit = (transcript or "").rstrip() + "\n\n" + credit_line
        response_obj["transcript"] = transcript_with_credit
        response_obj["credits_used"] = credits_used
        response_obj["credits_remaining"] = credits_remaining
        response_obj["api_cost_usd"] = api_cost_usd
    return response_obj
