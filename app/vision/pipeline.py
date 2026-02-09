from __future__ import annotations

import base64
import json
import os
from typing import Tuple

from openai import AsyncOpenAI

from .schema import ChartFacts
from .preprocess import preprocess_to_png_bytes
from .prompt import SYSTEM, USER

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")


def _to_data_url_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    return text.replace("```json", "").replace("```", "").strip()


def _is_num(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _filter_horizontal_levels(
    levels: list[float] | None,
    *,
    vwap: float | None,
    ema9: float | None,
    ema21: float | None,
) -> list[float]:
    """Option A: keep support/resistance strictly as horizontal levels, not MAs.

    We can't perfectly know if the model accidentally returned EMA/VWAP as a 'level',
    so we defensively remove anything that is *very close* to EMA9/EMA21/VWAP.
    """
    if not levels:
        return []

    out: list[float] = []
    for lv in levels:
        try:
            x = float(lv)
        except Exception:
            continue

        # Tolerance: tight enough to strip MA echoes, loose enough to keep real levels
        tol = max(0.02, abs(x) * 0.00015)  # ~1.5 bps (or 2 cents min)

        too_close = False
        for ma in (vwap, ema9, ema21):
            if ma is None:
                continue
            try:
                if abs(x - float(ma)) <= tol:
                    too_close = True
                    break
            except Exception:
                continue

        if not too_close:
            out.append(x)

    out = sorted(set(out))
    return out[:4]


def build_transcript_from_facts(f: ChartFacts, mode: str = "brief") -> str:
    """Mode-aware transcript with *short* brief output."""
    if f.confidence < 0.55 or not f.symbol or not f.timeframe:
        return "I can’t read this chart clearly enough. Keep looking."

    price = f.last_price
    sym_tf = f"{f.symbol} on {f.timeframe}."
    price_line = f"Price {price}." if price is not None else ""

    setup_word = (f.setup or "unclear")

    # One key level each for short modes
    sup = f.support[0] if f.support else None
    res = f.resistance[0] if f.resistance else None

    if mode == "brief":
        # Hard-cap: 2–3 short sentences, no indicator dump
        bits: list[str] = [sym_tf]
        if price_line:
            bits.append(price_line)
        bits.append(f"Setup: {setup_word}.")
        if sup is not None:
            bits.append(f"Key support {sup}.")
        elif res is not None:
            bits.append(f"Key resistance {res}.")
        return " ".join(bits[:3])

    if mode == "momentum":
        bits: list[str] = [sym_tf]
        if price_line:
            bits.append(price_line)
        bits.append(f"Setup: {setup_word}.")
        if res is not None:
            bits.append(f"Trigger reclaim {res}.")
        if sup is not None:
            bits.append(f"Invalidate below {sup}.")
        bits.append("Momentum focus: avoid chasing into overhead resistance; wait for clean reclaim.")
        return " ".join([b for b in bits if b])

    # full
    parts: list[str] = [sym_tf]
    if price_line:
        parts.append(price_line)

    parts.append(
        f"VWAP {f.vwap if f.vwap is not None else 'not visible'}, "
        f"EMA9 {f.ema9 if f.ema9 is not None else 'n/a'}, "
        f"EMA21 {f.ema21 if f.ema21 is not None else 'n/a'}."
    )
    parts.append(f"Setup: {setup_word}.")

    if f.premarket_high is not None or f.premarket_low is not None:
        parts.append(f"Premarket high {f.premarket_high}, low {f.premarket_low}.")

    if f.support:
        parts.append(f"Support: {', '.join(str(x) for x in f.support[:2])}.")
    if f.resistance:
        parts.append(f"Resistance: {', '.join(str(x) for x in f.resistance[:2])}.")

    if f.notes:
        parts.append("Notes: " + " ".join(f.notes[:2]))

    return " ".join(parts)


async def analyze_chart_image_bytes(raw_image_bytes: bytes, mode: str = "brief") -> Tuple[ChartFacts, str]:
    """Vision → ChartFacts (+ transcript).

    Enforces Option A post-parse: support/resistance should be horizontal levels,
    not EMA/VWAP values.
    """
    client = AsyncOpenAI()

    png = preprocess_to_png_bytes(raw_image_bytes)
    data_url = _to_data_url_png(png)

    resp = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    text = _strip_code_fences((resp.choices[0].message.content or "").strip())

    try:
        data = json.loads(text)

        if data.get("support") is None or not isinstance(data.get("support"), list):
            data["support"] = []
        if data.get("resistance") is None or not isinstance(data.get("resistance"), list):
            data["resistance"] = []

        vwap = float(data["vwap"]) if _is_num(data.get("vwap")) else None
        ema9 = float(data["ema9"]) if _is_num(data.get("ema9")) else None
        ema21 = float(data["ema21"]) if _is_num(data.get("ema21")) else None

        data["support"] = _filter_horizontal_levels(
            [x for x in data.get("support", []) if _is_num(x)],
            vwap=vwap,
            ema9=ema9,
            ema21=ema21,
        )
        data["resistance"] = _filter_horizontal_levels(
            [x for x in data.get("resistance", []) if _is_num(x)],
            vwap=vwap,
            ema9=ema9,
            ema21=ema21,
        )

        facts = ChartFacts.model_validate(data)
    except Exception as e:
        facts = ChartFacts(confidence=0.0, notes=[f"Vision JSON parse failed: {type(e).__name__}"])
        return facts, "I had trouble reading the chart output. Keep looking."

    transcript = build_transcript_from_facts(facts, mode=mode)
    return facts, transcript
