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


def _fmt_level(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _fmt_levels(label: str, levels: list[float] | None) -> str | None:
    if not levels:
        return None
    shown = levels[:4]
    if len(shown) == 1:
        return f"{label} {_fmt_level(shown[0])}."
    if len(shown) == 2:
        return f"{label} {_fmt_level(shown[0])} then {_fmt_level(shown[1])}."
    middle = ", ".join(_fmt_level(v) for v in shown[:-1])
    last = _fmt_level(shown[-1])
    return f"{label} {middle}, then {last}."


def build_transcript_from_facts(f: ChartFacts, mode: str = "brief") -> str:
    if f.confidence < 0.55 or not f.symbol or not f.timeframe:
        return "I can’t read this chart clearly enough. Keep looking."

    parts: list[str] = []

    parts.append(f"{f.symbol} on {f.timeframe}. Price {f.last_price}.")
    parts.append(
        f"VWAP {f.vwap if f.vwap is not None else 'not visible'}, "
        f"EMA9 {f.ema9 if f.ema9 is not None else 'n/a'}, "
        f"EMA21 {f.ema21 if f.ema21 is not None else 'n/a'}."
    )

    if f.premarket_high is not None or f.premarket_low is not None:
        parts.append(f"Premarket high {f.premarket_high}, low {f.premarket_low}.")

    sup = _fmt_levels("Support", f.support)
    res = _fmt_levels("Resistance", f.resistance)
    if sup:
        parts.append(sup)
    if res:
        parts.append(res)

    pretty = {
        "breakout": "a breakout",
        "pullback": "a pullback",
        "failed breakout": "a failed breakout",
        "range": "range-bound",
        "unclear": "unclear",
    }.get(f.setup or "unclear", f.setup or "unclear")
    parts.append(f"Setup: {pretty}.")

    if f.support:
        parts.append(f"Invalidate below {_fmt_level(f.support[0])}.")

    if mode == "brief":
        return " ".join(parts[:6])

    if mode == "momentum":
        return " ".join(parts[:6]) + " Momentum focus: watch the next resistance break; invalidate on support loss."

    if f.notes:
        parts.append("Notes: " + " ".join(f.notes[:3]))
    return " ".join(parts)


async def analyze_chart_image_bytes(raw_image_bytes: bytes, mode: str = "brief") -> Tuple[ChartFacts, str]:
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
        facts = ChartFacts.model_validate(data)
    except Exception as e:
        facts = ChartFacts(confidence=0.0, notes=[f"Vision JSON parse failed: {type(e).__name__}"])
        return facts, "I had trouble reading the chart output. Keep looking."

    transcript = build_transcript_from_facts(facts, mode=mode)
    return facts, transcript
