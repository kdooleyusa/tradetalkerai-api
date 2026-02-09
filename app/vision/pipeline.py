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


def build_transcript_from_facts(f: ChartFacts, mode: str = "brief") -> str:
    """Legacy core transcript (kept for backwards compatibility)."""
    if f.confidence < 0.55 or not f.symbol or not f.timeframe:
        return "I can’t read this chart clearly enough. Keep looking."

    parts: list[str] = []
    parts.append(f"{f.symbol} on {f.timeframe}. Price {f.last_price}.")
    parts.append(
        f"VWAP {f.vwap if f.vwap is not None else 'not visible'}, "
        f"EMA9 {f.ema9 if f.ema9 is not None else 'n/a'}, "
        f"EMA21 {f.ema21 if f.ema21 is not None else 'n/a'}."
    )
    if f.support:
        parts.append(f"Support near {f.support[:4]}.")
    if f.resistance:
        parts.append(f"Resistance near {f.resistance[:4]}.")
    parts.append(f"Setup looks like {f.setup}.")
    return " ".join(parts[:5])


async def analyze_chart_image_bytes(raw_image_bytes: bytes, mode: str = "brief") -> Tuple[ChartFacts, str]:
    """Vision → ChartFacts (+ legacy transcript)."""
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
