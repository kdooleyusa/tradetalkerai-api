from __future__ import annotations

import base64
import json
import os
from typing import Tuple

from openai import AsyncOpenAI

from .schema import ChartFacts
from .preprocess import preprocess_to_png_bytes
from .prompt import SYSTEM, USER

DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

def _to_data_url_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_transcript_from_facts(f: ChartFacts, mode: str = "brief") -> str:
    """
    mode: brief | full | momentum (you can map your F8/F9/F10 later)
    """
    if f.confidence < 0.55 or not f.symbol or not f.timeframe:
        return "I can’t read this chart clearly enough. Keep looking."

    parts = []
    parts.append(f"{f.symbol} on {f.timeframe}. Price {f.last_price}.")

    parts.append(
        f"VWAP {f.vwap if f.vwap is not None else 'not visible'}, "
        f"EMA9 {f.ema9 if f.ema9 is not None else 'n/a'}, "
        f"EMA21 {f.ema21 if f.ema21 is not None else 'n/a'}."
    )

    if f.premarket_high is not None or f.premarket_low is not None:
        parts.append(f"Premarket high {f.premarket_high}, low {f.premarket_low}.")

    if f.support:
        parts.append(f"Support near {f.support[:4]}.")
    if f.resistance:
        parts.append(f"Resistance near {f.resistance[:4]}.")

    parts.append(f"Setup looks like {f.setup}.")

    if mode == "brief":
        # keep it tight
        return " ".join(parts[:5])

    if mode == "momentum":
        return " ".join(parts[:5]) + " Momentum focus: watch the next resistance break; invalidate on support loss."

    # full
    if f.notes:
        parts.append("Notes: " + " ".join(f.notes[:3]))
    return " ".join(parts)

async def analyze_chart_image_bytes(raw_image_bytes: bytes, mode: str = "brief") -> Tuple[ChartFacts, str]:
    """
    Returns (chart_facts, transcript)
    """
    client = AsyncOpenAI()

    png = preprocess_to_png_bytes(raw_image_bytes)
    data_url = _to_data_url_png(png)

    # Use the Responses API with image input (works with the current OpenAI SDK style)
    resp = await client.responses.create(
        model=DEFAULT_VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": SYSTEM + "\n" + USER},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }],
    )

    text = (resp.output_text or "").strip()

    # Defensive JSON parsing
    try:
        data = json.loads(text)
        facts = ChartFacts.model_validate(data)
    except Exception:
        facts = ChartFacts(confidence=0.0, notes=["Vision JSON parse failed"])
        return facts, "I had trouble reading the chart output. Keep looking."

    transcript = build_transcript_from_facts(facts, mode=mode)
    return facts, transcript
