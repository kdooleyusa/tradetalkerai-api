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


def build_transcript_from_facts(f: ChartFacts, mode: str = "brief") -> str:
    # If we can't confidently read basics, say "keep looking"
    if f.confidence < 0.55 or not f.symbol or not f.timeframe:
        return "I can’t read this chart clearly enough. Keep looking."

    parts: list[str] = []

    # Core
    parts.append(f"{f.symbol} on {f.timeframe}. Price {f.last_price}.")
    parts.append(
        f"VWAP {f.vwap if f.vwap is not None else 'not visible'}, "
        f"EMA9 {f.ema9 if f.ema9 is not None else 'n/a'}, "
        f"EMA21 {f.ema21 if f.ema21 is not None else 'n/a'}."
    )

    # Premarket
    if f.premarket_high is not None or f.premarket_low is not None:
        parts.append(f"Premarket high {f.premarket_high}, low {f.premarket_low}.")

    # Levels
    if f.support:
        parts.append(f"Support near {f.support[:4]}.")
    if f.resistance:
        parts.append(f"Resistance near {f.resistance[:4]}.")

    # Setup
    parts.append(f"Setup looks like {f.setup}.")

    # Mode tuning (you can map brief/full/momentum later to F-keys)
    if mode == "brief":
        return " ".join(parts[:5])

    if mode == "momentum":
        return " ".join(parts[:5]) + " Momentum focus: watch the next resistance break; invalidate on support loss."

    # Full
    if f.notes:
        parts.append("Notes: " + " ".join(f.notes[:3]))
    return " ".join(parts)


async def analyze_chart_image_bytes(raw_image_bytes: bytes, mode: str = "brief") -> Tuple[ChartFacts, str]:
    """
    Vision → ChartFacts → transcript

    Uses Chat Completions with image input (works with AsyncOpenAI on openai==1.61.0).
    """
    client = AsyncOpenAI()

    # Preprocess (upscale/contrast) for better readability
    png = preprocess_to_png_bytes(raw_image_bytes)
    data_url = _to_data_url_png(png)

    # Call vision model and force JSON-only response via prompt
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


    text = (resp.choices[0].message.content or "").strip()

    # Parse JSON into our schema
    try:
        data = json.loads(text)
        facts = ChartFacts.model_validate(data)
    except Exception:
        facts = ChartFacts(confidence=0.0, notes=["Vision JSON parse failed"])
        return facts, "I had trouble reading the chart output. Keep looking."

    transcript = build_transcript_from_facts(facts, mode=mode)
    return facts, transcript
