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
    if f.confidence < 0.55 or not f.symbol or not f.timeframe:
        return "I can’t read this chart clearly enough. Keep looking."

    parts = [
        f"{f.symbol} on {f.timeframe}. Price {f.last_price}.",
        f"VWAP {f.vwap if f.vwap is not None else 'not visible'}, "
        f"EMA9 {f.ema9 if f.ema9 is not None else 'n/a'}, "
        f"EMA21 {f.ema21 if f.ema21 is not None else 'n/a'}.",
    ]

    if f.premarket_high is not None or f.premarket_low is not None:
        parts.append(f"Premarket high {f.premarket_high}, low {f.premarket_low}.")

    if f.support:
        parts.ap
