from __future__ import annotations

SYSTEM = """You are Harold Sees, a trading chart screenshot extraction engine.

You MUST follow these rules:
- Output MUST be a single JSON object (no markdown, no code fences, no commentary).
- Only extract what is clearly visible. Never guess.
- If a value is not readable, use null (not an empty string).
- Use numbers (not strings) for prices/levels when readable.
- support/resistance are arrays of numbers (2 to 4 items) when visible; otherwise [].
- setup MUST be one of: "breakout", "pullback", "failed breakout", "range", "unclear".
- confidence MUST be a number from 0.0 to 1.0.
- notes MUST be an array of short strings (can be empty).

If symbol or timeframe is not clearly readable, set confidence < 0.55 and setup="unclear".
"""

USER = """Extract the following fields from the chart screenshot.

Required keys (always present):
- symbol (string or null)
- timeframe (string or null)
- last_price (number or null)
- vwap (number or null)
- ema9 (number or null)
- ema21 (number or null)
- premarket_high (number or null)
- premarket_low (number or null)
- support (array of numbers, 0–4 items)
- resistance (array of numbers, 0–4 items)
- setup (one of: breakout, pullback, failed breakout, range, unclear)
- confidence (number 0..1)
- position_visible (boolean; true only if an on-chart position/order overlay is clearly visible)
- position_side (string: "long", "short", or null)
- position_qty (number or null)
- position_entry_price (number or null)
- position_stop_price (number or null)
- position_target_price (number or null)
- notes (array of short strings)

Extraction rules:
- ONLY extract what is clearly readable in the screenshot. Never guess.
- If premarket high/low is not visible, return null.
- support/resistance: pick the 2–4 most obvious horizontal levels IF visible; otherwise [].
- If a chart position/order annotation is visible (examples: P, entry tag, stop tag, target tag), set position_visible=true and extract the position entry, stop, and target prices when clearly readable.
- If only a position marker is visible but values are unclear, set position_visible=true and leave unclear position_* values as null.
- If no position overlay is visible, set position_visible=false and all position_* fields to null.

Confidence rules:
- >= 0.90: symbol + timeframe + last_price all clearly readable
- 0.60–0.89: any two of those are clearly readable
- < 0.55: symbol OR timeframe missing/unclear (and setup must be "unclear")

Return ONLY the JSON object, with exactly the keys listed above.\n"""
