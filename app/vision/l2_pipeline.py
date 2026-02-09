from __future__ import annotations

import base64
import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI

from .preprocess import preprocess_to_png_bytes
from .schema import L2Snapshot, L2Delta, L2Row
from .prompt_l2 import SYSTEM_L2, USER_L2

L2_MODEL = os.getenv("L2_MODEL", os.getenv("VISION_MODEL", "gpt-4o-mini"))

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMB])?\s*$", re.IGNORECASE)

def _to_data_url_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    return text.replace("```json", "").replace("```", "").strip()

def _parse_size(x) -> Optional[float]:
    """Accept numeric or strings like '26.12K', '1.2M'. Return shares as float."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s == "":
            return None
        m = _SIZE_RE.match(s)
        if not m:
            return None
        val = float(m.group(1))
        suf = (m.group(2) or "").upper()
        mult = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}.get(suf, 1.0)
        return val * mult
    return None

def _normalize_rows(rows) -> list[L2Row]:
    out: list[L2Row] = []
    if not isinstance(rows, list):
        return out
    for r in rows[:8]:
        if not isinstance(r, dict):
            continue
        price = r.get("price")
        size = r.get("size")
        try:
            price_f = float(price)
        except Exception:
            continue
        size_f = _parse_size(size)
        out.append(L2Row(price=price_f, size=size_f))
    return out

def _sum_sizes(rows: list[L2Row]) -> float:
    return float(sum((x.size or 0.0) for x in rows))

def _has_meaningful_l2(snap: L2Snapshot) -> bool:
    if snap.best_bid_price is not None or snap.best_ask_price is not None:
        return True
    if snap.bids or snap.asks:
        return True
    return False

async def analyze_l2_image_bytes(raw_image_bytes: bytes) -> L2Snapshot:
    """Extract a snapshot of the ladder/L2 visible in the screenshot."""
    client = AsyncOpenAI()
    png = preprocess_to_png_bytes(raw_image_bytes)
    data_url = _to_data_url_png(png)

    resp = await client.chat.completions.create(
        model=L2_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_L2},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_L2},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    text = _strip_code_fences((resp.choices[0].message.content or "").strip())
    data = json.loads(text)

    ladder_visible = bool(data.get("ladder_visible", False))
    bids = _normalize_rows(data.get("bids"))
    asks = _normalize_rows(data.get("asks"))

    snap = L2Snapshot(
        ladder_visible=ladder_visible,
        best_bid_price=float(data["best_bid_price"]) if data.get("best_bid_price") not in (None, "") else (bids[0].price if bids else None),
        best_bid_size=_parse_size(data.get("best_bid_size")) if data.get("best_bid_size") not in ("", None) else (bids[0].size if bids else None),
        best_ask_price=float(data["best_ask_price"]) if data.get("best_ask_price") not in (None, "") else (asks[0].price if asks else None),
        best_ask_size=_parse_size(data.get("best_ask_size")) if data.get("best_ask_size") not in ("", None) else (asks[0].size if asks else None),
        bids=bids,
        asks=asks,
    )

    if snap.ladder_visible and not _has_meaningful_l2(snap):
        snap.ladder_visible = False

    snap.bid_sum = _sum_sizes(bids[:5])
    snap.ask_sum = _sum_sizes(asks[:5])
    denom = snap.bid_sum + snap.ask_sum
    snap.imbalance = (snap.bid_sum / denom) if denom > 0 else None
    return snap

def compute_l2_delta(a: L2Snapshot, b: L2Snapshot) -> L2Delta:
    """2-frame inference: compare size sums and detect pulling/adding in top levels."""
    delta = L2Delta(
        bid_sum_a=a.bid_sum,
        bid_sum_b=b.bid_sum,
        ask_sum_a=a.ask_sum,
        ask_sum_b=b.ask_sum,
        imbalance_a=a.imbalance,
        imbalance_b=b.imbalance,
        bid_sum_change=(b.bid_sum - a.bid_sum),
        ask_sum_change=(b.ask_sum - a.ask_sum),
        imbalance_change=(None if (a.imbalance is None or b.imbalance is None) else (b.imbalance - a.imbalance)),
    )

    pull_bids = add_bids = 0
    for i in range(min(5, len(a.bids), len(b.bids))):
        sa = a.bids[i].size or 0.0
        sb = b.bids[i].size or 0.0
        if sb < sa * 0.75:
            pull_bids += 1
        elif sb > sa * 1.25:
            add_bids += 1

    pull_asks = add_asks = 0
    for i in range(min(5, len(a.asks), len(b.asks))):
        sa = a.asks[i].size or 0.0
        sb = b.asks[i].size or 0.0
        if sb < sa * 0.75:
            pull_asks += 1
        elif sb > sa * 1.25:
            add_asks += 1

    delta.bid_pull_count = pull_bids
    delta.bid_add_count = add_bids
    delta.ask_pull_count = pull_asks
    delta.ask_add_count = add_asks
    return delta

def build_l2_commentary(a: L2Snapshot | None, b: L2Snapshot | None, d: L2Delta | None) -> str:
    """Honest narration based on snapshot + 2-frame deltas."""
    if not a or not a.ladder_visible:
        return "Level two ladder not visible."

    if b is not None and not b.ladder_visible:
        return "Level two ladder not visible."

    if not b or not d:
        if a.imbalance is None:
            return "Level two snapshot captured."
        if a.imbalance >= 0.62:
            return "Level two snapshot: bids look stacked near the market."
        if a.imbalance <= 0.38:
            return "Level two snapshot: asks look stacked overhead."
        return "Level two snapshot: book looks fairly balanced."

    lines: list[str] = []

    if d.imbalance_change is not None:
        if d.imbalance_change > 0.05:
            lines.append("Bids strengthening.")
        elif d.imbalance_change < -0.05:
            lines.append("Asks strengthening.")
        # else: say nothing (no more 'roughly steady')

    if (d.ask_pull_count or 0) >= 2 and (d.bid_add_count or 0) >= 1:
        lines.append("Asks thinning while bids add.")
    elif (d.bid_pull_count or 0) >= 2 and (d.ask_add_count or 0) >= 1:
        lines.append("Bids thinning while asks add.")
    elif (d.ask_pull_count or 0) >= 2:
        lines.append("Asks pulling back.")
    elif (d.bid_pull_count or 0) >= 2:
        lines.append("Bids pulling back.")

    return " ".join(lines) if lines else "Level two delta captured."
