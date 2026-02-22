from __future__ import annotations

import json
import os
from typing import List, Optional, Any

from openai import AsyncOpenAI

from .schema import ChartFacts, TradePlan
from .news import NewsItem


TEXT_MODEL = os.getenv("TEXT_MODEL", os.getenv("VISION_MODEL", "gpt-4o-mini"))

def _usage_meta(resp, *, model_fallback: str | None = None) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    return {
        "model": getattr(resp, "model", None) or model_fallback,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
    }



def _clip(s: str, n: int) -> str | tuple[str, dict[str, Any]]:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1].rstrip() + "…")


async def build_full_transcript_llm(
    facts: ChartFacts,
    plan: TradePlan,
    *,
    l2_comment: Optional[str] = None,
    news: Optional[List[NewsItem]] = None,
    return_meta: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Create the F8/FULL spoken transcript (chart + plan + catalyst scan)."""

    symbol = (facts.symbol or "").strip().upper() or "UNKNOWN"
    tf = (facts.timeframe or "").strip() or "unknown timeframe"

    headlines = []
    for it in (news or [])[:8]:
        headlines.append(
            {
                "ts": _clip(it.ts, 24),
                "headline": _clip(it.headline, 140),
                "source": _clip(it.source or "", 40) or None,
            }
        )

    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "chart_facts": facts.model_dump(),
        "trade_plan": plan.model_dump(),
        "l2_comment": _clip(l2_comment or "", 320) or None,
        "headlines": headlines,
    }

    system = (
        "You are TradeTalkerAI. Speak like an elite intraday to 1-week stock trader. "
        "Create a spoken analysis (plain text) that is actionable and risk-aware. "
        "You only know what is inside the JSON; do not invent volume, float, fundamentals, or extra news."
    )

    user = (
        "Using the JSON below, produce an in-depth FULL (F8) analysis.\n\n"
        "Rules:\n"
        "- Output plain text only.\n"
        "- 10 to 16 sentences max. No bullet points.\n"
        "- Horizon: 1 day to 1 week.\n"
        "- MUST include: trend/structure, key levels, setup quality, a catalyst summary (from the headlines), "
        "and a clear plan.\n"
        "- If setup is NOT actionable, explain why in concrete terms and state what would make it actionable.\n"
        "- If actionable, give: entry trigger, stop/invalidation, target 1 and target 2, plus timing guidance.\n"
        "- Mention the Level 2 comment only if it changes confidence or requires caution.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = _clip((resp.choices[0].message.content or "").strip(), 1800)
    if return_meta:
        return text, _usage_meta(resp, model_fallback=TEXT_MODEL)
    return text
