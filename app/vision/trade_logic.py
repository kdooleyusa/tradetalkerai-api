from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

from pydantic import BaseModel, Field

# Import your existing ChartFacts model
from .schema import ChartFacts


SetupQuality = Literal["A", "B", "C", "D", "F"]
BiasType = Literal["long", "none"]


class TradePlan(BaseModel):
    """
    A lightweight, deterministic trade plan derived from ChartFacts.
    Designed to be safe: if we can't compute a reasonable plan, we step aside.
    """
    bias: BiasType = "none"

    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    rr: Optional[float] = None  # reward/risk
    setup_quality: SetupQuality = "F"
    step_aside: bool = True

    reasons: List[str] = Field(default_factory=list)


def _as_sorted_unique(vals: List[float]) -> List[float]:
    out = []
    for v in vals or []:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv not in out:
            out.append(fv)
    out.sort()
    return out


def _nearest_below(levels: List[float], x: float) -> Optional[float]:
    below = [v for v in levels if v < x]
    return max(below) if below else None


def _nearest_above(levels: List[float], x: float) -> Optional[float]:
    above = [v for v in levels if v > x]
    return min(above) if above else None


def _round_price(p: Optional[float]) -> Optional[float]:
    if p is None:
        return None
    # Keep it human-friendly; Webull is usually 2 decimals for most stocks.
    return round(float(p), 2)


def _compute_rr(entry: float, stop: float, target: float) -> Optional[float]:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return reward / risk


def _grade(rr: Optional[float], confidence: float) -> SetupQuality:
    """
    Conservative grading:
    - Confidence gates the grade ceiling.
    - RR gates the base grade.
    """
    if rr is None:
        return "F"

    # RR-based base grade
    if rr >= 3.0:
        base = "A"
    elif rr >= 2.0:
        base = "B"
    elif rr >= 1.4:
        base = "C"
    elif rr >= 1.0:
        base = "D"
    else:
        base = "F"

    # Confidence ceiling
    if confidence < 0.55:
        return "F"
    if confidence < 0.70 and base == "A":
        return "B"
    if confidence < 0.70 and base == "B":
        return "C"
    if confidence < 0.80 and base == "A":
        return "A"  # allow A only above 0.8
    return base


def compute_trade_plan(f: ChartFacts, mode: str = "brief") -> TradePlan:
    """
    Deterministic plan based on extracted chart facts.
    Bullish-only for now (matches your "keep looking" / step-aside style).
    """
    plan = TradePlan()
    support = _as_sorted_unique(f.support)
    resistance = _as_sorted_unique(f.resistance)

    # Basic safety checks
    if f.confidence < 0.55 or not f.symbol or not f.timeframe or f.last_price is None:
        plan.reasons.append("Low confidence or missing basics (symbol/timeframe/price).")
        return plan

    last = float(f.last_price)
    plan.bias = "long"

    # Helper anchors
    vwap = f.vwap
    ema9 = f.ema9
    ema21 = f.ema21

    sup = _nearest_below(support, last) or _nearest_below(support, vwap or last) or _nearest_below(support, ema9 or last)
    res = _nearest_above(resistance, last) or _nearest_above(resistance, vwap or last) or _nearest_above(resistance, ema9 or last)

    setup = (f.setup or "unclear").lower().strip()

    # --- Setup-specific heuristics ---
    if setup == "pullback":
        # Prefer entry near VWAP/EMA9 if present
        entry = vwap or ema9 or last
        stop = sup if sup is not None else entry * 0.995  # 0.5% default
        target = res if res is not None else entry * 1.01  # 1% default
        plan.reasons.append("Pullback: entry near VWAP/EMA9, stop under nearest support, target at nearest resistance.")

    elif setup == "breakout":
        # Entry at (or just above) nearest resistance, if visible
        entry = res if res is not None else last
        # Stop at VWAP/EMA9 (whichever is lower) or slightly below entry
        anchor = None
        for a in (vwap, ema9, ema21):
            if a is not None:
                anchor = float(a) if anchor is None else min(anchor, float(a))
        stop = anchor if anchor is not None else entry * 0.99
        # Target: next resistance above entry if available, else 2R target
        next_res = _nearest_above(resistance, entry)  # since resistance is sorted
        if next_res is not None:
            target = next_res
            plan.reasons.append("Breakout: entry at resistance, target at next resistance.")
        else:
            rr2_target = entry + abs(entry - stop) * 2.0
            target = rr2_target
            plan.reasons.append("Breakout: entry at resistance, target set to 2R (no next resistance visible).")
        plan.reasons.append("Stop anchored to VWAP/EMA levels when visible.")

    elif setup == "range":
        # Bullish-only range: buy support, sell to mid/upper range
        if sup is None or res is None:
            plan.reasons.append("Range: missing clear support/resistance.")
            return plan
        entry = sup
        stop = sup * 0.995
        target = res
        plan.reasons.append("Range: bullish bounce off support to resistance.")

    elif setup == "failed breakout":
        # Bullish-only engine: step aside
        plan.reasons.append("Failed breakout: step aside (bullish-only).")
        return plan

    else:
        # Unclear: step aside unless RR is excellent and levels are clear
        plan.reasons.append("Unclear setup: default step aside unless plan quality is strong.")
        entry = vwap or ema9 or last
        stop = sup if sup is not None else entry * 0.995
        target = res if res is not None else entry * 1.01

    # Normalize
    entry = float(entry)
    stop = float(stop)
    target = float(target)

    # Sanity: long trade must have stop < entry < target
    if not (stop < entry < target):
        plan.reasons.append("Sanity check failed: stop/entry/target not in bullish order.")
        return plan

    rr = _compute_rr(entry, stop, target)
    qual = _grade(rr, float(f.confidence))

    plan.entry = _round_price(entry)
    plan.stop_loss = _round_price(stop)
    plan.take_profit = _round_price(target)
    plan.rr = round(rr, 2) if rr is not None else None
    plan.setup_quality = qual

    # Step-aside policy:
    # - Always step aside for D/F
    # - For C, allow only if mode != brief (i.e., "full" might still show it)
    if qual in ("A", "B"):
        plan.step_aside = False
    elif qual == "C":
        plan.step_aside = (mode == "brief")
        if plan.step_aside:
            plan.reasons.append("Quality C: brief mode steps aside; use full to see marginal setups.")
    else:
        plan.step_aside = True

    return plan


def plan_to_transcript(f: ChartFacts, plan: TradePlan) -> str:
    """
    Converts plan into a spoken summary.
    Keep it compact and trader-friendly.
    """
    if plan.step_aside or plan.bias == "none" or plan.entry is None:
        return "Step aside. Keep looking."

    bits = []
    bits.append(f"Entry {plan.entry}. Stop {plan.stop_loss}. Target {plan.take_profit}.")
    if plan.rr is not None:
        bits.append(f"R R {plan.rr}.")
    bits.append(f"Setup quality {plan.setup_quality}.")
    return " ".join(bits)
