from __future__ import annotations

from typing import Optional

from .schema import ChartFacts, L2Snapshot, L2Delta, TradePlan


def _buffer(price: Optional[float]) -> float:
    """Small adaptive buffer to avoid exact-touch entries/stops."""
    if not price or price <= 0:
        return 0.01
    return max(0.01, price * 0.0001)  # 1 bp or 1 cent, whichever larger


def _nearest_below(levels: list[float], x: float) -> Optional[float]:
    below = [v for v in levels if v <= x]
    return max(below) if below else None


def _nearest_above(levels: list[float], x: float) -> Optional[float]:
    above = [v for v in levels if v >= x]
    return min(above) if above else None


def _rr(entry: float, stop: float, target: float) -> Optional[float]:
    risk = entry - stop
    if risk <= 0:
        return None
    reward = target - entry
    if reward <= 0:
        return None
    return reward / risk


def _quality_from(rr_val: Optional[float], confidence: float, l2_score: float) -> str:
    if confidence < 0.55:
        return "F"

    rr_base = rr_val or 0.0
    if rr_base >= 3.0:
        q = "A"
    elif rr_base >= 2.2:
        q = "B"
    elif rr_base >= 1.6:
        q = "C"
    elif rr_base >= 1.2:
        q = "D"
    else:
        q = "F"

    # Nudge by confidence and L2
    if confidence >= 0.9 and q in ("B", "C"):
        q = "A" if q == "B" else "B"
    if l2_score <= -0.35 and q in ("A", "B", "C"):
        q = "B" if q == "A" else ("C" if q == "B" else "D")
    if l2_score >= 0.35 and q in ("C", "D"):
        q = "B" if q == "C" else "C"
    return q


def _l2_score(l2_a: Optional[L2Snapshot], l2_b: Optional[L2Snapshot], delta: Optional[L2Delta]) -> float:
    """Convert L2 into a simple -1..+1 score."""
    if not l2_a or not l2_a.ladder_visible:
        return 0.0

    score = 0.0

    imb_a = l2_a.imbalance
    if imb_a is not None:
        score += (imb_a - 0.5) * 1.2

    if l2_b and l2_b.imbalance is not None and imb_a is not None:
        score += (l2_b.imbalance - imb_a) * 1.5

    if delta:
        score += (delta.bid_add_count - delta.bid_pull_count) * 0.05
        score -= (delta.ask_add_count - delta.ask_pull_count) * 0.05
        if delta.imbalance_change is not None:
            score += delta.imbalance_change * 1.5

    return max(-1.0, min(1.0, score))


def build_trade_plan(
    facts: ChartFacts,
    l2_frame1: Optional[L2Snapshot] = None,
    l2_frame2: Optional[L2Snapshot] = None,
    l2_delta: Optional[L2Delta] = None,
) -> TradePlan:
    """Bullish-only v1 plan generator."""
    plan = TradePlan(side="none")

    if facts.confidence < 0.55 or not facts.symbol or not facts.timeframe or facts.setup in (None, "unclear"):
        plan.step_aside.append("Low confidence or unclear setup")
        plan.rationale.append("Vision couldn't confidently read symbol/timeframe/setup")
        return plan

    price = facts.last_price
    if price is None:
        plan.step_aside.append("Missing last_price")
        plan.rationale.append("No readable last_price on screenshot")
        return plan

    buf = _buffer(price)
    l2s = _l2_score(l2_frame1, l2_frame2, l2_delta)

    support = sorted(set([float(x) for x in (facts.support or [])]))
    resistance = sorted(set([float(x) for x in (facts.resistance or [])]))

    vwap = facts.vwap
    ema9 = facts.ema9
    ema21 = facts.ema21
    ma_stack = [x for x in [vwap, ema9, ema21] if x is not None]
    ma_mid = sum(ma_stack) / len(ma_stack) if ma_stack else None

    setup = facts.setup or "unclear"

    plan.side = "long"
    plan.rationale.append(f"Setup={setup}, confidence={facts.confidence:.2f}, l2_score={l2s:+.2f}")

    if setup == "breakout":
        trigger = _nearest_above(resistance, price) or price
        plan.entry = trigger + buf
        stop_ref = _nearest_below(support, price) or (ma_mid if ma_mid is not None else price - (buf * 5))
        plan.stop = stop_ref - buf

        t1 = _nearest_above(resistance, plan.entry + buf)
        if t1 is not None:
            plan.targets.append(t1)
            t2 = _nearest_above(resistance, t1 + buf)
            if t2 is not None:
                plan.targets.append(t2)
        if not plan.targets:
            risk = plan.entry - plan.stop
            plan.targets = [plan.entry + risk * 1.5, plan.entry + risk * 2.5]

        plan.rationale.append("Breakout: entry over resistance, stop under support/MA, targets to next resistance.")

    elif setup == "pullback":
        reclaim = (ma_mid + buf) if ma_mid is not None else None
        first_res = _nearest_above(resistance, price)
        plan.entry = reclaim if reclaim is not None else (first_res + buf if first_res is not None else price + buf)

        stop_ref = _nearest_below(support, price)
        if stop_ref is None and vwap is not None:
            stop_ref = vwap
        if stop_ref is None:
            stop_ref = price - (buf * 6)
        plan.stop = stop_ref - buf

        t1 = _nearest_above(resistance, plan.entry + buf)
        if t1 is not None:
            plan.targets.append(t1)
            t2 = _nearest_above(resistance, t1 + buf)
            if t2 is not None:
                plan.targets.append(t2)
        if not plan.targets:
            risk = plan.entry - plan.stop
            plan.targets = [plan.entry + risk * 1.2, plan.entry + risk * 2.0]

        plan.rationale.append("Pullback: entry on reclaim, stop under support/VWAP, targets at resistance.")

    elif setup == "failed breakout":
        plan.step_aside.append("Failed breakout (bullish-only) — need reclaim confirmation")
        plan.rationale.append("Bullish-only engine: skipping until reclaim + bid support is clear.")
        trigger = (vwap if vwap is not None else ma_mid)
        if trigger is not None:
            plan.entry = trigger + buf
            stop_ref = _nearest_below(support, trigger) or (trigger - buf * 6)
            plan.stop = stop_ref - buf
            t1 = _nearest_above(resistance, plan.entry + buf)
            if t1 is not None:
                plan.targets.append(t1)
        return plan

    elif setup == "range":
        plan.step_aside.append("Range setup — needs clean break or defined mean-reversion rules")
        plan.rationale.append("Range is ambiguous for this breakout/pullback engine.")
        hi = max(resistance) if resistance else None
        lo = min(support) if support else None
        if hi is not None:
            plan.entry = hi + buf
            plan.stop = (lo - buf) if lo is not None else (hi - buf * 8)
            plan.targets = [hi + (hi - (lo or (hi - buf * 8))) * 1.2]
        return plan

    else:
        plan.step_aside.append("Unclear setup")
        return plan

    if plan.entry is not None and plan.stop is not None and plan.targets:
        rr_val = _rr(plan.entry, plan.stop, plan.targets[0])
        plan.rr = round(rr_val, 2) if rr_val is not None else None
    else:
        plan.rr = None

    plan.quality = _quality_from(plan.rr, facts.confidence, l2s)  # type: ignore

    if plan.rr is not None and plan.rr < 1.5:
        plan.step_aside.append("RR too low (< 1.5)")
    if facts.confidence < 0.75:
        plan.step_aside.append("Chart read confidence moderate — consider closer screenshot")
    if l2_frame1 and l2_frame1.ladder_visible:
        if l2s <= -0.35:
            plan.step_aside.append("L2 pressure bearish (bid pull / ask add)")
        elif l2s >= 0.35:
            plan.rationale.append("L2 supportive (bid add / ask pull)")

    plan.targets = sorted(set([round(float(t), 4) for t in plan.targets]))
    return plan


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


def _fmt(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


def _short_l2(l2_comment: Optional[str]) -> Optional[str]:
    """Only keep high-signal L2 phrases. Drop filler."""
    if not l2_comment:
        return None
    low = l2_comment.lower()

    keep = (
        "asks strengthening",
        "bids strengthening",
        "asks thinning",
        "bids thinning",
        "bids add",
        "asks add",
        "asks pulling",
        "bids pulling",
        "bids stacked",
        "asks stacked",
    )
    if any(k in low for k in keep):
        return l2_comment.strip().rstrip(".") + "."
    return None


def _confidence_adj(conf: float | None) -> str:
    if conf is None:
        return "unknown-confidence"
    if conf >= 0.85:
        return "high-confidence"
    if conf >= 0.70:
        return "medium-confidence"
    if conf >= 0.55:
        return "low-confidence"
    return "very-low-confidence"


def _sentiment_word(setup: str | None, side: str | None, l2_comment: Optional[str]) -> str:
    # tiny, scanner-style words
    setup = (setup or "unclear").lower().strip()
    side = (side or "none").lower().strip()

    if setup == "range":
        return "choppy"
    if setup == "breakout":
        return "hot"
    if setup == "pullback":
        return "reset"
    if setup == "failed breakout":
        return "rejecting"
    # If L2 is strong, let it influence a tiny sentiment
    if l2_comment:
        low = l2_comment.lower()
        if "asks strengthening" in low or "asks stacked" in low:
            return "heavy"
        if "bids strengthening" in low or "bids stacked" in low:
            return "supported"
        if "asks thinning" in low:
            return "lifting"
        if "bids thinning" in low:
            return "soft"
    # fallback based on side
    if side == "long":
        return "bullish"
    if side == "short":
        return "bearish"
    return "foggy"


def build_trade_transcript(
    facts: ChartFacts,
    plan: TradePlan,
    l2_comment: str | None = None,
    mode: str = "brief",
) -> str:
    """
    Mode A+ (Ultra-short screener voice, slightly richer):
      - Adds confidence adjective + tiny sentiment word
      - Unclear path: "SYMBOL high-confidence. Price X. choppy. Position unclear because ... Move on."
      - Actionable path stays short: entry/stop/target in one line
    """
    symbol = (facts.symbol or "Chart").strip()
    setup = (facts.setup or "unclear").strip()
    adj = _confidence_adj(facts.confidence)
    price = _fmt(facts.last_price) or "?"
    l2 = _short_l2(l2_comment)
    senti = _sentiment_word(setup, plan.side, l2)

    # ---- Unclear / move-on gates ----
    if facts.confidence is None or facts.confidence < 0.55:
        return f"{symbol} {adj}. Price {price}. {senti}. Position unclear because low confidence read. Move on."

    if not facts.symbol or not facts.timeframe:
        return f"{symbol} {adj}. Price {price}. {senti}. Position unclear because symbol or timeframe missing. Move on."

    if setup.lower() == "unclear":
        return f"{symbol} {adj}. Price {price}. {senti}. Position unclear because setup not confirmed. Move on."

    # Step-aside/no-trade -> move on (ultra short)
    if plan.side in (None, "none") or plan.step_aside:
        reason = None
        if plan.step_aside:
            reason = plan.step_aside[0].rstrip(".")
        if mode == "brief" or not reason:
            out = f"{symbol} {adj}. Price {price}. {senti}. Move on."
        else:
            out = f"{symbol} {adj}. Price {price}. {senti}. {reason}. Move on."
        if l2:
            out = out.rstrip(".") + ". " + l2
        return out[:220].rstrip()

    entry = _fmt(plan.entry)
    stop = _fmt(plan.stop)
    t1 = _fmt(plan.targets[0]) if plan.targets else None

    # ---- BRIEF ----
    if mode == "brief":
        bits = [f"{symbol} {adj}.", f"Price {price}.", f"{senti}."]
        if entry and stop and t1:
            bits.append(f"Entry {entry}, stop {stop}, target {t1}.")
        elif entry and stop:
            bits.append(f"Entry {entry}, stop {stop}.")
        else:
            bits.append("Move on.")
        if l2:
            bits.append(l2)
        return " ".join(bits)[:180].rstrip()

    # ---- MOMENTUM ----
    if mode == "momentum":
        bits = [f"{symbol} {adj}.", f"Price {price}.", f"{senti}."]
        if entry and stop and t1:
            bits.append(f"Entry {entry}, stop {stop}, target {t1}.")
        if entry and stop:
            bits.append(f"Trigger {entry}. Invalidate {stop}.")
        if l2:
            bits.append(l2)
        return " ".join(bits)[:240].rstrip()

    # ---- FULL (still short) ----
    bits = [f"{symbol} {adj}.", f"Price {price}.", f"{senti}."]
    if entry and stop and t1:
        bits.append(f"Entry {entry}, stop {stop}, target {t1}.")
    elif entry and stop:
        bits.append(f"Entry {entry}, stop {stop}.")
    if plan.rationale:
        bits.append(plan.rationale[0].rstrip(".") + ".")
    if l2:
        bits.append(l2)
    return " ".join(bits)[:320].rstrip()
