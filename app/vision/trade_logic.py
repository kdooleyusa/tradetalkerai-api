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


def build_momentum_trade_plan(
    facts: ChartFacts,
    l2_frame1: Optional[L2Snapshot] = None,
    l2_frame2: Optional[L2Snapshot] = None,
    l2_delta: Optional[L2Delta] = None,
) -> TradePlan:
    """Bullish-only MOMENTUM-specific plan generator.
    Stricter than build_trade_plan():
      - prefers breakout / clean reclaim setups
      - enforces no-chase behavior
      - requires room to first target
      - weights L2 more heavily when visible
      - produces tighter, execution-focused plans
    """
    plan = TradePlan(side="none")

    if facts.confidence < 0.55 or not facts.symbol or not facts.timeframe or facts.setup in (None, "unclear"):
        plan.step_aside.append("Low confidence or unclear setup")
        plan.rationale.append("Momentum mode requires a clear chart read and setup classification")
        return plan

    price = facts.last_price
    if price is None:
        plan.step_aside.append("Missing last_price")
        plan.rationale.append("No readable last_price on screenshot")
        return plan

    setup = (facts.setup or "unclear").strip().lower()
    if setup in ("range", "unclear"):
        plan.step_aside.append("Momentum requires breakout or reclaim structure")
        plan.rationale.append(f"Setup={setup} is too ambiguous for momentum execution")
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

    trend_score = 0.0
    trend_notes: list[str] = []

    if vwap is not None:
        if price >= vwap:
            trend_score += 1.0
            trend_notes.append("price >= VWAP")
        else:
            trend_score -= 1.0
            trend_notes.append("price < VWAP")

    if ema9 is not None:
        if price >= ema9:
            trend_score += 1.0
            trend_notes.append("price >= EMA9")
        else:
            trend_score -= 0.5
            trend_notes.append("price < EMA9")

    if ema9 is not None and ema21 is not None:
        if ema9 >= ema21:
            trend_score += 1.0
            trend_notes.append("EMA9 >= EMA21")
        else:
            trend_score -= 1.0
            trend_notes.append("EMA9 < EMA21")

    if trend_score <= -1.5:
        plan.step_aside.append("Trend alignment weak for momentum (VWAP/EMA stack)")
        plan.rationale.append(" / ".join(trend_notes) if trend_notes else "Weak trend alignment")
        return plan

    plan.side = "long"
    plan.rationale.append(f"Momentum setup={setup}, confidence={facts.confidence:.2f}, l2_score={l2s:+.2f}")
    if trend_notes:
        plan.rationale.append("Trend: " + "; ".join(trend_notes))

    trigger = None
    stop_ref = None

    if setup == "breakout":
        breakout_level = _nearest_above(resistance, price) or price
        trigger = breakout_level + buf
        plan.entry = trigger

        stop_ref = _nearest_below(support, breakout_level)
        if stop_ref is None and ma_mid is not None:
            stop_ref = ma_mid
        if stop_ref is None:
            stop_ref = price - (buf * 5)
        plan.stop = stop_ref - buf

        plan.rationale.append("Breakout momentum: trigger over resistance with tight invalidation under support/MA")

    elif setup == "pullback":
        reclaim_candidates = [x for x in [ema9, vwap, ma_mid] if x is not None]
        reclaim_base = max(reclaim_candidates) if reclaim_candidates else None

        if reclaim_base is None:
            first_res = _nearest_above(resistance, price)
            if first_res is not None:
                trigger = first_res + buf
            else:
                trigger = price + buf
        else:
            trigger = reclaim_base + buf

        plan.entry = trigger

        stop_ref = _nearest_below(support, trigger)
        if stop_ref is None:
            for alt in [vwap, ema21, ma_mid]:
                if alt is not None and alt < trigger:
                    stop_ref = alt
                    break
        if stop_ref is None:
            stop_ref = price - (buf * 6)

        plan.stop = stop_ref - buf
        plan.rationale.append("Pullback momentum: reclaim trigger with invalidation below support/VWAP/EMA")

    elif setup == "failed breakout":
        if l2_frame1 and l2_frame1.ladder_visible and l2s <= -0.15:
            plan.step_aside.append("Failed breakout with weak L2 confirmation")
            plan.rationale.append("Momentum mode rejects failed-breakout reclaims when L2 is bearish")
            plan.side = "none"
            plan.entry = None
            plan.stop = None
            plan.targets = []
            return plan

        reclaim = None
        for cand in [vwap, ema9, ma_mid]:
            if cand is not None:
                reclaim = cand
                break

        if reclaim is None:
            plan.step_aside.append("Failed breakout needs a reclaim reference")
            plan.rationale.append("No VWAP/EMA/MA reclaim level visible")
            plan.side = "none"
            return plan

        trigger = reclaim + buf
        plan.entry = trigger

        stop_ref = _nearest_below(support, reclaim)
        if stop_ref is None:
            stop_ref = reclaim - (buf * 6)
        plan.stop = stop_ref - buf

        plan.rationale.append("Failed-breakout reclaim allowed only with reclaim trigger and non-bearish L2")

    else:
        plan.step_aside.append("Unsupported setup for momentum mode")
        plan.rationale.append(f"Setup={setup}")
        plan.side = "none"
        return plan

    if plan.entry is None or plan.stop is None:
        plan.step_aside.append("Incomplete momentum plan")
        plan.rationale.append("Missing entry/stop after setup logic")
        plan.side = "none"
        return plan

    if plan.stop >= plan.entry:
        repaired_stop = plan.entry - max(buf * 4, 0.02)
        plan.rationale.append("Repaired stop geometry using momentum fallback stop")
        plan.stop = repaired_stop

    t1 = _nearest_above(resistance, plan.entry + buf)
    if t1 is not None:
        plan.targets.append(t1)
        t2 = _nearest_above(resistance, t1 + buf)
        if t2 is not None:
            plan.targets.append(t2)
    else:
        risk = plan.entry - plan.stop
        if risk > 0:
            plan.targets = [plan.entry + risk * 1.4, plan.entry + risk * 2.0]
        else:
            plan.targets = []

    risk_now = max(plan.entry - plan.stop, 0.0)
    chase_allowance = max(buf * 2, min(0.08, risk_now * 0.35))
    if price > (plan.entry + chase_allowance):
        plan.step_aside.append("Extended above trigger — no-chase momentum")
        plan.rationale.append(
            f"Price {price:.4f} is already above trigger {plan.entry:.4f} beyond chase allowance"
        )

    if plan.targets:
        rr_val = _rr(plan.entry, plan.stop, plan.targets[0])
        plan.rr = round(rr_val, 2) if rr_val is not None else None
    else:
        plan.rr = None

    if plan.rr is None:
        plan.step_aside.append("No valid target / RR for momentum entry")
    elif plan.rr < 1.2:
        plan.step_aside.append("RR too low for momentum (< 1.2)")
        plan.rationale.append("Overhead room is too tight relative to risk")

    if plan.targets:
        t1_dist = plan.targets[0] - plan.entry
        if risk_now > 0 and t1_dist <= (risk_now * 1.15):
            plan.step_aside.append("Overhead resistance too close for momentum")
            plan.rationale.append("First target too near vs required momentum risk")

    if l2_frame1 and l2_frame1.ladder_visible:
        if l2s <= -0.25:
            plan.step_aside.append("L2 pressure bearish for momentum")
            plan.rationale.append("Bid pull / ask add pattern weakens breakout follow-through")
        elif l2s >= 0.20:
            plan.rationale.append("L2 confirms momentum (supportive bid/ask pressure)")
    else:
        plan.rationale.append("No L2 visible — momentum plan built from chart only")

    score = 0.0
    if setup in ("breakout", "pullback"):
        score += 2.0
    elif setup == "failed breakout":
        score += 1.0

    score += max(0.0, min(3.0, trend_score + 1.0))

    if facts.confidence >= 0.90:
        score += 2.0
    elif facts.confidence >= 0.80:
        score += 1.5
    elif facts.confidence >= 0.70:
        score += 1.0
    elif facts.confidence >= 0.60:
        score += 0.5

    if plan.rr is not None:
        if plan.rr >= 2.5:
            score += 2.0
        elif plan.rr >= 1.8:
            score += 1.5
        elif plan.rr >= 1.4:
            score += 1.0
        elif plan.rr >= 1.2:
            score += 0.5

    if l2_frame1 and l2_frame1.ladder_visible:
        if l2s >= 0.35:
            score += 1.0
        elif l2s >= 0.15:
            score += 0.5
        elif l2s <= -0.35:
            score -= 2.0
        elif l2s <= -0.15:
            score -= 1.0

    score -= 1.0 * len(plan.step_aside)

    if score >= 8.0:
        q = "A"
    elif score >= 6.5:
        q = "B"
    elif score >= 5.0:
        q = "C"
    elif score >= 4.0:
        q = "D"
    else:
        q = "F"

    if (not l2_frame1 or not l2_frame1.ladder_visible) and q == "A":
        q = "B"
        plan.rationale.append("A-grade capped to B (no L2 confirmation visible)")

    if plan.step_aside and q in ("A", "B", "C"):
        q = "D" if q == "C" else "C"

    plan.quality = q  # type: ignore

    blocker_phrases = (
        "Extended above trigger",
        "RR too low",
        "Overhead resistance too close",
        "L2 pressure bearish",
        "Low confidence",
        "Momentum requires breakout or reclaim structure",
    )
    if any(any(bp in s for bp in blocker_phrases) for s in plan.step_aside):
        plan.side = "none"

    plan.targets = sorted(set([round(float(t), 4) for t in plan.targets]))
    if plan.entry is not None:
        plan.entry = round(float(plan.entry), 4)
    if plan.stop is not None:
        plan.stop = round(float(plan.stop), 4)

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


def _mode_norm(mode: str | None) -> str:
    m = (mode or "brief").strip().lower()
    if m in ("momo", "momentum", "mom"):
        return "momentum"
    if m in ("full", "detail", "detailed"):
        return "full"
    if m in ("brief", "quick", "scan"):
        return "brief"
    return m


_MOMENTUM_HARD_BLOCKER_PHRASES = (
    "Low confidence",
    "Missing last_price",
    "Momentum requires breakout or reclaim structure",
    "Unsupported setup for momentum mode",
    "Incomplete momentum plan",
    "No valid target / RR for momentum entry",
    "RR too low",
    "Overhead resistance too close",
    "Extended above trigger",
    "L2 pressure bearish",
    "Failed breakout with weak L2 confirmation",
)


def trade_plan_has_hard_blockers(plan: TradePlan, mode: str = "brief") -> bool:
    mode_norm = (mode or "brief").strip().lower()
    if mode_norm in ("momentum", "momo", "mom"):
        return any(
            any(p in (reason or "") for p in _MOMENTUM_HARD_BLOCKER_PHRASES)
            for reason in (plan.step_aside or [])
        )
    return bool(plan.step_aside)


def should_keep_looking_from_plan(
    plan: TradePlan,
    mode: str = "brief",
    preexisting_keep_looking: bool = False,
) -> bool:
    return bool(
        preexisting_keep_looking
        or plan.side != "long"
        or (plan.quality in ("D", "F"))
        or trade_plan_has_hard_blockers(plan, mode=mode)
    )


def _voice_l2_commentary(l2_comment: str | None) -> str | None:
    """Normalize L2 commentary for BRIEF/MOMENTUM so a visible ladder is always addressed."""
    if not l2_comment:
        return None
    raw = (l2_comment or "").strip()
    if not raw:
        return None
    low = raw.lower()

    # Make low-signal / insufficient reads explicit instead of vague.
    if "delta captured" in low:
        return "Level two is visible, but not enough change is shown yet to read pressure or sweeps clearly."
    if "snapshot captured" in low:
        return "Level two is visible, but there is only a snapshot, so pressure and stacking are limited."
    if "book looks fairly balanced" in low:
        return "Level two looks fairly balanced, so pressure is mixed right now."

    # Minor wording cleanup for voice.
    if raw.endswith('.'):
        return raw
    return raw + '.'


def build_trade_transcript(
    facts: ChartFacts,
    plan: TradePlan,
    l2_comment: str | None = None,
    mode: str = "brief",
) -> str:
    """Professional screener voice.
    Uses setup quality (A/B/C/D/F). No sentiment words or numeric confidence.
    """

    mode = _mode_norm(mode)

    symbol = (facts.symbol or "Chart").strip()
    setup = (facts.setup or "unclear").strip().lower()
    price = _fmt_level(facts.last_price) if facts.last_price is not None else "?"

    quality = (plan.quality or "C").upper()
    quality_spoken = f"{quality} quality"

    # ---- HARD MOVE ON ----
    if (
        facts.confidence is None
        or facts.confidence < 0.55
        or setup in ("unclear", "range")
        or plan.side in (None, "none")
        or plan.quality in ("D", "F")
        or trade_plan_has_hard_blockers(plan, mode=mode)
    ):
        reason = None
        if plan.step_aside:
            reason = plan.step_aside[0].rstrip(".")
        elif plan.rationale:
            reason = plan.rationale[0].rstrip(".")

        if mode == "brief":
            bits = [f"{symbol}.", f"Price {price}."]
            l2_line = _voice_l2_commentary(l2_comment)
            if l2_line:
                bits.append(l2_line)
            bits.append("Move on.")
            return " ".join(bits)

        if mode == "momentum":
            bits = [f"{symbol}.", f"Price {price}."]
            if reason:
                bits.append(reason + ".")
            l2_line = _voice_l2_commentary(l2_comment)
            if l2_line:
                bits.append(l2_line)
            bits.append("Move on.")
            return " ".join(bits)

        if reason:
            return f"{symbol}. Price {price}. {reason}. Move on."
        return f"{symbol}. Price {price}. Move on."

    entry = _fmt_level(plan.entry) if plan.entry is not None else None
    stop = _fmt_level(plan.stop) if plan.stop is not None else None
    t1 = _fmt_level(plan.targets[0]) if getattr(plan, "targets", None) else None

    if mode == "brief":
        bits = [f"{symbol}.", quality_spoken + "."]
        if entry and stop and t1:
            bits.append(f"Entry {entry}, stop {stop}, target {t1}.")
        elif entry and stop:
            bits.append(f"Entry {entry}, stop {stop}.")
        l2_line = _voice_l2_commentary(l2_comment)
        if l2_line:
            bits.append(l2_line)
        return " ".join(bits)

    if mode == "momentum":
        bits = [f"{symbol}.", quality_spoken + "."]
        if entry and stop and t1:
            bits.append(f"Entry {entry}, stop {stop}, target {t1}.")
        if entry and stop:
            bits.append(f"Trigger {entry}. Invalidate {stop}.")
        for r in (plan.rationale or []):
            txt = (r or "").strip().rstrip(".")
            if txt:
                bits.append(txt + ".")
                break
        l2_line = _voice_l2_commentary(l2_comment)
        if l2_line:
            bits.append(l2_line)
        return " ".join(bits)

    bits = [f"{symbol}.", quality_spoken + "."]
    if entry and stop and t1:
        bits.append(f"Entry {entry}, stop {stop}, target {t1}.")
    elif entry and stop:
        bits.append(f"Entry {entry}, stop {stop}.")

    for r in (plan.rationale or []):
        bits.append(r.rstrip(".") + ".")
        break

    return " ".join(bits)
