from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field

# ----------------------------
# Chart extraction (Vision)
# ----------------------------
SetupType = Literal["breakout", "pullback", "failed breakout", "range", "unclear"]


class ChartFacts(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    last_price: Optional[float] = None

    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None

    premarket_high: Optional[float] = None
    premarket_low: Optional[float] = None

    support: List[float] = Field(default_factory=list)      # 2–4
    resistance: List[float] = Field(default_factory=list)   # 2–4

    setup: SetupType = "unclear"

    confidence: float = 0.0

    # Optional on-chart position overlay (Webull/TradingView/etc.) if visible
    position_visible: bool = False
    position_side: Optional[str] = None  # long / short / null if unclear
    position_qty: Optional[float] = None
    position_entry_price: Optional[float] = None
    position_stop_price: Optional[float] = None
    position_target_price: Optional[float] = None

    notes: List[str] = Field(default_factory=list)


# ----------------------------
# Level 2 (optional, 1–2 frames)
# ----------------------------
class L2Row(BaseModel):
    price: float
    size: Optional[float] = None  # shares (normalized)


class L2Snapshot(BaseModel):
    ladder_visible: bool = False
    best_bid_price: Optional[float] = None
    best_bid_size: Optional[float] = None
    best_ask_price: Optional[float] = None
    best_ask_size: Optional[float] = None
    bids: List[L2Row] = Field(default_factory=list)
    asks: List[L2Row] = Field(default_factory=list)

    # convenience metrics (filled in post)
    bid_sum: float = 0.0
    ask_sum: float = 0.0
    imbalance: Optional[float] = None  # bid_sum / (bid_sum + ask_sum)


class L2Delta(BaseModel):
    bid_sum_a: float = 0.0
    bid_sum_b: float = 0.0
    ask_sum_a: float = 0.0
    ask_sum_b: float = 0.0
    bid_sum_change: float = 0.0
    ask_sum_change: float = 0.0

    imbalance_a: Optional[float] = None
    imbalance_b: Optional[float] = None
    imbalance_change: Optional[float] = None

    bid_pull_count: int = 0
    bid_add_count: int = 0
    ask_pull_count: int = 0
    ask_add_count: int = 0


# ----------------------------
# Trade logic layer output
# ----------------------------
TradeSide = Literal["long", "short", "none"]
TradeQuality = Literal["A", "B", "C", "D", "F"]


class TradePlan(BaseModel):
    side: TradeSide = "none"

    # Core plan
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: List[float] = Field(default_factory=list)

    # Derived
    rr: Optional[float] = None
    quality: TradeQuality = "F"
    step_aside: List[str] = Field(default_factory=list)

    # Human/debug
    rationale: List[str] = Field(default_factory=list)
