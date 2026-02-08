# Add these models to your existing app/vision/schema.py

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

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
    notes: List[str] = Field(default_factory=list)



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
