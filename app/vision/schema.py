from pydantic import BaseModel
from typing import Optional, List, Literal

SetupType = Literal["breakout","pullback","failed breakout","range","unclear"]

class ChartFacts(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    last_price: Optional[float] = None

    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None

    premarket_high: Optional[float] = None
    premarket_low: Optional[float] = None

    support: List[float] = []
    resistance: List[float] = []

    setup: SetupType = "unclear"

    confidence: float = 0.0
    notes: List[str] = []
