from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

@dataclass
class PriceHistorySchema:
    """Schema for historical price data."""
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
@dataclass
class FundamentalSchema:
    """Schema for fundamental data."""
    ticker: str
    fetch_date: datetime
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    revenue_growth: Optional[float]
    sector: str

@dataclass
class SignalSchema:
    """Schema for a trading signal."""
    ticker: str
    signal_type: str # 'BUY' or 'SELL'
    score: float
    phase: str
    timestamp: datetime
    reasons: List[str]
    metadata: Dict[str, Any]

@dataclass
class PortfolioActionSchema:
    """Schema for a recommended portfolio action."""
    ticker: str
    action: str # 'BUY', 'SELL', 'HOLD', 'REDUCE', 'ADD'
    quantity: Optional[int]
    price: Optional[float]
    reason: str
    timestamp: datetime

@dataclass
class ScanSummarySchema:
    """Schema for a daily scan summary."""
    scan_date: datetime
    total_scanned: int
    buy_signals: int
    sell_signals: int
    top_buys: List[SignalSchema]
    top_sells: List[SignalSchema]
    market_breadth: Dict[str, float]
