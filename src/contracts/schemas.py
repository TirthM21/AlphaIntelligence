"""Stable payload schemas for scanner and portfolio outputs."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DailySignal:
    ticker: str
    signal_type: str
    score: float
    phase: str
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "score": self.score,
            "phase": self.phase,
            "generated_at": self.generated_at,
            "reasons": self.reasons,
        }


@dataclass
class QuarterlyAllocation:
    ticker: str
    asset_type: str
    target_weight: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "asset_type": self.asset_type,
            "target_weight": self.target_weight,
            "rationale": self.rationale,
        }


@dataclass
class ScanSummary:
    total_processed: int
    total_analyzed: int
    buy_signals: int
    sell_signals: int
    runtime_seconds: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "total_analyzed": self.total_analyzed,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "runtime_seconds": self.runtime_seconds,
            "metadata": self.metadata or {},
        }


@dataclass
class PriceHistorySchema:
    """Normalized OHLCV row contract used by fetchers and downstream analytics."""

    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class SignalSchema:
    """Generic buy/sell signal contract with optional metadata."""

    ticker: str
    signal: str
    score: float
    phase: str
    reasons: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "score": self.score,
            "phase": self.phase,
            "reasons": self.reasons,
            "generated_at": self.generated_at,
            "metadata": self.metadata or {},
        }
