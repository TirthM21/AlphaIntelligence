"""Unit tests for data contracts schemas"""

from datetime import datetime
from src.contracts.schemas import PriceHistorySchema, SignalSchema

def test_price_history_schema():
    ph = PriceHistorySchema(
        ticker="AAPL",
        date=datetime(2023, 1, 1),
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
        volume=1000000
    )
    assert ph.ticker == "AAPL"
    assert ph.close == 152.0

def test_signal_schema():
    sig = SignalSchema(
        ticker="MSFT",
        signal_type="BUY",
        score=85.5,
        phase="2",
        timestamp=datetime.now(),
        reasons=["Breakout"],
        metadata={}
    )
    assert sig.score > 80
    assert sig.signal_type == "BUY"
