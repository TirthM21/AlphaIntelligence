from src.contracts.schemas import PriceHistorySchema, SignalSchema


def test_price_history_schema_to_dict():
    row = PriceHistorySchema(
        ticker='AAPL',
        date='2026-01-01',
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1234567,
    )
    payload = row.to_dict()
    assert payload['ticker'] == 'AAPL'
    assert payload['close'] == 100.5


def test_signal_schema_to_dict():
    sig = SignalSchema(ticker='AAPL', signal='buy', score=80.0, phase='Phase 2', reasons=['trend'])
    payload = sig.to_dict()
    assert payload['signal'] == 'buy'
    assert payload['score'] == 80.0
