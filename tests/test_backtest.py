import pandas as pd

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.metrics import compute_performance_metrics


def _sample_prices(n: int = 260):
    idx = pd.date_range('2023-01-01', periods=n, freq='D')
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame({'Close': close})


def test_backtest_engine_runs_and_summarizes():
    engine = BacktestEngine(BacktestConfig(short_window=5, long_window=20, initial_capital=10000))
    df = engine.run(_sample_prices())

    assert not df.empty
    assert 'equity_curve' in df.columns
    summary = engine.summarize(df)
    assert 'total_return_pct' in summary
    assert 'trades' in summary


def test_backtest_metrics_compute_values():
    engine = BacktestEngine(BacktestConfig(short_window=5, long_window=20))
    df = engine.run(_sample_prices())
    metrics = compute_performance_metrics(df)

    assert 'win_rate' in metrics
    assert 'max_drawdown_pct' in metrics
    assert 'volatility_pct' in metrics
