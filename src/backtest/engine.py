"""Simple backtesting engine for price-series strategies."""

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class BacktestConfig:
    short_window: int = 50
    long_window: int = 200
    initial_capital: float = 10000.0


class BacktestEngine:
    """Runs a long-only SMA crossover backtest on OHLCV data."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, price_df: pd.DataFrame) -> pd.DataFrame:
        if price_df.empty or 'Close' not in price_df.columns:
            return pd.DataFrame()

        df = price_df.copy()
        df = df.sort_index()
        df['sma_short'] = df['Close'].rolling(self.config.short_window).mean()
        df['sma_long'] = df['Close'].rolling(self.config.long_window).mean()
        df['signal'] = (df['sma_short'] > df['sma_long']).astype(int)
        df['position'] = df['signal'].shift(1).fillna(0)
        df['returns'] = df['Close'].pct_change().fillna(0)
        df['strategy_returns'] = df['position'] * df['returns']
        df['equity_curve'] = self.config.initial_capital * (1 + df['strategy_returns']).cumprod()
        return df

    def summarize(self, result_df: pd.DataFrame) -> Dict[str, float]:
        if result_df.empty:
            return {
                'total_return_pct': 0.0,
                'buy_hold_return_pct': 0.0,
                'trades': 0,
            }

        start_capital = self.config.initial_capital
        end_capital = float(result_df['equity_curve'].iloc[-1])
        total_return_pct = (end_capital / start_capital - 1) * 100

        bh_start = float(result_df['Close'].iloc[0])
        bh_end = float(result_df['Close'].iloc[-1])
        buy_hold_return_pct = ((bh_end / bh_start) - 1) * 100 if bh_start else 0.0

        trades = int((result_df['position'].diff().fillna(0).abs() > 0).sum())

        return {
            'total_return_pct': round(total_return_pct, 2),
            'buy_hold_return_pct': round(buy_hold_return_pct, 2),
            'trades': trades,
        }
