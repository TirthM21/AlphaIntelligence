"""Performance metrics for backtest results."""

from typing import Dict

import pandas as pd


def compute_performance_metrics(result_df: pd.DataFrame) -> Dict[str, float]:
    if result_df.empty or 'strategy_returns' not in result_df.columns:
        return {'win_rate': 0.0, 'max_drawdown_pct': 0.0, 'volatility_pct': 0.0}

    returns = result_df['strategy_returns'].dropna()
    if returns.empty:
        return {'win_rate': 0.0, 'max_drawdown_pct': 0.0, 'volatility_pct': 0.0}

    win_rate = (returns > 0).mean() * 100

    curve = (1 + returns).cumprod()
    rolling_max = curve.cummax()
    drawdown = (curve / rolling_max - 1) * 100
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    volatility = returns.std() * (252 ** 0.5) * 100

    return {
        'win_rate': round(float(win_rate), 2),
        'max_drawdown_pct': round(float(max_drawdown), 2),
        'volatility_pct': round(float(volatility), 2),
    }
