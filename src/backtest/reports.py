"""Report builders for backtest outputs."""

from typing import Dict


def render_markdown_report(ticker: str, summary: Dict[str, float], metrics: Dict[str, float]) -> str:
    return f"""# Backtest Report: {ticker}

## Summary
- Strategy total return: {summary.get('total_return_pct', 0)}%
- Buy & hold return: {summary.get('buy_hold_return_pct', 0)}%
- Trade count: {summary.get('trades', 0)}

## Risk & Quality
- Win rate: {metrics.get('win_rate', 0)}%
- Max drawdown: {metrics.get('max_drawdown_pct', 0)}%
- Annualized volatility: {metrics.get('volatility_pct', 0)}%
"""
