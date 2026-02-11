"""Backtesting package for evaluating signal quality and risk metrics."""

from .engine import BacktestEngine, BacktestConfig
from .metrics import compute_performance_metrics

__all__ = ["BacktestEngine", "BacktestConfig", "compute_performance_metrics"]
