#!/usr/bin/env python3
"""Run backtest for a ticker and save markdown output."""

import argparse
from pathlib import Path

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.metrics import compute_performance_metrics
from src.backtest.reports import render_markdown_report
from src.data.fetcher import YahooFinanceFetcher


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SMA crossover backtest')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--period', default='5y')
    parser.add_argument('--short-window', type=int, default=50)
    parser.add_argument('--long-window', type=int, default=200)
    parser.add_argument('--output', default='data/reports/backtest_report.md')
    args = parser.parse_args()

    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_price_history(args.ticker, period=args.period)

    engine = BacktestEngine(BacktestConfig(short_window=args.short_window, long_window=args.long_window))
    result = engine.run(data)
    summary = engine.summarize(result)
    metrics = compute_performance_metrics(result)

    out = render_markdown_report(args.ticker, summary, metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out)

    print(out)


if __name__ == '__main__':
    main()
