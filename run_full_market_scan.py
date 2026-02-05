#!/usr/bin/env python3
"""Full market scanner for ALL publicly traded US stocks.

This script:
1. Fetches the latest list of all US-listed stocks
2. Filters out penny stocks and low-volume stocks
3. Processes them in batches with rate limiting (1 TPS)
4. Tracks progress and can resume if interrupted
5. Generates comprehensive buy/sell lists
6. Saves results incrementally

Designed to run daily at 6:30 AM EST after market data is updated.

Usage:
    python run_full_market_scan.py
    python run_full_market_scan.py --resume
    python run_full_market_scan.py --clear-progress
    python run_full_market_scan.py --min-price 10 --min-volume 500000
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.data.universe_fetcher import USStockUniverseFetcher
from src.screening.batch_processor import BatchStockProcessor
from src.screening.benchmark import (
    analyze_spy_trend,
    calculate_market_breadth,
    format_benchmark_summary,
    should_generate_signals
)
from src.screening.signal_engine import score_buy_signal, score_sell_signal
from src.data.fundamentals_fetcher import create_fundamental_snapshot
from src.data.enhanced_fundamentals import EnhancedFundamentalsFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_daily_report(
    results: dict,
    buy_signals: list,
    sell_signals: list,
    spy_analysis: dict,
    breadth: dict,
    output_dir: str = "./data/daily_scans"
):
    """Save comprehensive daily market scan report.

    Args:
        results: Batch processing results
        buy_signals: List of buy signals
        sell_signals: List of sell signals
        spy_analysis: SPY analysis
        breadth: Market breadth
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Create comprehensive report
    output = []

    output.append("="*80)
    output.append("FULL MARKET SCAN - ALL US-LISTED STOCKS")
    output.append(f"Scan Date: {date_str}")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    output.append("="*80)
    output.append("")

    # Scanning statistics
    output.append("SCANNING STATISTICS")
    output.append("-"*80)
    output.append(f"Total Universe: {results['total_processed']:,} stocks")
    output.append(f"Analyzed: {results['total_analyzed']:,} stocks")
    output.append(f"Filtered Out: {results['total_processed'] - results['total_analyzed']:,} stocks")
    output.append(f"Processing Time: {results['processing_time_seconds']/3600:.1f} hours")
    output.append(f"Buy Signals: {len(buy_signals)}")
    output.append(f"Sell Signals: {len(sell_signals)}")
    output.append("")

    # Benchmark summary
    output.append(format_benchmark_summary(spy_analysis, breadth))
    output.append("")

    # Top buy signals
    output.append("="*80)
    output.append(f"TOP BUY SIGNALS (Score >= 70) - {len(buy_signals)} Total")
    output.append("="*80)
    output.append("")

    if buy_signals:
        # Show top 50 with details, then just ticker list for rest
        top_n = 50

        for i, signal in enumerate(buy_signals[:top_n], 1):
            output.append(f"\n{'#'*80}")
            output.append(f"BUY #{i}: {signal['ticker']} | Score: {signal['score']}/100")
            output.append(f"{'#'*80}")
            output.append(f"Phase: {signal['phase']}")

            if signal.get('breakout_price'):
                output.append(f"Breakout Price: ${signal['breakout_price']:.2f}")

            details = signal.get('details', {})
            if 'rs_slope' in details:
                output.append(f"RS Slope: {details['rs_slope']:.3f}")
            if 'volume_ratio' in details:
                output.append(f"Volume: {details['volume_ratio']:.1f}x average")

            output.append("\nKey Reasons:")
            for reason in signal['reasons'][:5]:  # Top 5 reasons
                output.append(f"  • {reason}")

            if signal.get('fundamental_snapshot'):
                output.append(signal['fundamental_snapshot'])

        # Remaining buys as ticker list
        if len(buy_signals) > top_n:
            output.append(f"\n{'='*80}")
            output.append(f"ADDITIONAL BUY SIGNALS ({len(buy_signals) - top_n} more)")
            output.append(f"{'='*80}\n")

            remaining = [s['ticker'] for s in buy_signals[top_n:]]
            # Format as comma-separated list, 10 per line
            for i in range(0, len(remaining), 10):
                output.append(", ".join(remaining[i:i+10]))

    else:
        output.append("✗ NO BUY SIGNALS TODAY")
        output.append("\nMarket conditions not favorable for new positions.")

    # Top sell signals
    output.append(f"\n\n{'='*80}")
    output.append(f"TOP SELL SIGNALS (Score >= 60) - {len(sell_signals)} Total")
    output.append(f"{'='*80}")
    output.append("")

    if sell_signals:
        # Show top 30 with details
        top_n = 30

        for i, signal in enumerate(sell_signals[:top_n], 1):
            output.append(f"\n{'#'*80}")
            output.append(f"SELL #{i}: {signal['ticker']} | Score: {signal['score']}/100 | Severity: {signal['severity'].upper()}")
            output.append(f"{'#'*80}")
            output.append(f"Phase: {signal['phase']}")

            if signal.get('breakdown_level'):
                output.append(f"Breakdown Level: ${signal['breakdown_level']:.2f}")

            details = signal.get('details', {})
            if 'rs_slope' in details:
                output.append(f"RS Rollover: {details['rs_slope']:.3f}")
            if 'volume_ratio' in details:
                output.append(f"Volume: {details['volume_ratio']:.1f}x average")

            output.append("\nKey Reasons:")
            for reason in signal['reasons'][:5]:
                output.append(f"  • {reason}")

        # Remaining sells as ticker list
        if len(sell_signals) > top_n:
            output.append(f"\n{'='*80}")
            output.append(f"ADDITIONAL SELL SIGNALS ({len(sell_signals) - top_n} more)")
            output.append(f"{'='*80}\n")

            remaining = [s['ticker'] for s in sell_signals[top_n:]]
            for i in range(0, len(remaining), 10):
                output.append(", ".join(remaining[i:i+10]))

    else:
        output.append("✗ NO SELL SIGNALS TODAY")

    output.append(f"\n\n{'='*80}")
    output.append("END OF DAILY MARKET SCAN")
    output.append(f"{'='*80}\n")

    # Save report
    report_text = "\n".join(output)

    filepath = Path(output_dir) / f"market_scan_{timestamp}.txt"
    with open(filepath, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {filepath}")

    # Also save as "latest" for easy access
    latest_path = Path(output_dir) / "latest_scan.txt"
    with open(latest_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Latest scan link: {latest_path}")

    # Print summary to console
    print(report_text)

    return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Full Market Scanner - All US-Listed Stocks'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    parser.add_argument(
        '--clear-progress',
        action='store_true',
        help='Clear progress and start fresh'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        default=5.0,
        help='Minimum stock price (default: $5.00)'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        default=10000.0,
        help='Maximum stock price (default: $10,000)'
    )
    parser.add_argument(
        '--min-volume',
        type=int,
        default=100000,
        help='Minimum average daily volume (default: 100,000)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Seconds between API calls (default: 1.0 = 1 TPS)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode - only process first 100 stocks'
    )
    parser.add_argument(
        '--use-fmp',
        action='store_true',
        help='Use FMP for enhanced fundamentals on buy signals'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("FULL MARKET SCANNER - ALL US-LISTED STOCKS")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # Initialize enhanced fundamentals fetcher
    fundamentals_fetcher = EnhancedFundamentalsFetcher()
    if args.use_fmp and fundamentals_fetcher.fmp_available:
        logger.info("FMP enabled - will use for refined buy signal analysis")
    elif args.use_fmp:
        logger.warning("--use-fmp specified but FMP_API_KEY not set or invalid. Using yfinance only.")

    try:
        # Initialize components
        universe_fetcher = USStockUniverseFetcher()
        processor = BatchStockProcessor(rate_limit_delay=args.rate_limit)

        # Clear progress if requested
        if args.clear_progress:
            processor.clear_progress()
            logger.info("Progress cleared - starting fresh")

        # Fetch stock universe
        logger.info("Fetching US stock universe...")
        tickers = universe_fetcher.fetch_universe()

        if not tickers:
            logger.error("Failed to fetch stock universe")
            sys.exit(1)

        logger.info(f"Universe loaded: {len(tickers):,} stocks")

        # Test mode - limit to first 100
        if args.test_mode:
            tickers = tickers[:100]
            logger.info(f"TEST MODE: Limited to {len(tickers)} stocks")

        # Process all stocks
        results = processor.process_batch(
            tickers,
            resume=args.resume,
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume
        )

        if 'error' in results:
            logger.error(f"Processing failed: {results['error']}")
            sys.exit(1)

        # Analyze SPY and calculate breadth
        logger.info("Calculating market metrics...")
        spy_analysis = analyze_spy_trend(processor.spy_data, processor.spy_price)
        breadth = calculate_market_breadth(results['phase_results'])

        # Determine if we should generate signals
        signal_rec = should_generate_signals(spy_analysis, breadth)

        # Score buy signals
        buy_signals = []
        if signal_rec['should_generate_buys']:
            logger.info("Scoring buy signals...")
            for analysis in results['analyses']:
                if analysis['phase_info']['phase'] in [1, 2]:
                    buy_signal = score_buy_signal(
                        ticker=analysis['ticker'],
                        price_data=analysis['price_data'],
                        current_price=analysis['current_price'],
                        phase_info=analysis['phase_info'],
                        rs_series=analysis['rs_series'],
                        fundamentals=analysis.get('fundamental_analysis')
                    )

                    if buy_signal['is_buy']:
                        # Use EnhancedFundamentalsFetcher if FMP is enabled
                        buy_signal['fundamental_snapshot'] = fundamentals_fetcher.create_snapshot(
                            analysis['ticker'],
                            quarterly_data=analysis.get('quarterly_data', {}),
                            use_fmp=args.use_fmp
                        )
                        buy_signals.append(buy_signal)

            buy_signals = sorted(buy_signals, key=lambda x: x['score'], reverse=True)
            logger.info(f"Found {len(buy_signals)} buy signals")

        # Score sell signals
        sell_signals = []
        if signal_rec['should_generate_sells']:
            logger.info("Scoring sell signals...")
            for analysis in results['analyses']:
                if analysis['phase_info']['phase'] in [3, 4]:
                    sell_signal = score_sell_signal(
                        ticker=analysis['ticker'],
                        price_data=analysis['price_data'],
                        current_price=analysis['current_price'],
                        phase_info=analysis['phase_info'],
                        rs_series=analysis['rs_series']
                    )

                    if sell_signal['is_sell']:
                        sell_signals.append(sell_signal)

            sell_signals = sorted(sell_signals, key=lambda x: x['score'], reverse=True)
            logger.info(f"Found {len(sell_signals)} sell signals")

        # Generate and save report
        logger.info("Generating daily report...")
        save_daily_report(
            results,
            buy_signals,
            sell_signals,
            spy_analysis,
            breadth
        )

        logger.info("="*80)
        logger.info("FULL MARKET SCAN COMPLETE")
        logger.info(f"Total Buy Signals: {len(buy_signals)}")
        logger.info(f"Total Sell Signals: {len(sell_signals)}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

    except KeyboardInterrupt:
        logger.info("\nScan interrupted by user - progress has been saved")
        logger.info("Run with --resume to continue from where you left off")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
