"""Main Quant Analysis & Execution Engine.

This is the autonomous engine that:
1. Fetches data (price history + fundamentals)
2. Detects breakout phases
3. Scores stocks
4. Outputs daily buy/sell lists
5. Produces derivative metrics and summaries
6. Provides benchmark stats
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.data.fetcher import YahooFinanceFetcher
from src.data.fundamentals_fetcher import (
    analyze_fundamentals_for_signal
)
from src.data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from .phase_indicators import (
    classify_phase,
    calculate_relative_strength,
    calculate_sma
)
from .signal_engine import (
    score_buy_signal,
    score_sell_signal,
    format_signal_output
)
from .benchmark import (
    analyze_spy_trend,
    calculate_market_breadth,
    format_benchmark_summary,
    should_generate_signals
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantAnalysisEngine:
    """Autonomous Quant Analysis & Execution Engine."""

    def __init__(self, cache_dir: str = "./data/cache", use_fmp: bool = False):
        """Initialize the engine.

        Args:
            cache_dir: Directory for caching data
            use_fmp: Whether to use FMP for enhanced fundamentals
        """
        self.fetcher = YahooFinanceFetcher(cache_dir=cache_dir)
        self.enhanced_fetcher = EnhancedFundamentalsFetcher()
        self.use_fmp = use_fmp
        self.spy_data = None
        self.spy_price = None
        logger.info(f"QuantAnalysisEngine initialized (FMP: {use_fmp})")

    def fetch_spy_data(self) -> bool:
        """Fetch SPY benchmark data.

        Returns:
            True if successful
        """
        try:
            logger.info("Fetching SPY data...")
            spy_hist = self.fetcher.fetch_price_history('SPY', period='2y')

            if spy_hist.empty:
                logger.error("Failed to fetch SPY data")
                return False

            self.spy_data = spy_hist
            self.spy_price = spy_hist['Close'].iloc[-1]
            logger.info(f"SPY data fetched: {len(spy_hist)} days, current price: ${self.spy_price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return False

    def analyze_stock(self, ticker: str) -> Optional[Dict]:
        """Analyze a single stock.

        Args:
            ticker: Stock ticker

        Returns:
            Analysis dict or None if failed
        """
        try:
            logger.info(f"Analyzing {ticker}...")

            # Fetch price history
            price_data = self.fetcher.fetch_price_history(ticker, period='2y')

            if price_data.empty or len(price_data) < 200:
                logger.warning(f"{ticker}: Insufficient price data ({len(price_data)} days)")
                return None

            current_price = price_data['Close'].iloc[-1]

            # Classify phase
            phase_info = classify_phase(price_data, current_price)

            # Calculate relative strength vs SPY
            rs_series = calculate_relative_strength(
                price_data['Close'],
                self.spy_data['Close'],
                period=63
            )

            # Fetch quarterly fundamentals
            quarterly_data = self.enhanced_fetcher.fetch_quarterly_data(ticker, use_fmp=self.use_fmp)
            fundamental_analysis = analyze_fundamentals_for_signal(quarterly_data)

            return {
                'ticker': ticker,
                'price_data': price_data,
                'current_price': current_price,
                'phase_info': phase_info,
                'rs_series': rs_series,
                'quarterly_data': quarterly_data,
                'fundamental_analysis': fundamental_analysis
            }

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    def screen_stocks(self, tickers: List[str]) -> Dict[str, any]:
        """Screen a list of stocks for buy/sell signals.

        Args:
            tickers: List of stock tickers

        Returns:
            Dict with buy list, sell list, and benchmark data
        """
        logger.info(f"Screening {len(tickers)} stocks...")

        # Ensure SPY data is loaded
        if self.spy_data is None:
            if not self.fetch_spy_data():
                logger.error("Cannot proceed without SPY data")
                return {
                    'error': 'Failed to fetch SPY data',
                    'buys': [],
                    'sells': []
                }

        # Analyze SPY
        spy_analysis = analyze_spy_trend(self.spy_data, self.spy_price)

        # Analyze all stocks
        all_analyses = []
        phase_results = []

        for ticker in tickers:
            analysis = self.analyze_stock(ticker)
            if analysis:
                all_analyses.append(analysis)
                phase_results.append({
                    'ticker': ticker,
                    'phase': analysis['phase_info']['phase']
                })

        logger.info(f"Successfully analyzed {len(all_analyses)}/{len(tickers)} stocks")

        # Calculate market breadth
        breadth = calculate_market_breadth(phase_results)

        # Determine if we should generate signals
        signal_recommendation = should_generate_signals(spy_analysis, breadth)

        # Score buy signals
        buy_candidates = []
        if signal_recommendation['should_generate_buys']:
            for analysis in all_analyses:
                buy_signal = score_buy_signal(
                    ticker=analysis['ticker'],
                    price_data=analysis['price_data'],
                    current_price=analysis['current_price'],
                    phase_info=analysis['phase_info'],
                    rs_series=analysis['rs_series'],
                    fundamentals=analysis['fundamental_analysis']
                )

                if buy_signal['is_buy']:
                    # Add fundamental snapshot
                    buy_signal['fundamental_snapshot'] = create_fundamental_snapshot(
                        analysis['ticker'],
                        analysis['quarterly_data']
                    )
                    buy_candidates.append(buy_signal)

        # Sort by score
        buy_candidates = sorted(buy_candidates, key=lambda x: x['score'], reverse=True)

        # Score sell signals
        sell_candidates = []
        if signal_recommendation['should_generate_sells']:
            for analysis in all_analyses:
                sell_signal = score_sell_signal(
                    ticker=analysis['ticker'],
                    price_data=analysis['price_data'],
                    current_price=analysis['current_price'],
                    phase_info=analysis['phase_info'],
                    rs_series=analysis['rs_series'],
                    previous_phase=None  # Could track this in a database
                )

                if sell_signal['is_sell']:
                    sell_candidates.append(sell_signal)

        # Sort by score
        sell_candidates = sorted(sell_candidates, key=lambda x: x['score'], reverse=True)

        return {
            'timestamp': datetime.now().isoformat(),
            'spy_analysis': spy_analysis,
            'breadth': breadth,
            'signal_recommendation': signal_recommendation,
            'buys': buy_candidates,
            'sells': sell_candidates,
            'total_analyzed': len(all_analyses)
        }

    def run(self, tickers: List[str]) -> str:
        """Run the complete screening engine and return formatted output.

        Args:
            tickers: List of stock tickers to screen

        Returns:
            Formatted output string
        """
        logger.info("="*60)
        logger.info("QUANT ANALYSIS & EXECUTION ENGINE - STARTING")
        logger.info("="*60)

        # Screen stocks
        results = self.screen_stocks(tickers)

        if 'error' in results:
            return f"ERROR: {results['error']}"

        # Format output
        output = []

        # Header
        output.append("\n" + "="*60)
        output.append("QUANT ANALYSIS & EXECUTION ENGINE")
        output.append(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Stocks Analyzed: {results['total_analyzed']}")
        output.append("="*60)

        # Benchmark Summary
        output.append(format_benchmark_summary(
            results['spy_analysis'],
            results['breadth']
        ))

        # Buy List
        output.append("\n" + "="*60)
        output.append("BUY LIST (Score >= 70)")
        output.append("="*60)

        if results['buys']:
            output.append(f"\nFound {len(results['buys'])} BUY signals:\n")

            for i, buy in enumerate(results['buys'], 1):
                output.append(f"\n{'#'*60}")
                output.append(f"BUY #{i}: {buy['ticker']} | Score: {buy['score']}/100")
                output.append(f"{'#'*60}")
                output.append(f"Phase: {buy['phase']}")
                output.append(f"Breakout Price: ${buy['breakout_price']:.2f}" if buy['breakout_price'] else "Breakout Price: N/A")

                # Details
                details = buy.get('details', {})
                if 'rs_slope' in details:
                    output.append(f"RS Slope (3-week): {details['rs_slope']:.3f}")
                if 'volume_ratio' in details:
                    output.append(f"Volume vs Avg: {details['volume_ratio']:.1f}x")
                if 'distance_from_50sma' in buy['phase']:
                    phase_info = buy.get('phase_info', {})
                    output.append(f"Distance from 50 SMA: {phase_info.get('distance_from_50sma', 0):.1f}%")

                # Reasons
                output.append("\nReasons:")
                for reason in buy['reasons']:
                    output.append(f"  • {reason}")

                # Fundamental Snapshot
                if 'fundamental_snapshot' in buy:
                    output.append(buy['fundamental_snapshot'])

        else:
            output.append("\n✗ NO BUYS TODAY")
            reasons = results['signal_recommendation'].get('reasons', [])
            if reasons:
                output.append("\nReasons:")
                for reason in reasons:
                    output.append(f"  • {reason}")

        # Sell List
        output.append("\n" + "="*60)
        output.append("SELL LIST (Score >= 60)")
        output.append("="*60)

        if results['sells']:
            output.append(f"\nFound {len(results['sells'])} SELL signals:\n")

            for i, sell in enumerate(results['sells'], 1):
                output.append(f"\n{'#'*60}")
                output.append(f"SELL #{i}: {sell['ticker']} | Score: {sell['score']}/100 | Severity: {sell['severity'].upper()}")
                output.append(f"{'#'*60}")
                output.append(f"Phase: {sell['phase']}")
                output.append(f"Breakdown Level: ${sell['breakdown_level']:.2f}" if sell['breakdown_level'] else "Breakdown Level: N/A")

                # Details
                details = sell.get('details', {})
                if 'rs_slope' in details:
                    output.append(f"RS Rollover: {details['rs_slope']:.3f}")
                if 'volume_ratio' in details:
                    output.append(f"Volume vs Avg: {details['volume_ratio']:.1f}x")

                # Reasons
                output.append("\nReasons:")
                for reason in sell['reasons']:
                    output.append(f"  • {reason}")

        else:
            output.append("\n✗ NO SELLS TODAY")

        # Footer
        output.append("\n" + "="*60)
        output.append("END OF REPORT")
        output.append("="*60)

        report = "\n".join(output)
        logger.info("Screening complete")

        return report
