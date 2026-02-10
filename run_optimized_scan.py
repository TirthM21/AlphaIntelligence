#!/usr/bin/env python3
"""Optimized full market scanner with parallel processing.

This version uses parallel workers to achieve 10-25 TPS safely while
avoiding rate limits through:
- Thread pool with 5 workers
- Per-worker rate limiting (0.2s = 5 TPS each)
- Adaptive backoff on errors
- Session pooling

Expected runtime: 15-30 minutes for 3,800+ stocks

Usage:
    python run_optimized_scan.py
    python run_optimized_scan.py --workers 10  # Faster but riskier
    python run_optimized_scan.py --conservative  # Slower but safer (3 workers)
"""

import argparse
import logging
import sys
import os
import time
import io
from datetime import datetime
from pathlib import Path

# Force UTF-8 encoding for stdout to prevent Unicode crashes on Windows consoles
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.data.universe_fetcher import USStockUniverseFetcher
from src.screening.optimized_batch_processor import OptimizedBatchProcessor
from src.screening.benchmark import (
    analyze_spy_trend,
    calculate_market_breadth,
    format_benchmark_summary,
    should_generate_signals
)
from src.screening.signal_engine import score_buy_signal, score_sell_signal
from src.data.fmp_fetcher import FMPFetcher
from src.data.finnhub_fetcher import FinnhubFetcher
from src.data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from src.reporting.newsletter_generator import NewsletterGenerator
from src.reporting.portfolio_manager import PortfolioManager
from src.notifications.email_notifier import EmailNotifier
from src.database.db_manager import DBManager
from src.data.sec_fetcher import SECFetcher
from src.ai.ai_agent import AIAgent
from src.long_term.etf_universe import ETFUniverse
from src.long_term.etf_engine import ETFEngine
from src.data.fetcher import YahooFinanceFetcher


log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_report(results, buy_signals, sell_signals, spy_analysis, breadth, filter_breakdown=None, output_dir="./data/daily_scans"):
    """Save comprehensive report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_str = datetime.now().strftime('%Y-%m-%d')

    output = []
    output.append("="*80)
    output.append("OPTIMIZED FULL MARKET SCAN - ALL US STOCKS")
    output.append(f"Scan Date: {date_str}")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("="*80)
    output.append("")

    # Stats
    output.append("SCANNING STATISTICS")
    output.append("-"*80)
    output.append(f"Total Universe: {results['total_processed']:,} stocks")
    output.append(f"Analyzed: {results['total_analyzed']:,} stocks")
    output.append(f"Processing Time: {results['processing_time_seconds']/60:.1f} minutes")
    output.append(f"Actual TPS: {results['actual_tps']:.2f}")

    if filter_breakdown:
        output.append("\nFILTER BREAKDOWN (Why stocks were skipped)")
        output.append("-"*80)
        sorted_filters = sorted(filter_breakdown.items(), key=lambda x: x[1], reverse=True)
        total_filtered = results.get('filtered_count', 0)
        for reason, count in sorted_filters:
            pct = (count / total_filtered * 100) if total_filtered > 0 else 0
            output.append(f"• {reason:25}: {count:5} ({pct:.1f}%)")
        output.append("-" * 80)

    error_rate = results['error_rate'] * 100
    if error_rate < 1:
        error_emoji = "🟢"
    elif error_rate < 5:
        error_emoji = "🟡"
    else:
        error_emoji = "🔴"
    output.append(f"{error_emoji} Error Rate: {error_rate:.2f}%")

    # Buy/Sell signal counts with emoji
    if len(buy_signals) > 0:
        output.append(f"🟢 Buy Signals: {len(buy_signals)}")
    else:
        output.append(f"Buy Signals: {len(buy_signals)}")

    if len(sell_signals) > 0:
        output.append(f"🔴 Sell Signals: {len(sell_signals)}")
    else:
        output.append(f"Sell Signals: {len(sell_signals)}")
    output.append("")

    # Benchmark
    output.append(format_benchmark_summary(spy_analysis, breadth))
    output.append("")

    # Buy signals
    output.append("="*80)
    output.append(f"🟢 TOP BUY SIGNALS (Score >= 70) - {len(buy_signals)} Total")
    output.append("="*80)
    output.append("")

    if buy_signals:
        for i, signal in enumerate(buy_signals[:50], 1):
            score = signal['score']
            # Score-based emoji (green/yellow with star for exceptional)
            if score >= 90:
                score_emoji = "⭐"  # Exceptional - star
            elif score >= 80:
                score_emoji = "🟢"  # Very good - green
            elif score >= 70:
                score_emoji = "🟢"  # Good - green
            else:
                score_emoji = "🟡"  # Borderline - yellow

            output.append(f"\n{'#'*80}")
            output.append(f"{score_emoji} BUY #{i}: {signal['ticker']} | Score: {signal['score']}/110")
            output.append(f"{'#'*80}")
            output.append(f"Phase: {signal['phase']}")

            # Entry quality with emoji
            entry_quality = signal.get('entry_quality', 'Unknown')
            if entry_quality == 'Good':
                output.append(f"🟢 Entry Quality: {entry_quality}")
            elif entry_quality == 'Extended':
                output.append(f"🟡 Entry Quality: {entry_quality}")
            else:
                output.append(f"🔴 Entry Quality: {entry_quality}")

            # CRITICAL: Stop loss and R/R ratio
            if signal.get('stop_loss'):
                output.append(f"Stop Loss: ${signal['stop_loss']:.2f}")
                details = signal.get('details', {})
                risk_amt = details.get('risk_amount', 0)
                reward_amt = details.get('reward_amount', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                # R/R ratio emoji
                if rr_ratio >= 3:
                    rr_emoji = "🟢"  # Excellent R/R
                elif rr_ratio >= 2:
                    rr_emoji = "🟢"  # Good R/R
                else:
                    rr_emoji = "🟡"  # Poor R/R
                output.append(f"{rr_emoji} Risk/Reward: {rr_ratio:.1f}:1 (Risk ${risk_amt:.2f}, Reward ${reward_amt:.2f})")

            if signal.get('breakout_price'):
                output.append(f"Breakout: ${signal['breakout_price']:.2f}")

            details = signal.get('details', {})
            if 'rs_slope' in details:
                rs_slope = details['rs_slope']
                # RS emoji (green = good, yellow = ok, red = bad)
                if rs_slope > 0.5:
                    rs_emoji = "🟢"  # Strong RS
                elif rs_slope > 0:
                    rs_emoji = "🟡"  # Positive RS
                else:
                    rs_emoji = "🔴"  # Weak RS
                output.append(f"{rs_emoji} RS: {rs_slope:.3f}")
            if 'volume_ratio' in details:
                vol_ratio = details['volume_ratio']
                # Volume emoji
                if vol_ratio > 1.5:
                    vol_emoji = "🟢"  # High volume
                elif vol_ratio > 1.0:
                    vol_emoji = "🟡"  # Above average
                else:
                    vol_emoji = "🔴"  # Low volume
                output.append(f"{vol_emoji} Volume: {vol_ratio:.1f}x")

            # VCP pattern details if detected
            vcp_data = details.get('vcp_data')
            if vcp_data:
                vcp_quality = vcp_data.get('quality', 0)
                contractions = vcp_data.get('contractions', 0)
                pattern = vcp_data.get('pattern', 'N/A')

                if vcp_quality >= 80:
                    vcp_emoji = "⭐"  # Exceptional VCP
                elif vcp_quality >= 60:
                    vcp_emoji = "🟢"  # Good VCP
                elif vcp_quality >= 50:
                    vcp_emoji = "🟡"  # Marginal VCP
                else:
                    vcp_emoji = "🟡"  # Partial pattern

                if vcp_quality >= 50:
                    output.append(f"{vcp_emoji} VCP: {pattern} (quality: {vcp_quality:.0f}/100)")

            output.append("\nKey Reasons:")
            for reason in signal['reasons'][:7]:  # Show 7 instead of 5
                output.append(f"  • {reason}")

            if signal.get('fundamental_snapshot'):
                output.append(signal['fundamental_snapshot'])

        if len(buy_signals) > 50:
            output.append(f"\n{'='*80}")
            output.append(f"ADDITIONAL BUYS ({len(buy_signals)-50} more)")
            output.append(f"{'='*80}\n")
            remaining = [s['ticker'] for s in buy_signals[50:]]
            for i in range(0, len(remaining), 10):
                output.append(", ".join(remaining[i:i+10]))
    else:
        output.append("✗ NO BUY SIGNALS TODAY")

    # Sell signals
    output.append(f"\n\n{'='*80}")
    output.append(f"🔴 TOP SELL SIGNALS (Score >= 60) - {len(sell_signals)} Total")
    output.append(f"{'='*80}")
    output.append("")

    if sell_signals:
        for i, signal in enumerate(sell_signals[:30], 1):
            score = signal['score']
            severity = signal['severity']

            # Severity emoji (red/yellow with alarm for critical)
            if severity == 'critical':
                severity_emoji = "🚨"  # Critical - alarm
            elif severity == 'high':
                severity_emoji = "🔴"  # High - red
            else:
                severity_emoji = "🟡"  # Moderate - yellow

            # Score emoji (higher score = more urgent to sell)
            if score >= 80:
                score_emoji = "🚨"  # Very urgent - alarm
            elif score >= 70:
                score_emoji = "🔴"  # Urgent - red
            else:
                score_emoji = "🟡"  # Warning - yellow

            output.append(f"\n{'#'*80}")
            output.append(f"{score_emoji} SELL #{i}: {signal['ticker']} | Score: {signal['score']}/110")
            output.append(f"{'#'*80}")
            output.append(f"Phase: {signal['phase']} | {severity_emoji} Severity: {severity.upper()}")
            if signal.get('breakdown_level'):
                output.append(f"Breakdown: ${signal['breakdown_level']:.2f}")
            details = signal.get('details', {})
            if 'rs_slope' in details:
                rs_slope = details['rs_slope']
                # RS emoji for sell signals (negative is expected)
                if rs_slope < -0.5:
                    rs_emoji = "🔴"  # Very weak RS
                elif rs_slope < 0:
                    rs_emoji = "🟡"  # Weak RS
                else:
                    rs_emoji = "🟢"  # Still positive RS (unusual for sell)
                output.append(f"{rs_emoji} RS: {rs_slope:.3f}")
            output.append("\nSell Reasons:")
            for reason in signal['reasons'][:5]:
                output.append(f"  • {reason}")

            if signal.get('fundamental_snapshot'):
                output.append(signal['fundamental_snapshot'])

        if len(sell_signals) > 30:
            output.append(f"\n{'='*80}")
            output.append(f"ADDITIONAL SELLS ({len(sell_signals)-30} more)")
            output.append(f"{'='*80}\n")
            remaining = [s['ticker'] for s in sell_signals[30:]]
            for i in range(0, len(remaining), 10):
                output.append(", ".join(remaining[i:i+10]))
    else:
        output.append("✗ NO SELL SIGNALS TODAY")

    output.append(f"\n\n{'='*80}")
    output.append("END OF SCAN")
    output.append(f"{'='*80}\n")

    report_text = "\n".join(output)

    # Save
    filepath = Path(output_dir) / f"optimized_scan_{timestamp}.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_text)

    latest_path = Path(output_dir) / "latest_optimized_scan.txt"
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"Full report saved to {filepath}")


def run_etf_scan():
    """Run thematic ETF discovery and scoring."""
    logger.info("=" * 60)
    logger.info("THEMATIC ETF ANALYSIS")
    logger.info("=" * 60)

    try:
        universe = ETFUniverse()
        engine = ETFEngine(universe=universe)
        fetcher = YahooFinanceFetcher()

        etfs = universe.discover_thematic_etfs()
        etfs = universe.filter_by_quality(etfs)

        scored_etfs = []

        # Fetch SPY returns for benchmark
        logger.info("Fetching SPY benchmark for ETF scoring...")
        spy_hist = fetcher.fetch_price_history('SPY', period='5y')
        if spy_hist.empty:
            logger.warning("SPY history unavailable for ETF scoring")
            return []

        current_spy = spy_hist['Close'].iloc[-1]
        spy_1y = spy_hist['Close'].iloc[-252] if len(spy_hist) > 252 else spy_hist['Close'].iloc[0]
        spy_3y = spy_hist['Close'].iloc[-756] if len(spy_hist) > 756 else spy_hist['Close'].iloc[0]
        spy_5y = spy_hist['Close'].iloc[0]

        spy_returns = {
            'spy_return_1yr': (current_spy / spy_1y) - 1,
            'spy_return_3yr': (current_spy / spy_3y) ** (1 / 3) - 1 if spy_3y > 0 else 0,
            'spy_return_5yr': (current_spy / spy_5y) ** (1 / 5) - 1 if spy_5y > 0 else 0
        }

        logger.info(f"Scoring {len(etfs)} thematic ETFs...")
        for etf in etfs:
            try:
                hist = fetcher.fetch_price_history(etf.ticker, period='5y')
                if hist.empty:
                    continue

                current_price = hist['Close'].iloc[-1]
                price_1y = hist['Close'].iloc[-252] if len(hist) > 252 else hist['Close'].iloc[0]
                price_3y = hist['Close'].iloc[-756] if len(hist) > 756 else hist['Close'].iloc[0]
                price_5y = hist['Close'].iloc[0]

                price_data = {
                    'return_1yr': (current_price / price_1y) - 1 if price_1y > 0 else 0,
                    'return_3yr': (current_price / price_3y) ** (1 / 3) - 1 if price_3y > 0 else 0,
                    'return_5yr': (current_price / price_5y) ** (1 / 5) - 1 if price_5y > 0 else 0,
                    **spy_returns
                }

                score = engine.score_etf(etf.__dict__, price_data)
                if score:
                    scored_etfs.append(score)
            except Exception as e:
                logger.debug(f"Failed to score ETF {etf.ticker}: {e}")

        # Rank by score
        logger.info(f"✓ Ranked {len(scored_etfs)} ETFs")
        return sorted([s.__dict__ for s in scored_etfs], key=lambda x: x['total_score'], reverse=True)

    except Exception as e:
        logger.error(f"ETF Scan failed: {e}")
        return []

    print(report_text)

    return filepath


def main():
    parser = argparse.ArgumentParser(description='Optimized Full Market Scanner')
    parser.add_argument('--workers', type=int, default=3, help='Parallel workers (default: 3)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay per worker (default: 0.5s)')
    parser.add_argument('--conservative', action='store_true', help='Ultra-conservative mode (2 workers, 1.0s delay)')
    parser.add_argument('--aggressive', action='store_true', help='Faster mode (5 workers, 0.3s delay) - MAY HIT RATE LIMITS!')
    parser.add_argument('--resume', action='store_true', help='Resume from progress')
    parser.add_argument('--clear-progress', action='store_true', help='Clear progress')
    parser.add_argument('--limit', type=int, help='Limit scan to first N stocks (e.g. 50)')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of specific tickers (e.g. AAPL,MSFT,TSLA)')
    parser.add_argument('--test-mode', action='store_true', help='Test with 100 stocks')
    parser.add_argument('--min-price', type=float, default=5.0, help='Min price')
    parser.add_argument('--max-price', type=float, default=10000.0, help='Max price')
    parser.add_argument('--min-volume', '--min-vol', type=int, default=100000, help='Min volume')
    parser.add_argument('--max-drawdown', type=float, default=0.70, help='Max 5y drawdown limit (0.7 = 70%)')
    parser.add_argument('--use-fmp', action='store_true', help='Use FMP for enhanced fundamentals on buy signals')
    parser.add_argument('--git-storage', action='store_true', help='Use Git-based storage for fundamentals (recommended)')
    parser.add_argument('--ai', action='store_true', help='Enable AI analysis for top buy signals')
    parser.add_argument('--download-sec', action='store_true', help='Download SEC 10-Qs for top buy signals (requires sec-edgar-toolkit)')
    parser.add_argument('--send-email', action='store_true', help='Send newsletter via email (requires EMAIL_SENDER and EMAIL_PASSWORD env vars)')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostic check for API keys and SEC access')
    parser.add_argument('--broad', action='store_true', help='Broad Scan: lower price ($2) and volume (20k) thresholds')
    parser.add_argument('--slow-api', action='store_true', help='Use 5s rate limit for FMP and Finnhub APIs')

    args = parser.parse_args()

    # Presets
    if args.conservative:
        args.workers = 2
        args.delay = 1.0
        logger.info("Ultra-conservative mode: 2 workers, 1.0s delay (~2 TPS)")
    elif args.aggressive:
        args.workers = 5
        args.delay = 0.3
        logger.warning("Aggressive mode: 5 workers, 0.3s delay (~17 TPS) - MAY HIT RATE LIMITS!")

    if args.broad:
        args.min_price = 2.0
        args.min_volume = 20000
        args.max_drawdown = 0.85
        logger.info("Broad Scan enabled: Price >$2, Volume >20k, Max Drawdown 85%")

    effective_tps = args.workers / args.delay
    logger.info(f"Configuration: {args.workers} workers x {1/args.delay:.1f} TPS = ~{effective_tps:.1f} TPS effective")

    # Diagnostics mode
    if args.diagnostics:
        logger.info("="*60)
        logger.info("DIAGNOSTIC CHECK")
        logger.info("="*60)
        
        # 1. Check FMP
        fmp_key = os.getenv('FMP_API_KEY')
        if not fmp_key:
            logger.error("✗ FMP_API_KEY not found in .env")
        if fmp_key:
            f = FMPFetcher(api_key=fmp_key)
            test_data = f.fetch_income_statement("AAPL", limit=1)
            if test_data:
                logger.info("✓ FMP API: Working correctly")
            else:
                logger.error("✗ FMP API: Key found but failed to fetch data (Check tier/limits)")
        
        # 2. Check SEC
        try:
            s = SECFetcher(download_dir="./data/test_sec")
            logger.info("✓ SEC Fetcher: Module loaded")
        except Exception as e:
            logger.error(f"✗ SEC Fetcher: Failed to initialize: {e}")
            
        # 3. Check Email
        n = EmailNotifier()
        if n.enabled:
            logger.info(f"✓ Email: Configured (Sender: {n.sender_email})")
        else:
            logger.warning("• Email: Not configured (Optional)")
            
        logger.info("="*60)
        sys.exit(0)

    # Initialize fetchers
    fmp_fetcher = FMPFetcher()
    finnhub_fetcher = FinnhubFetcher()
    
    # Apply slow-api limit if requested
    if args.slow_api:
        logger.info("🐢 Slow API mode enabled: Setting 5s delay for FMP and Finnhub")
        fmp_fetcher.rate_limit_delay = 5.0
        finnhub_fetcher.min_delay = 5.0
        
    enhanced_fetcher = EnhancedFundamentalsFetcher(fmp_fetcher=fmp_fetcher, finnhub_fetcher=finnhub_fetcher)
    if args.use_fmp and enhanced_fetcher.fmp_available:
        logger.info("FMP enabled - will use for buy signal fundamentals (DCF + Insider + Margins)")
    elif args.use_fmp:
        logger.warning("--use-fmp specified but FMP_API_KEY not set. Using yfinance only.")

    try:
        # Fetch universe
        universe_fetcher = USStockUniverseFetcher()
        logger.info("Fetching stock universe...")
        tickers = universe_fetcher.fetch_universe()

        if not tickers:
            logger.error("Failed to fetch universe")
            sys.exit(1)

        logger.info(f"Universe: {len(tickers):,} stocks")

        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(',')]
            logger.info(f"CUSTOM TICKER LIST: {len(tickers)} stocks")
        elif args.limit:
            tickers = tickers[:args.limit]
            logger.info(f"LIMITED MODE: {len(tickers)} stocks")
        elif args.test_mode:
            tickers = tickers[:100]
            logger.info(f"TEST MODE: {len(tickers)} stocks")

        # Initialize processor
        processor = OptimizedBatchProcessor(
            max_workers=args.workers,
            rate_limit_delay=args.delay,
            use_git_storage=args.git_storage,
            use_fmp=args.use_fmp,
            max_drawdown=args.max_drawdown
        )

        if args.git_storage:
            logger.info("Git-based fundamental storage enabled - 74% API call reduction!")

        if args.use_fmp:
            logger.info("FMP API integration enabled for high-fidelity fundamentals.")
        else:
            logger.info("Using standard yfinance fundamentals (FMP disabled).")

        if args.download_sec:
            logger.info("SEC Filing verification/download enabled for top signals.")

        if args.clear_progress:
            processor.clear_progress()

        # Process
        results = processor.process_batch_parallel(
            tickers,
            resume=args.resume,
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume
        )

        if 'error' in results:
            logger.error(results['error'])
            sys.exit(1)

        # Analysis
        logger.info("Generating signals...")
        spy_analysis = analyze_spy_trend(processor.spy_data, processor.spy_price)
        breadth = calculate_market_breadth(results['phase_results'])
        signal_rec = should_generate_signals(spy_analysis, breadth)

        # Buy signals
        buy_signals = []
        if signal_rec['should_generate_buys']:
            preliminary_buys = []
            for analysis in results['analyses']:
                if analysis['phase_info']['phase'] in [1, 2]:
                    # Quick preliminary score
                    signal = score_buy_signal(
                        ticker=analysis['ticker'],
                        price_data=analysis['price_data'],
                        current_price=analysis['current_price'],
                        phase_info=analysis['phase_info'],
                        rs_series=analysis['rs_series'],
                        fundamentals=analysis.get('quarterly_data'),
                        vcp_data=analysis.get('vcp_data')
                    )
                    if signal['is_buy']:
                        preliminary_buys.append((analysis, signal))

            # Sort and take top 15 for premium enrichment
            preliminary_buys = sorted(preliminary_buys, key=lambda x: x[1]['score'], reverse=True)[:15]
            
            ai_agent = AIAgent()
            
            logger.info(f"Enriching top {len(preliminary_buys)} candidates with SEC and AI analysis...")
            for analysis, signal in preliminary_buys:
                ticker = analysis['ticker']
                
                # 1. SEC Confirmation
                sec_status = "Not requested"
                if args.download_sec:
                    sec_status = enhanced_fetcher.download_sec_filing(ticker, '10-Q')
                
                # 2. AI Assessment
                ai_commentary = None
                if args.ai and ai_agent.api_key:
                    # Enrich with Finnhub data if available
                    finnhub_data = {}
                    if finnhub_fetcher.api_key:
                        sentiment = finnhub_fetcher.get_news_sentiment(ticker)
                        insider = finnhub_fetcher.get_insider_sentiment(ticker)
                        targets = finnhub_fetcher.get_price_target(ticker)
                        if sentiment: finnhub_data['news_sentiment'] = sentiment
                        if insider: finnhub_data['insider_sentiment'] = insider[0] if insider else None
                        if targets: finnhub_data['analyst_targets'] = targets
                    
                    ai_commentary = ai_agent.generate_commentary(ticker, {
                        "price": analysis['current_price'],
                        "technical_score": signal['score'],
                        "reasons": signal['reasons'],
                        "fundamentals": signal['details'].get('fundamental_analysis', {}),
                        "finnhub_enrichment": finnhub_data
                    })
                    signal['ai_commentary'] = ai_commentary
                
                # 3. Final Re-Score (The "Mixture")
                final_signal = score_buy_signal(
                    ticker=ticker,
                    price_data=analysis['price_data'],
                    current_price=analysis['current_price'],
                    phase_info=analysis['phase_info'],
                    rs_series=analysis['rs_series'],
                    fundamentals=analysis.get('quarterly_data'),
                    vcp_data=analysis.get('vcp_data'),
                    sec_status=sec_status,
                    premium_commentary=ai_commentary
                )
                
                # Attach extras for the report
                final_signal['fundamental_snapshot'] = fundamentals_fetcher.create_snapshot(
                    ticker,
                    quarterly_data=analysis.get('quarterly_data', {}),
                    use_fmp=args.use_fmp
                )
                final_signal['ai_commentary'] = ai_commentary
                final_signal['sec_status'] = sec_status
                
                buy_signals.append(final_signal)

        # Save buy signals for dashboard
        try:
            signals_to_save = []
            for s in buy_signals:
                signals_to_save.append({
                    'ticker': s['ticker'],
                    'score': s['score'],
                    'price': s.get('current_price') or s.get('breakout_price'),
                    'stop_loss': s.get('stop_loss'),
                    'reasons': s.get('reasons', []),
                    'phase': s.get('phase'),
                    'sector': s.get('sector', 'Unknown')
                })
            
            os.makedirs("data", exist_ok=True)
            with open("data/latest_market_signals.json", "w") as f:
                json.dump(signals_to_save, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save market signals JSON: {e}")

        # Sell signals
        sell_signals = []
        if signal_rec['should_generate_sells']:
            for analysis in results['analyses']:
                if analysis['phase_info']['phase'] in [3, 4]:
                    signal = score_sell_signal(
                        ticker=analysis['ticker'],
                        price_data=analysis['price_data'],
                        current_price=analysis['current_price'],
                        phase_info=analysis['phase_info'],
                        rs_series=analysis['rs_series'],
                        fundamentals=analysis.get('quarterly_data')  # Pass raw quarterly data, not analyzed
                    )
                    if signal['is_sell']:
                        # Add fundamental snapshot
                        signal['fundamental_snapshot'] = fundamentals_fetcher.create_snapshot(
                            analysis['ticker'],
                            quarterly_data=analysis.get('quarterly_data', {}),
                            use_fmp=args.use_fmp
                        )
                        sell_signals.append(signal)

        sell_signals = sorted(sell_signals, key=lambda x: x['score'], reverse=True)

        # Report
        save_report(results, buy_signals, sell_signals, spy_analysis, breadth, filter_breakdown=results.get('filter_reasons'))

        # Record Recommendations & Generate Portfolio Reports
        try:
            logger.info("Recording signals and generating portfolio management reports...")
            db = DBManager()
            # Record current signals for historical alpha tracking
            # Combine buys and sells for the database record
            all_signals = buy_signals + sell_signals
            db.record_recommendations(all_signals, spy_price=processor.spy_price)
            
            # --- HEDGE FUND PORTFOLIO ENHANCEMENT ---
            logger.info("Updating Hedge Fund Portfolio holdings and tracking performance...")
            db.update_portfolio_from_signals(buy_signals)
            db.update_daily_performance(spy_price=processor.spy_price)
            # ----------------------------------------
            
            # Generate advanced reports (Allocation, Rebalance, Alpha Tracker)
            pm = PortfolioManager()
            pm.generate_reports(buy_signals, sell_signals)
        except Exception as pm_err:
            logger.error(f"Failed to record signals or generate portfolio reports: {pm_err}")

        # Generate Newsletter
        try:
            logger.info("Generating daily newsletter...")
            newsletter_gen = NewsletterGenerator()
            
            # Prepare status dict
            market_status = {
                'spy': spy_analysis,
                'breadth': breadth
            }
            
            # Optional: Run ETF Scan
            top_etfs = []
            try:
                top_etfs = run_etf_scan()
            except Exception as etf_err:
                logger.error(f"ETF Scan integration failed: {etf_err}")

            newsletter_path = newsletter_gen.generate_newsletter(
                market_status=market_status,
                top_buys=buy_signals,
                top_sells=sell_signals,
                top_etfs=top_etfs
            )
            logger.info(f"Newsletter ready: {newsletter_path}")
            
            # Send Email
            if args.send_email:
                try:
                    logger.info("Preparing for premium email delivery...")
                    
                    # Get subscribers from DB
                    db_subscribers = []
                    try:
                        db = DBManager()
                        db_subscribers = db.get_active_subscribers()
                    except Exception as db_err:
                        logger.warning(f"Could not fetch subscribers from DB: {db_err}")
                    
                    # Always include the default recipient from env
                    primary_recipient = os.getenv('EMAIL_RECIPIENT') or os.getenv('EMAIL_SENDER')
                    
                    subscribers = list(set(db_subscribers + ([primary_recipient] if primary_recipient else [])))
                    
                    if not subscribers:
                        logger.warning("No recipients found (DB empty and EMAIL_RECIPIENT not set).")
                    else:
                        logger.info(f"Sending newsletter to {len(subscribers)} recipients...")
                        notifier = EmailNotifier()
                        
                        # Use latest_optimized_scan.txt as attachment if it exists
                        latest_report = Path("./data/daily_scans/latest_optimized_scan.txt")
                        report_to_attach = str(latest_report) if latest_report.exists() else None
                        
                        success_count = 0
                        for email in subscribers:
                            try:
                                # Overwrite recipient for this call
                                notifier.recipient_email = email
                                if notifier.send_newsletter(
                                    newsletter_path=newsletter_path,
                                    scan_report_path=report_to_attach
                                ):
                                    success_count += 1
                                    # Anti-spam delay for bulk
                                    import time
                                    time.sleep(0.5)
                            except Exception as e:
                                logger.error(f"Failed to send to {email}: {e}")
                        
                        logger.info(f"✅ Bulk delivery complete: {success_count}/{len(subscribers)} successful.")
                except Exception as email_err:
                    logger.error(f"Failed to send newsletter email: {email_err}")
            
            # Print preview
            print("\\n" + "="*60) 
            print("DAILY NEWSLETTER PREVIEW")
            print("="*60)
            with open(newsletter_path, 'r', encoding='utf-8') as f:
                # Print first 20 lines
                print("".join(f.readlines()[:20]))
            print("...\\n(See full file for more)")
            
        except Exception as e:
            logger.error(f"Failed to generate newsletter: {e}")

        # NEW: Generate AI Deep-Dive Intelligence Report
        try:
            from run_ai_report import generate_deep_dive
            logger.info("Generating AI Deep-Dive Intelligence Report...")
            generate_deep_dive()
        except Exception as ai_report_err:
            logger.error(f"Failed to generate AI Deep-Dive Report: {ai_report_err}")

        # Show FMP usage if enabled
        if args.use_fmp:
            usage = fundamentals_fetcher.get_api_usage()
            logger.info("="*60)
            logger.info("FMP API USAGE")
            logger.info(f"Calls used: {usage['fmp_calls_used']}/{usage['fmp_daily_limit']}")
            logger.info(f"Calls remaining: {usage['fmp_calls_remaining']}")
            if 'bandwidth_used_mb' in usage:
                logger.info(f"Bandwidth used: {usage['bandwidth_used_mb']:.1f} MB / {usage['bandwidth_limit_gb']:.1f} GB ({usage['bandwidth_pct_used']:.1f}%)")
                logger.info(f"Earnings season: {'Yes' if usage['is_earnings_season'] else 'No'} (cache: {usage['cache_hours']}h)")
            logger.info("="*60)

        logger.info("="*60)
        logger.info("SCAN COMPLETE")
        logger.info(f"Time: {results['processing_time_seconds']/60:.1f} minutes")
        logger.info(f"Actual TPS: {results['actual_tps']:.2f}")
        logger.info(f"Buy signals: {len(buy_signals)}")
        logger.info(f"Sell signals: {len(sell_signals)}")
        if args.download_sec:
            logger.info(f"SEC Filings: Downloaded for top {min(len(buy_signals), 10)} buys")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("\nInterrupted - progress saved")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        # Send error alert if email configured
        try:
            import traceback
            notifier = EmailNotifier()
            if notifier.enabled:
                notifier.send_error_alert(
                    error_message=str(e),
                    error_details=traceback.format_exc()
                )
        except Exception as alert_err:
            logger.error(f"Failed to send error alert: {alert_err}")
        sys.exit(1)


if __name__ == '__main__':
    main()
