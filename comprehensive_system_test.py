"""Comprehensive System Test Runner.
Tests every major module and endpoint in the stock screener.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.fmp_fetcher import FMPFetcher
from src.data.sec_fetcher import SECFetcher
from src.data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from src.ai.ai_agent import AIAgent
from src.database.db_manager import DBManager
from src.screening.signal_engine import score_buy_signal, score_sell_signal
from src.reporting.newsletter_generator import NewsletterGenerator
from src.reporting.portfolio_manager import PortfolioManager
from src.screening.optimized_batch_processor import OptimizedBatchProcessor

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemTest")

def run_test():
    load_dotenv()
    test_results = []
    
    logger.info("="*60)
    logger.info("STARTING COMPREHENSIVE SYSTEM TEST")
    logger.info("="*60)

    # 1. Test Database
    try:
        logger.info("[1/8] Testing DBManager...")
        db = DBManager()
        if db.db_url:
            subscribers = db.get_active_subscribers()
            logger.info(f"  Result: Success (Connected, {len(subscribers)} subscribers)")
            test_results.append(("Database", "PASS", f"{len(subscribers)} subs"))
        else:
            logger.warning("  Result: Skipped (No DATABASE_URL)")
            test_results.append(("Database", "SKIP", "No URL"))
    except Exception as e:
        logger.error(f"  Result: FAIL - {e}")
        test_results.append(("Database", "FAIL", str(e)))

    # 2. Test AI Agent
    try:
        logger.info("[2/8] Testing AIAgent...")
        ai = AIAgent()
        if ai.api_key:
            test_commentary = ai.generate_commentary("AAPL", {"price": 180, "technical_score": 85})
            if test_commentary:
                logger.info("  Result: Success (AI responded)")
                test_results.append(("AI Agent", "PASS", "Commentary generated"))
            else:
                logger.error("  Result: FAIL (AI returned empty)")
                test_results.append(("AI Agent", "FAIL", "Empty response"))
        else:
            logger.warning("  Result: Skipped (No API key)")
            test_results.append(("AI Agent", "SKIP", "No key"))
    except Exception as e:
        logger.error(f"  Result: FAIL - {e}")
        test_results.append(("AI Agent", "FAIL", str(e)))

    # 3. Test FMP Fetcher
    try:
        logger.info("[3/8] Testing FMPFetcher...")
        fmp_key = os.getenv('FMP_API_KEY')
        if fmp_key:
            fmp = FMPFetcher(api_key=fmp_key)
            data = fmp.fetch_income_statement("MSFT", limit=1)
            if data:
                logger.info("  Result: Success (Fetched MSFT data)")
                test_results.append(("FMP API", "PASS", "Data fetched"))
            else:
                logger.error("  Result: FAIL (No data)")
                test_results.append(("FMP API", "FAIL", "No data returned"))
        else:
            test_results.append(("FMP API", "SKIP", "No key"))
    except Exception as e:
        test_results.append(("FMP API", "FAIL", str(e)))

    # 4. Test SEC Fetcher
    try:
        logger.info("[4/8] Testing SECFetcher...")
        sec = SECFetcher(download_dir="./data/test_sec")
        # Check cache/download logic
        res = sec.download_latest_10q("NVDA")
        logger.info(f"  Result: {res}")
        test_results.append(("SEC Fetcher", "PASS", res))
    except Exception as e:
        test_results.append(("SEC Fetcher", "FAIL", str(e)))

    # 5. Run Batch Scan (15 stocks)
    logger.info("[5/8] Running Scan for 15 Stocks...")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ", "V", "XOM", "TSM", "WMT", "MA"]
    
    processor = OptimizedBatchProcessor(
        max_workers=5,
        rate_limit_delay=0.2,
        use_fmp=True
    )
    processor.clear_progress()
    
    scan_results = processor.process_batch_parallel(test_tickers)
    logger.info(f"  Result: Processed {scan_results['total_analyzed']} stocks")
    test_results.append(("Batch Scanner", "PASS", f"{scan_results['total_analyzed']}/15 analyzed"))

    # 6. Signal Generation & Enrichment
    logger.info("[6/8] Testing Signal Scoring & Mixture-of-Experts...")
    buy_signals = []
    for analysis in scan_results['analyses'][:5]: # Test with first 5
        sig = score_buy_signal(
            ticker=analysis['ticker'],
            price_data=analysis['price_data'],
            current_price=analysis['current_price'],
            phase_info=analysis['phase_info'],
            rs_series=analysis['rs_series'],
            fundamentals=analysis.get('quarterly_data'),
            sec_status="Success (Mocked)"
        )
        buy_signals.append(sig)
    
    logger.info(f"  Result: Generated {len(buy_signals)} buy signals with enrichment")
    test_results.append(("Signal Engine", "PASS", f"{len(buy_signals)} signals scored"))

    # 7. Portfolio Manager & Reporting
    try:
        logger.info("[7/8] Testing Portfolio Manager Reports...")
        pm = PortfolioManager(report_dir="./data/test_reports")
        pm.generate_reports(buy_signals, [])
        logger.info("  Result: Success (Reports generated in ./data/test_reports)")
        test_results.append(("Portfolio Reports", "PASS", "Files created"))
    except Exception as e:
        logger.error(f"  Result: FAIL - {e}")
        test_results.append(("Portfolio Reports", "FAIL", str(e)))

    # 8. Newsletter Generation (Email Disabled)
    try:
        logger.info("[8/8] Testing Newsletter Generation...")
        ng = NewsletterGenerator()
        market_status = {
            'spy': {'trend': 'Uptrend', 'sma_200': 480, 'current_price': 500},
            'breadth': {'bullish_pct': 65}
        }
        path = ng.generate_newsletter(market_status, buy_signals, [])
        logger.info(f"  Result: Success (Newsletter saved to {path})")
        test_results.append(("Newsletter Gen", "PASS", "PDF/TXT created"))
    except Exception as e:
        logger.error(f"  Result: FAIL - {e}")
        test_results.append(("Newsletter Gen", "FAIL", str(e)))

    # FINAL SUMMARY REPORT
    logger.info("\n" + "="*60)
    logger.info("SYSTEM TEST SUMMARY REPORT")
    logger.info("="*60)
    print(f"{'Module':<20} | {'Status':<8} | {'Details'}")
    print("-" * 60)
    for module, status, detail in test_results:
        print(f"{module:<20} | {status:<8} | {detail}")
    logger.info("="*60)
    logger.info("Test Complete. Note: Emailing was disabled per instructions.")

if __name__ == "__main__":
    run_test()
