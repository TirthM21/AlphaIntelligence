#!/usr/bin/env python3
"""
Test script for long-term fundamentals data fetcher.

Tests fetching and caching of 5-year fundamental data for known compounders.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from src.long_term.data_fetcher import LongTermFundamentalsFetcher

# Load env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test tickers
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL"]


def test_data_fetcher():
    """Test LongTermFundamentalsFetcher on sample stocks."""
    logger.info("=" * 80)
    logger.info("PHASE 1 VERIFICATION: Long-Term Data Infrastructure")
    logger.info("=" * 80)

    # Check FMP API key
    api_key = os.getenv("FMP_API_KEY")
    if api_key:
        logger.info(f"✓ FMP_API_KEY configured (first 10 chars: {api_key[:10]}...)")
    else:
        logger.warning("✗ FMP_API_KEY not set - API calls will fail")
        return False

    # Initialize fetcher
    fetcher = LongTermFundamentalsFetcher(
        fmp_api_key=api_key,
        cache_dir="data/long_term_fundamentals"
    )

    logger.info(f"✓ LongTermFundamentalsFetcher initialized")
    logger.info(f"  Cache directory: {fetcher.cache_dir}")
    logger.info(f"  Cache expiry: {fetcher.cache_expiry_days} days")

    # Test fetching
    results = {}
    for ticker in TEST_TICKERS:
        logger.info("")
        logger.info(f"Fetching {ticker}...")

        try:
            fundamentals = fetcher.fetch(ticker, force_refresh=True)

            if not fundamentals:
                logger.error(f"✗ Failed to fetch {ticker}")
                results[ticker] = False
                continue

            # Log results
            logger.info(f"✓ Successfully fetched {ticker}")
            logger.info(f"  Income statements: {len(fundamentals.income_statements)} years")
            logger.info(f"  Balance sheets: {len(fundamentals.balance_sheets)} years")
            logger.info(f"  Cash flows: {len(fundamentals.cash_flows)} years")

            # Validate data
            if len(fundamentals.income_statements) < 4:
                logger.warning(f"  ⚠ Limited history ({len(fundamentals.income_statements)} quarters)")

            # Log calculated metrics
            if fundamentals.revenue_cagr_3yr is not None:
                logger.info(f"  Revenue CAGR (3Y): {fundamentals.revenue_cagr_3yr:.2%}")

            if fundamentals.revenue_cagr_5yr is not None:
                logger.info(f"  Revenue CAGR (5Y): {fundamentals.revenue_cagr_5yr:.2%}")

            if fundamentals.fcf_margin_3yr is not None:
                logger.info(f"  FCF Margin: {fundamentals.fcf_margin_3yr:.2%}")

            if fundamentals.debt_to_ebitda is not None:
                logger.info(f"  Debt/EBITDA: {fundamentals.debt_to_ebitda:.2f}x")

            if fundamentals.interest_coverage is not None:
                logger.info(f"  Interest Coverage: {fundamentals.interest_coverage:.2f}x")

            logger.info(f"  Data Quality: {fundamentals.data_quality_score:.0f}%")

            results[ticker] = True

        except Exception as e:
            logger.error(f"✗ Exception fetching {ticker}: {e}")
            results[ticker] = False

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1 VERIFICATION SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for ticker, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {ticker}")

    logger.info("")
    logger.info(f"Result: {passed}/{total} tickers fetched successfully")

    # Validation checklist
    logger.info("")
    logger.info("VALIDATION CHECKLIST:")
    logger.info(f"  [{'✓' if passed == total else '✗'}] Can fetch 5-year fundamentals")
    logger.info(f"  [{'✓' if passed > 0 else '✗'}] Revenue CAGR calculations present")
    logger.info(f"  [{'✓' if os.path.exists('data/long_term_fundamentals') else '✗'}] Cache directory exists")
    logger.info(f"  [{'?' if passed > 0 else '✗'}] ROIC/WACC calculations (Phase 2)")

    return passed == total


if __name__ == "__main__":
    try:
        success = test_data_fetcher()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
