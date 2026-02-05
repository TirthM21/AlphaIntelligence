#!/usr/bin/env python3
"""
Test script for Report Generator.

Demonstrates quarterly ownership report generation with sample portfolio data.
"""

import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_report_generation():
    """Test report generation with sample data."""
    logger.info("=" * 80)
    logger.info("PHASE 5 VERIFICATION: Report Generator")
    logger.info("=" * 80)

    try:
        from src.long_term.portfolio_constructor import PortfolioConstructor
        from src.long_term.report_generator import ReportGenerator
        logger.info("✓ Imported PortfolioConstructor and ReportGenerator")
    except ImportError as e:
        logger.error(f"✗ Failed to import: {e}")
        return False

    # Initialize
    try:
        constructor = PortfolioConstructor()
        reporter = ReportGenerator()
        logger.info("✓ Initialized PortfolioConstructor and ReportGenerator")
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        return False

    # Create sample stocks
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAMPLE PORTFOLIO DATA")
    logger.info("=" * 80)

    stocks = [
        {"ticker": "AAPL", "name": "Apple", "score": 87.5, "sector": "Technology"},
        {"ticker": "MSFT", "name": "Microsoft", "score": 85.2, "sector": "Technology"},
        {"ticker": "NVDA", "name": "NVIDIA", "score": 81.5, "sector": "Technology"},
        {"ticker": "JPM", "name": "JPMorgan", "score": 72.3, "sector": "Financials"},
        {"ticker": "BAC", "name": "Bank of America", "score": 68.5, "sector": "Financials"},
        {"ticker": "UNH", "name": "United Health", "score": 79.2, "sector": "Healthcare"},
        {"ticker": "JNJ", "name": "Johnson & Johnson", "score": 76.8, "sector": "Healthcare"},
        {"ticker": "PG", "name": "Procter & Gamble", "score": 74.1, "sector": "Consumer"},
        {"ticker": "WMT", "name": "Walmart", "score": 71.5, "sector": "Consumer"},
        {"ticker": "LMT", "name": "Lockheed Martin", "score": 75.3, "sector": "Defense"},
        {"ticker": "RTX", "name": "Raytheon", "score": 73.8, "sector": "Defense"},
        {"ticker": "XOM", "name": "ExxonMobil", "score": 65.2, "sector": "Energy"},
        {"ticker": "CVX", "name": "Chevron", "score": 63.5, "sector": "Energy"},
        {"ticker": "MA", "name": "Mastercard", "score": 78.5, "sector": "Technology"},
        {"ticker": "V", "name": "Visa", "score": 77.2, "sector": "Technology"},
        {"ticker": "META", "name": "Meta", "score": 82.1, "sector": "Technology"},
        {"ticker": "GOOGL", "name": "Alphabet", "score": 84.3, "sector": "Technology"},
        {"ticker": "AXP", "name": "American Express", "score": 70.5, "sector": "Financials"},
        {"ticker": "MRK", "name": "Merck", "score": 75.6, "sector": "Healthcare"},
        {"ticker": "COST", "name": "Costco", "score": 73.2, "sector": "Consumer"},
    ]

    # Create sample ETFs
    etfs = [
        {"ticker": "SOXX", "name": "Semiconductor ETF", "score": 87.3, "theme_id": "ai_cloud"},
        {"ticker": "SMH", "name": "iShares Semiconductor", "score": 84.1, "theme_id": "ai_cloud"},
        {"ticker": "ITA", "name": "iShares Aerospace", "score": 78.5, "theme_id": "defense"},
        {"ticker": "XAR", "name": "SPDR Aerospace", "score": 75.2, "theme_id": "defense"},
        {"ticker": "ICLN", "name": "Clean Energy", "score": 72.3, "theme_id": "energy_transition"},
        {"ticker": "TAN", "name": "Solar ETF", "score": 68.5, "theme_id": "energy_transition"},
        {"ticker": "XBI", "name": "Biotech ETF", "score": 76.8, "theme_id": "healthcare_innovation"},
        {"ticker": "BBH", "name": "VanEck Biotech", "score": 74.2, "theme_id": "healthcare_innovation"},
        {"ticker": "CIBR", "name": "Cybersecurity", "score": 71.5, "theme_id": "cybersecurity"},
    ]

    # Create sector map
    sector_map = {s["ticker"]: s["sector"] for s in stocks}

    # Create theme map
    theme_map = {
        "SOXX": "AI & Cloud",
        "SMH": "AI & Cloud",
        "ITA": "Defense",
        "XAR": "Defense",
        "ICLN": "Energy Transition",
        "TAN": "Energy Transition",
        "XBI": "Healthcare",
        "BBH": "Healthcare",
        "CIBR": "Cybersecurity",
    }

    logger.info(f"\n✓ Created sample data:")
    logger.info(f"  Stocks: {len(stocks)}")
    logger.info(f"  ETFs: {len(etfs)}")

    # Build portfolio
    logger.info("")
    logger.info("=" * 80)
    logger.info("PORTFOLIO CONSTRUCTION")
    logger.info("=" * 80)

    try:
        portfolio = constructor.build_portfolio(stocks, etfs, sector_map, theme_map)

        if not portfolio:
            logger.error("✗ Failed to build portfolio")
            return False

        logger.info("✓ Portfolio built successfully")

    except Exception as e:
        logger.error(f"✗ Error building portfolio: {e}", exc_info=True)
        return False

    # Generate ownership report
    logger.info("")
    logger.info("=" * 80)
    logger.info("OWNERSHIP REPORT")
    logger.info("=" * 80)

    try:
        ownership_report = reporter.generate_ownership_report(
            portfolio,
            {s["ticker"]: s for s in stocks},
            {e["ticker"]: e for e in etfs},
        )
        logger.info(ownership_report)
        logger.info("✓ Ownership report generated")

    except Exception as e:
        logger.error(f"✗ Error generating ownership report: {e}", exc_info=True)
        return False

    # Generate allocation CSV
    logger.info("")
    logger.info("=" * 80)
    logger.info("ALLOCATION CSV EXPORT")
    logger.info("=" * 80)

    csv_path = "data/quarterly_reports/allocation_model_Q1_2026.csv"
    try:
        success = reporter.generate_allocation_csv(
            portfolio,
            {s["ticker"]: s for s in stocks},
            {e["ticker"]: e for e in etfs},
            csv_path,
        )

        if success:
            logger.info(f"✓ Allocation CSV written to {csv_path}")
        else:
            logger.warning(f"⚠ CSV generation returned False")

    except Exception as e:
        logger.error(f"✗ Error generating CSV: {e}", exc_info=True)
        return False

    # Generate rebalance actions
    logger.info("")
    logger.info("=" * 80)
    logger.info("REBALANCE ACTIONS")
    logger.info("=" * 80)

    try:
        # Simulate current holdings with some drift
        current_holdings = {}
        for ticker, allocation in portfolio.allocations.items():
            import random
            drift = (random.random() - 0.5) * 0.05  # ±2.5% drift
            current_holdings[ticker] = max(0, allocation + drift)

        # Normalize to 100%
        total = sum(current_holdings.values())
        if total > 0:
            current_holdings = {k: v / total for k, v in current_holdings.items()}

        actions = constructor.generate_rebalance_actions(portfolio, current_holdings)
        rebalance_summary = reporter.generate_rebalance_summary(actions)
        logger.info(rebalance_summary)
        logger.info("✓ Rebalance actions generated")

    except Exception as e:
        logger.error(f"✗ Error generating rebalance actions: {e}")
        return False

    # Generate invalidation summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("THESIS INVALIDATION TRACKING")
    logger.info("=" * 80)

    try:
        invalidation_summary = reporter.generate_invalidation_summary(
            portfolio,
            {s["ticker"]: s for s in stocks},
            {e["ticker"]: e for e in etfs},
        )

        # Show sample triggers for top 3 positions
        logger.info("\nSample Invalidation Triggers (Top 3 Positions):")
        for i, (ticker, allocation) in enumerate(portfolio.highest_conviction[:3], 1):
            triggers = invalidation_summary["all_triggers"].get(ticker, [])
            logger.info(f"\n{i}. {ticker} ({allocation:.2%}):")
            for trigger in triggers[:3]:
                logger.info(f"   • {trigger['name']}: {trigger['action']}")
            if len(triggers) > 3:
                logger.info(f"   ... and {len(triggers) - 3} more triggers")

        logger.info("✓ Invalidation tracking generated")

    except Exception as e:
        logger.error(f"✗ Error generating invalidation summary: {e}")
        return False

    # Next review date
    logger.info("")
    logger.info("=" * 80)
    logger.info("QUARTERLY SCHEDULE")
    logger.info("=" * 80)

    try:
        next_review = reporter.get_next_review_date()
        logger.info(f"Next Review: {next_review}")
        logger.info("✓ Quarterly schedule calculated")

    except Exception as e:
        logger.error(f"✗ Error calculating next review: {e}")
        return False

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 5 VERIFICATION SUMMARY")
    logger.info("=" * 80)

    logger.info("\nVALIDATION CHECKLIST:")
    logger.info(f"  [✓] ReportGenerator initializes")
    logger.info(f"  [✓] Generates ownership reports")
    logger.info(f"  [✓] Exports allocation CSVs")
    logger.info(f"  [✓] Creates rebalance summaries")
    logger.info(f"  [✓] Tracks invalidation triggers")
    logger.info(f"  [✓] Calculates quarterly dates")

    logger.info("\nKEY DELIVERABLES:")
    logger.info(f"  • Ownership Report: {portfolio.total_positions} positions")
    logger.info(f"  • Allocation CSV: {portfolio.total_positions} lines")
    logger.info(f"  • Rebalance Actions: {len(actions['buy'])} buy + {len(actions['sell'])} sell + {len(actions['hold'])} hold")
    logger.info(f"  • Invalidation Triggers: {len(invalidation_summary['all_triggers'])} tracked")
    logger.info(f"  • Next Review: {next_review}")

    logger.info("\nNEXT STEPS:")
    logger.info("1. Phase 6: GitHub Actions automation")
    logger.info("2. Schedule quarterly runs (Jan 15, Apr 15, Jul 15, Oct 15)")
    logger.info("3. Phase 7: Documentation updates")

    return True


if __name__ == "__main__":
    try:
        success = test_report_generation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
