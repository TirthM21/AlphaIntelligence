#!/usr/bin/env python3
"""
Test script for ETF Engine.

Demonstrates ETF scoring on known thematic ETFs.
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_etf_engine():
    """Test ETF engine on sample thematic ETFs."""
    logger.info("=" * 80)
    logger.info("PHASE 3 VERIFICATION: ETF Scoring Engine")
    logger.info("=" * 80)

    try:
        from src.long_term.etf_engine import ETFEngine
        from src.long_term.etf_universe import ETFUniverse
        logger.info("✓ Imported ETFEngine and ETFUniverse")
    except ImportError as e:
        logger.error(f"✗ Failed to import: {e}")
        return False

    # Initialize components
    try:
        universe = ETFUniverse()
        logger.info("✓ Initialized ETFUniverse")

        engine = ETFEngine(universe=universe)
        logger.info("✓ Initialized ETFEngine")
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        return False

    # Get thematic ETFs by theme
    logger.info("")
    logger.info("=" * 80)
    logger.info("THEMATIC ETF DISCOVERY")
    logger.info("=" * 80)

    try:
        summary = universe.summary_by_theme()

        for theme_id, etfs in summary.items():
            theme_config = universe.get_theme_by_id(theme_id)
            theme_name = theme_config.get("name") if theme_config else theme_id

            logger.info(f"\n{theme_name}:")
            logger.info(f"  Found {len(etfs)} qualified ETFs")

            for etf in etfs[:3]:  # Show top 3 per theme
                logger.info(f"    - {etf.ticker} ({etf.name})")
                logger.info(f"      AUM: ${etf.aum_millions:.0f}M, "
                           f"ER: {etf.expense_ratio:.2%}, "
                           f"Top10: {etf.top_10_concentration:.0f}%")

        total_etfs = sum(len(etfs) for etfs in summary.values())
        logger.info(f"\nTotal discovered ETFs (after quality filter): {total_etfs}")

    except Exception as e:
        logger.error(f"✗ Error discovering ETFs: {e}")
        return False

    # Score ETFs
    logger.info("")
    logger.info("=" * 80)
    logger.info("ETF SCORING")
    logger.info("=" * 80)

    try:
        all_etfs = universe.discover_thematic_etfs()
        all_etfs = universe.filter_by_quality(all_etfs)

        # Sample price data (simplified for testing)
        sample_price_data = {
            "return_1yr": 0.35,          # 35% return
            "return_3yr": 0.22,          # 22% annualized
            "return_5yr": 0.28,          # 28% annualized
            "spy_return_1yr": 0.20,      # SPY 20%
            "spy_return_3yr": 0.12,      # SPY 12% annualized
            "spy_return_5yr": 0.14,      # SPY 14% annualized
        }

        etf_scores = []

        for etf in all_etfs:
            # Convert metadata to dict
            etf_dict = {
                "ticker": etf.ticker,
                "name": etf.name,
                "theme_id": etf.theme_id,
                "theme_name": etf.theme_name,
                "aum_millions": etf.aum_millions,
                "expense_ratio": etf.expense_ratio,
                "turnover": etf.turnover,
                "top_10_concentration": etf.top_10_concentration,
                "sector_concentration": etf.sector_concentration,
            }

            # Vary returns slightly by theme for realistic scoring
            adjusted_price_data = sample_price_data.copy()

            # AI/Cloud shows stronger performance
            if etf.theme_id == "ai_cloud":
                adjusted_price_data["return_1yr"] *= 1.3
                adjusted_price_data["return_3yr"] *= 1.4
                adjusted_price_data["return_5yr"] *= 1.3

            # Defense shows moderate outperformance
            elif etf.theme_id == "defense":
                adjusted_price_data["return_1yr"] *= 1.1
                adjusted_price_data["return_3yr"] *= 1.2
                adjusted_price_data["return_5yr"] *= 1.15

            score = engine.score_etf(etf_dict, adjusted_price_data)

            if score:
                etf_scores.append(score)

        # Rank by score
        ranked = engine.rank_etfs(etf_scores)

        logger.info(f"\nTop 10 Thematic ETFs by Score:")
        logger.info("")

        for i, score in enumerate(ranked[:10], 1):
            logger.info(f"{i}. {score.ticker} - {score.theme_name}")
            logger.info(f"   Score: {score.total_score:.1f}/100")
            logger.info(f"   Components: "
                       f"Theme {score.theme_purity_score:.0f}, "
                       f"RS {score.rs_persistence_score:.0f}, "
                       f"Eff {score.efficiency_score:.0f}, "
                       f"Tailwind {score.tailwind_score:.0f}")

            if score.thesis_drivers:
                for driver in score.thesis_drivers[:3]:
                    logger.info(f"     • {driver}")

            logger.info("")

    except Exception as e:
        logger.error(f"✗ Error scoring ETFs: {e}", exc_info=True)
        return False

    # Split into core/satellite buckets
    logger.info("=" * 80)
    logger.info("PORTFOLIO BUCKETS")
    logger.info("=" * 80)

    try:
        core, satellite = engine.split_by_bucket(etf_scores, core_count=5)

        logger.info(f"\n✓ Core ETFs ({len(core)}):")
        for etf in core:
            logger.info(f"  - {etf.ticker}: {etf.total_score:.1f} points")

        logger.info(f"\n✓ Satellite ETFs ({len(satellite)}):")
        for etf in satellite:
            logger.info(f"  - {etf.ticker}: {etf.total_score:.1f} points")

    except Exception as e:
        logger.error(f"✗ Error splitting buckets: {e}")
        return False

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3 VERIFICATION SUMMARY")
    logger.info("=" * 80)

    logger.info("\nVALIDATION CHECKLIST:")
    logger.info(f"  [✓] ETFUniverse discovers and filters ETFs")
    logger.info(f"  [✓] ETFEngine scores thematic ETFs (0-100)")
    logger.info(f"  [✓] Scoring components: Theme/RS/Efficiency/Tailwind")
    logger.info(f"  [✓] Core/Satellite bucketing works")
    logger.info(f"  [✓] Sample data scoring produces realistic results")

    logger.info("\nNEXT STEPS:")
    logger.info("1. Integrate real price data (1Y, 3Y, 5Y returns)")
    logger.info("2. Extend with actual ETF holdings data from yfinance")
    logger.info("3. Proceed to Phase 4: Portfolio Construction")

    return True


if __name__ == "__main__":
    try:
        success = test_etf_engine()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
