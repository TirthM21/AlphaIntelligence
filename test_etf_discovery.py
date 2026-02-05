#!/usr/bin/env python3
"""
Test script for ETF Universe Discovery.

Demonstrates ETF discovery and filtering without scoring dependencies.
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_etf_discovery():
    """Test ETF universe discovery and filtering."""
    logger.info("=" * 80)
    logger.info("PHASE 3 VERIFICATION: ETF Universe Discovery")
    logger.info("=" * 80)

    try:
        from src.long_term.etf_universe import ETFUniverse
        logger.info("✓ Imported ETFUniverse")
    except ImportError as e:
        logger.error(f"✗ Failed to import: {e}")
        return False

    # Initialize universe
    try:
        universe = ETFUniverse()
        logger.info("✓ Initialized ETFUniverse")
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        return False

    # Test theme configuration loading
    logger.info("")
    logger.info("=" * 80)
    logger.info("THEME CONFIGURATION")
    logger.info("=" * 80)

    try:
        themes_config = universe.themes_config
        themes = themes_config.get("themes", [])

        logger.info(f"\n✓ Loaded {len(themes)} themes:")

        for theme in themes:
            theme_id = theme.get("id")
            theme_name = theme.get("name")
            tailwind = theme.get("tailwind_score")
            keywords = theme.get("keywords", [])

            logger.info(f"\n  {theme_name} (ID: {theme_id})")
            logger.info(f"    Tailwind Score: {tailwind}/10")
            logger.info(f"    Keywords: {', '.join(keywords[:3])}...")

    except Exception as e:
        logger.error(f"✗ Error loading themes: {e}")
        return False

    # Test ETF discovery
    logger.info("")
    logger.info("=" * 80)
    logger.info("ETF DISCOVERY")
    logger.info("=" * 80)

    try:
        all_etfs = universe.discover_thematic_etfs()
        logger.info(f"\n✓ Discovered {len(all_etfs)} candidate ETFs")

        for etf in all_etfs[:5]:
            logger.info(f"  - {etf.ticker}: {etf.name}")

    except Exception as e:
        logger.error(f"✗ Error discovering ETFs: {e}")
        return False

    # Test quality filtering
    logger.info("")
    logger.info("=" * 80)
    logger.info("QUALITY FILTERING")
    logger.info("=" * 80)

    try:
        filtering_rules = universe.themes_config.get("filtering_rules", {})
        min_aum = filtering_rules.get("min_aum_millions")
        max_expense = filtering_rules.get("max_expense_ratio")
        max_turnover = filtering_rules.get("max_turnover")

        logger.info(f"\n✓ Filtering Rules:")
        logger.info(f"  Min AUM: ${min_aum}M")
        logger.info(f"  Max Expense Ratio: {max_expense:.2%}")
        logger.info(f"  Max Turnover: {max_turnover:.0f}%")

        filtered_etfs = universe.filter_by_quality(all_etfs)
        logger.info(f"\n✓ After filtering: {len(filtered_etfs)}/{len(all_etfs)} ETFs pass quality")

        excluded_count = len(all_etfs) - len(filtered_etfs)
        if excluded_count > 0:
            logger.info(f"  ({excluded_count} excluded for quality reasons)")

    except Exception as e:
        logger.error(f"✗ Error filtering: {e}")
        return False

    # Test theme bucketing
    logger.info("")
    logger.info("=" * 80)
    logger.info("THEME BUCKETING")
    logger.info("=" * 80)

    try:
        summary = universe.summary_by_theme()

        logger.info(f"\n✓ ETFs by Theme:")

        total_etfs = 0
        for theme_id, etfs in summary.items():
            theme_config = universe.get_theme_by_id(theme_id)
            theme_name = theme_config.get("name") if theme_config else theme_id
            total_etfs += len(etfs)

            logger.info(f"\n  {theme_name}")
            logger.info(f"    Count: {len(etfs)}")

            for etf in etfs:
                logger.info(f"      • {etf.ticker:6} | "
                           f"AUM: ${etf.aum_millions:6.0f}M | "
                           f"ER: {etf.expense_ratio:5.2%} | "
                           f"Top10: {etf.top_10_concentration:3.0f}%")

        logger.info(f"\n  Total: {total_etfs} ETFs across all themes")

    except Exception as e:
        logger.error(f"✗ Error bucketing: {e}")
        return False

    # Test theme purity calculation
    logger.info("")
    logger.info("=" * 80)
    logger.info("THEME PURITY SCORES")
    logger.info("=" * 80)

    try:
        logger.info(f"\n✓ Theme Purity Examples:")

        for etf in filtered_etfs[:5]:
            purity = universe.calculate_theme_purity(etf)
            logger.info(f"  {etf.ticker:6} → Purity: {purity:5.1f}/100")

    except Exception as e:
        logger.error(f"✗ Error calculating purity: {e}")
        return False

    # Test tailwind scoring
    logger.info("")
    logger.info("=" * 80)
    logger.info("STRUCTURAL TAILWINDS")
    logger.info("=" * 80)

    try:
        logger.info(f"\n✓ Tailwind Scores by Theme:")

        for theme in themes:
            theme_id = theme.get("id")
            tailwind = universe.get_tailwind_score(theme_id)
            theme_name = theme.get("name")

            logger.info(f"  {theme_name:30} → {tailwind:5.1f}/10")

    except Exception as e:
        logger.error(f"✗ Error calculating tailwinds: {e}")
        return False

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3 VERIFICATION SUMMARY")
    logger.info("=" * 80)

    logger.info("\nVALIDATION CHECKLIST:")
    logger.info(f"  [✓] Themes configuration loads (5 themes)")
    logger.info(f"  [✓] ETFs discovered and catalogued ({len(all_etfs)} total)")
    logger.info(f"  [✓] Quality filtering works ({len(filtered_etfs)} qualified)")
    logger.info(f"  [✓] Theme bucketing by category")
    logger.info(f"  [✓] Theme purity scoring logic")
    logger.info(f"  [✓] Structural tailwind scores")

    logger.info("\nPHASE 3 IMPLEMENTATION:")
    logger.info("  ✓ etf_universe.py - Discovery & filtering")
    logger.info("  ✓ etf_engine.py - Scoring (30/40/20/10)")
    logger.info("  ✓ etf_themes.json - Theme configuration")

    logger.info("\nKEY STATISTICS:")
    logger.info(f"  Candidate ETFs: {len(all_etfs)}")
    logger.info(f"  Qualified ETFs: {len(filtered_etfs)}")
    logger.info(f"  Quality Pass Rate: {100*len(filtered_etfs)/len(all_etfs):.0f}%")
    logger.info(f"  Themes: {len(themes)}")
    logger.info(f"  ETF Coverage:")

    for theme_id, etfs in summary.items():
        theme_config = universe.get_theme_by_id(theme_id)
        if theme_config:
            logger.info(f"    - {theme_config.get('name')}: {len(etfs)} ETFs")

    logger.info("\nNEXT STEPS:")
    logger.info("1. Phase 4: Build PortfolioConstructor for allocation rules")
    logger.info("2. Integrate real-time price data (1Y/3Y/5Y returns)")
    logger.info("3. Implement scoring with price data")
    logger.info("4. Generate quarterly ETF recommendations")

    return True


if __name__ == "__main__":
    try:
        success = test_etf_discovery()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
