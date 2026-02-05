#!/usr/bin/env python3
"""
Verify long-term module structure and imports.

Tests that all modules exist and can be imported without errors.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all long-term modules import correctly."""
    logger.info("=" * 80)
    logger.info("PHASE 1 VERIFICATION: Long-Term Module Structure")
    logger.info("=" * 80)

    modules_to_test = [
        ("src.long_term", "Long-term module"),
        ("src.long_term.metrics", "Metrics calculator"),
        ("src.long_term.data_fetcher", "Data fetcher"),
    ]

    results = {}

    for module_name, description in modules_to_test:
        try:
            logger.info(f"\nTesting {description}...")
            exec(f"import {module_name}")
            logger.info(f"✓ Successfully imported {module_name}")
            results[module_name] = True
        except ImportError as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")
            results[module_name] = False
        except Exception as e:
            logger.error(f"✗ Unexpected error importing {module_name}: {e}")
            results[module_name] = False

    # Test specific classes
    logger.info("")
    logger.info("Testing key classes...")

    try:
        from src.long_term.metrics import MetricsCalculator
        logger.info("✓ MetricsCalculator imported successfully")

        # Test a simple calculation
        cagr = MetricsCalculator.calculate_cagr(100, 200, 5)
        logger.info(f"  Sample CAGR calculation: 100→200 over 5 years = {cagr:.2%}")
        results["MetricsCalculator"] = True

    except Exception as e:
        logger.error(f"✗ Failed to test MetricsCalculator: {e}")
        results["MetricsCalculator"] = False

    try:
        from src.long_term.data_fetcher import LongTermFundamentalsFetcher
        logger.info("✓ LongTermFundamentalsFetcher imported successfully")

        fetcher = LongTermFundamentalsFetcher()
        logger.info(f"  Cache directory: {fetcher.cache_dir}")
        logger.info(f"  Cache expiry: {fetcher.cache_expiry_days} days")
        results["LongTermFundamentalsFetcher"] = True

    except Exception as e:
        logger.error(f"✗ Failed to test LongTermFundamentalsFetcher: {e}")
        results["LongTermFundamentalsFetcher"] = False

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1 VERIFICATION SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for item, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {item}")

    logger.info("")
    logger.info("VALIDATION CHECKLIST:")
    logger.info(f"  [{'✓' if results.get('MetricsCalculator') else '✗'}] MetricsCalculator works")
    logger.info(f"  [{'✓' if results.get('LongTermFundamentalsFetcher') else '✗'}] LongTermFundamentalsFetcher initializes")
    logger.info(f"  [{'?' if results.get('LongTermFundamentalsFetcher') else '✗'}] FMP API integration (requires FMP_API_KEY)")
    logger.info(f"  [{'✓' if passed == total else '✗'}] All modules import successfully")

    logger.info("")
    if passed == total:
        logger.info(f"Result: {passed}/{total} items passed - PHASE 1 FOUNDATIO COMPLETE")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Set FMP_API_KEY environment variable for full data fetching tests")
        logger.info("2. Proceed to Phase 2: Implement CompounderEngine scoring logic")
        return True
    else:
        logger.error(f"Result: {passed}/{total} items passed - PHASE 1 INCOMPLETE")
        return False


if __name__ == "__main__":
    try:
        success = test_imports()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
