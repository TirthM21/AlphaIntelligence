#!/usr/bin/env python3
"""Quick test of the Quant Analysis Engine with a small set of stocks."""

import logging

from src.screening.quant_engine import QuantAnalysisEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run a quick test with a few stocks."""
    # Test with a small set of well-known stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    logger.info("Starting quick test of Quant Analysis Engine")
    logger.info(f"Testing with: {test_tickers}")

    # Initialize engine
    engine = QuantAnalysisEngine(cache_dir='./data/cache')

    # Run screening
    try:
        report = engine.run(test_tickers)
        print(report)

        logger.info("\n✓ Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
