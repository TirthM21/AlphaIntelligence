#!/usr/bin/env python3
"""Automated position reporting - runs as part of GitHub Actions workflow.

This version:
- Fetches positions from data/positions.json
- Uses YFinance for live pricing
- Generates stop loss recommendations
- Saves report to file
- Does NOT require Robinhood credentials
"""

import sys
import logging
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.data.yfinance_positions import YFinancePositionFetcher
    from src.analysis.position_manager import PositionManager

    logger.info("Starting automated position analysis...")

    # Initialize fetcher (no credentials needed for local file)
    try:
        fetcher = YFinancePositionFetcher()
        fetcher.login()
        
        # Fetch positions
        logger.info("Fetching positions from data/positions.json...")
        positions = fetcher.fetch_positions()
        fetcher.logout()

        if not positions:
            logger.info("No open positions found")
            # Create a placeholder report saying so
            # or just exit
            sys.exit(0)

        logger.info(f"Found {len(positions)} positions")

        # Analyze using cached data (no extra API calls)
        logger.info("Analyzing positions using cached market data...")
        manager = PositionManager(use_cache=True)
        analysis = manager.analyze_portfolio(positions)

        # Generate report
        report = manager.format_portfolio_report(analysis)

        # Save to file
        output_dir = Path("./data/position_reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"position_management_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write(report)

        logger.info(f"✓ Report saved to {filename}")

        # Also print to stdout for GitHub Actions log
        print("\n" + report)

        logger.info("✓ Automated position analysis complete")

    except Exception as e:
        logger.error(f"Error during position analysis: {e}", exc_info=True)
        sys.exit(1)

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Install dependencies with: pip install yfinance pandas")
    sys.exit(1)
