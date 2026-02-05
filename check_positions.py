#!/usr/bin/env python3
"""Quick script to check your current portfolio positions.

Usage:
    python check_positions.py

This script:
- ✓ Fetches your current stock positions from data/positions.json
- ✓ Fetches live prices using YFinance
- ✗ Does NOT connect to any brokerage
- ✗ Does NOT execute any trades

Requires:
    pip install yfinance
"""

import sys
import logging
from src.data.yfinance_positions import YFinancePositionFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*60)
    print("PORTFOLIO POSITION CHECKER (Local + YFinance)")
    print("="*60)
    print("\nThis will:")
    print("  ✓ Read positions from data/positions.json")
    print("  ✓ Fetch current prices from Yahoo Finance")
    print("\n" + "="*60 + "\n")

    # Initialize fetcher
    try:
        fetcher = YFinancePositionFetcher()
    except Exception as e:
        print(f"\nERROR: {e}\n")
        sys.exit(1)

    # Login (dummy)
    fetcher.login()

    try:
        # Fetch positions
        print("\nFetching positions...\n")
        positions = fetcher.fetch_positions()

        if not positions:
            print("="*60)
            print("No open positions found (or data/positions.json is empty)")
            print("="*60)
            return

        # Display formatted report
        print(fetcher.format_positions_report(positions))

        # Export option
        export = input("\nExport to file? (y/n): ").strip().lower()
        if export == 'y':
            from datetime import datetime
            filename = f"portfolio_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(filename, 'w') as f:
                f.write(fetcher.format_positions_report())

            print(f"\n✓ Exported to: {filename}")

    finally:
        fetcher.logout()
        print("\n✓ Done\n")


if __name__ == '__main__':
    main()
