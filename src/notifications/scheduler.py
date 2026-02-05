"""Scheduling module for automated stock screening and notifications.

Supports running screening on a schedule (daily, hourly, etc.) with
automatic notifications via email and/or Slack.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data import YahooFinanceFetcher, StockDatabase
from src.screening import screen_candidates
from .email_notifier import EmailNotifier
from .slack_notifier import SlackNotifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScreeningScheduler:
    """Automated stock screening with notifications.

    Runs screening on a schedule and sends results via email and/or Slack.

    Environment Variables:
        SCREENING_TICKERS: Comma-separated list of tickers to screen
        SCREENING_TOP_N: Number of top results to include in notifications (default: 10)
        SCREENING_MIN_SIGNAL: Minimum buy signal to include (default: 50)

    Example:
        >>> scheduler = ScreeningScheduler()
        >>> scheduler.run_screening()  # Run once
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        enable_email: bool = True,
        enable_slack: bool = True
    ) -> None:
        """Initialize the screening scheduler.

        Args:
            tickers: List of tickers to screen. Defaults to env SCREENING_TICKERS.
            enable_email: Whether to send email notifications.
            enable_slack: Whether to send Slack notifications.
        """
        # Load tickers
        if tickers:
            self.tickers = tickers
        else:
            tickers_env = os.getenv('SCREENING_TICKERS', '')
            if tickers_env:
                self.tickers = [t.strip() for t in tickers_env.split(',')]
            else:
                # Default ticker list
                self.tickers = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                    'JPM', 'BAC', 'WMT', 'JNJ', 'XOM'
                ]
                logger.warning(f"No tickers configured. Using defaults: {self.tickers}")

        self.top_n = int(os.getenv('SCREENING_TOP_N', '10'))
        self.min_signal = float(os.getenv('SCREENING_MIN_SIGNAL', '50'))

        # Initialize components
        self.fetcher = YahooFinanceFetcher()
        self.db = StockDatabase()

        # Initialize notifiers
        self.email_notifier = EmailNotifier() if enable_email else None
        self.slack_notifier = SlackNotifier() if enable_slack else None

        logger.info(f"ScreeningScheduler initialized with {len(self.tickers)} tickers")
        logger.info(f"Notifications: Email={enable_email}, Slack={enable_slack}")

    def fetch_data(self, update_cache: bool = True) -> bool:
        """Fetch latest data for all tickers.

        Args:
            update_cache: Whether to update cached data.

        Returns:
            True if data fetched successfully, False otherwise.
        """
        logger.info(f"Fetching data for {len(self.tickers)} tickers...")

        success_count = 0
        for i, ticker in enumerate(self.tickers, 1):
            try:
                logger.info(f"[{i}/{len(self.tickers)}] Fetching {ticker}...")

                # Fetch fundamentals
                fundamentals = self.fetcher.fetch_fundamentals(ticker)
                if fundamentals:
                    self.db.save_stock_fundamentals(ticker, fundamentals)
                    logger.debug(f"✓ {ticker} fundamentals saved")

                # Fetch price history
                prices = self.fetcher.fetch_price_history(ticker, period="1y")
                if not prices.empty:
                    self.db.save_price_history(ticker, prices)
                    logger.debug(f"✓ {ticker} price history saved")
                    success_count += 1

            except Exception as e:
                logger.error(f"✗ Failed to fetch {ticker}: {e}")
                continue

        logger.info(f"Data fetch complete: {success_count}/{len(self.tickers)} successful")
        return success_count > 0

    def run_screening(self) -> Optional[pd.DataFrame]:
        """Run screening on configured tickers.

        Returns:
            DataFrame with screening results, or None if failed.
        """
        logger.info("Starting screening analysis...")

        try:
            # Run screening
            results = screen_candidates(self.db, self.tickers)

            if results.empty:
                logger.warning("No screening results generated")
                return None

            # Filter by minimum signal
            results = results[results['buy_signal'] >= self.min_signal]

            if results.empty:
                logger.info(f"No candidates meet minimum signal threshold: {self.min_signal}")
                return None

            logger.info(f"Screening complete: {len(results)} candidates found")
            return results

        except Exception as e:
            logger.error(f"Screening failed: {e}")
            return None

    def send_notifications(self, results: pd.DataFrame) -> bool:
        """Send notifications with screening results.

        Args:
            results: DataFrame with screening results.

        Returns:
            True if at least one notification sent successfully.
        """
        if results.empty:
            logger.warning("No results to send")
            return False

        success = False

        # Send email
        if self.email_notifier:
            try:
                if self.email_notifier.send_screening_results(results, self.top_n):
                    logger.info("✓ Email notification sent")
                    success = True
                else:
                    logger.warning("✗ Email notification failed")
            except Exception as e:
                logger.error(f"Email notification error: {e}")

        # Send Slack
        if self.slack_notifier:
            try:
                # Use fewer results for Slack (more concise)
                slack_top_n = min(self.top_n, 5)
                if self.slack_notifier.send_screening_results(results, slack_top_n):
                    logger.info("✓ Slack notification sent")
                    success = True
                else:
                    logger.warning("✗ Slack notification failed")
            except Exception as e:
                logger.error(f"Slack notification error: {e}")

        return success

    def run_once(self, fetch_data: bool = True) -> bool:
        """Run screening once and send notifications.

        Args:
            fetch_data: Whether to fetch fresh data before screening.

        Returns:
            True if successful, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("AUTOMATED STOCK SCREENING - SINGLE RUN")
        logger.info("=" * 60)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Tickers: {len(self.tickers)}")
        logger.info(f"Top N: {self.top_n}")
        logger.info(f"Min Signal: {self.min_signal}")
        logger.info("=" * 60)

        try:
            # Fetch data if needed
            if fetch_data:
                if not self.fetch_data():
                    logger.error("Data fetch failed")
                    return False

            # Run screening
            results = self.run_screening()
            if results is None or results.empty:
                logger.warning("No screening results")
                return False

            # Send notifications
            if self.send_notifications(results):
                logger.info("✓ Screening and notifications complete")
                return True
            else:
                logger.warning("Screening complete but notifications failed")
                return False

        except Exception as e:
            logger.error(f"Screening run failed: {e}")
            return False

    def test_setup(self) -> bool:
        """Test configuration and connections.

        Returns:
            True if all tests pass, False otherwise.
        """
        logger.info("Testing setup...")

        all_passed = True

        # Test database
        try:
            all_tickers = self.db.get_all_tickers()
            logger.info(f"✓ Database connected ({len(all_tickers)} stocks)")
        except Exception as e:
            logger.error(f"✗ Database error: {e}")
            all_passed = False

        # Test email
        if self.email_notifier:
            if self.email_notifier.test_connection():
                logger.info("✓ Email configuration valid")
            else:
                logger.warning("✗ Email configuration invalid")
                all_passed = False

        # Test Slack
        if self.slack_notifier:
            if self.slack_notifier.test_connection():
                logger.info("✓ Slack configuration valid")
            else:
                logger.warning("✗ Slack configuration invalid")
                all_passed = False

        return all_passed


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='Stock Screening Scheduler')

    parser.add_argument(
        'command',
        choices=['run', 'test', 'fetch'],
        help='Command to execute'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers (overrides env var)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top results (default: 10)'
    )
    parser.add_argument(
        '--no-email',
        action='store_true',
        help='Disable email notifications'
    )
    parser.add_argument(
        '--no-slack',
        action='store_true',
        help='Disable Slack notifications'
    )

    args = parser.parse_args()

    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]

    # Initialize scheduler
    scheduler = ScreeningScheduler(
        tickers=tickers,
        enable_email=not args.no_email,
        enable_slack=not args.no_slack
    )

    if hasattr(scheduler, 'top_n'):
        scheduler.top_n = args.top_n

    # Execute command
    if args.command == 'test':
        success = scheduler.test_setup()
        sys.exit(0 if success else 1)

    elif args.command == 'fetch':
        success = scheduler.fetch_data()
        sys.exit(0 if success else 1)

    elif args.command == 'run':
        success = scheduler.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
