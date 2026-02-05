"""Slack notification module for sending screening alerts.

Sends formatted messages to Slack channels via webhooks or Bot tokens.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SlackNotifier:
    """Send screening results to Slack channels.

    Supports both webhook URLs and Bot tokens for posting messages.

    Environment Variables:
        SLACK_WEBHOOK_URL: Webhook URL for posting (easier setup)
        SLACK_BOT_TOKEN: Bot token (more features but requires app setup)
        SLACK_CHANNEL: Channel to post to (only needed with bot token)

    Example:
        >>> notifier = SlackNotifier()
        >>> notifier.send_screening_results(results_df, top_n=5)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel: Optional[str] = None
    ) -> None:
        """Initialize the Slack notifier.

        Args:
            webhook_url: Slack webhook URL. Defaults to env SLACK_WEBHOOK_URL.
            bot_token: Slack bot token. Defaults to env SLACK_BOT_TOKEN.
            channel: Slack channel. Defaults to env SLACK_CHANNEL.
        """
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.bot_token = bot_token or os.getenv('SLACK_BOT_TOKEN')
        self.channel = channel or os.getenv('SLACK_CHANNEL', '#stock-alerts')

        # Import slack_sdk only if needed
        self.client = None
        if self.bot_token:
            try:
                from slack_sdk import WebClient
                self.client = WebClient(token=self.bot_token)
                logger.info("Slack bot client initialized")
            except ImportError:
                logger.warning("slack-sdk not installed. Install with: pip install slack-sdk")

        if not self.webhook_url and not self.bot_token:
            logger.warning("Slack not configured. Set SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN.")

        logger.info(f"SlackNotifier initialized (webhook: {bool(self.webhook_url)}, bot: {bool(self.bot_token)})")

    def _format_slack_blocks(self, results: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Format screening results as Slack blocks.

        Args:
            results: DataFrame with screening results.
            top_n: Number of top candidates to include.

        Returns:
            List of Slack block dictionaries.
        """
        today = datetime.now().strftime('%B %d, %Y')
        total_candidates = len(results)
        top_results = results.head(top_n)

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 Daily Stock Screening Results - {today}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{total_candidates}* stocks screened. Top *{len(top_results)}* candidates below:"
                }
            },
            {"type": "divider"}
        ]

        # Add each top candidate
        for idx, row in top_results.iterrows():
            # Buy signal emoji
            if row['buy_signal'] >= 80:
                signal_emoji = "🔥"
                signal_text = "STRONG BUY"
            elif row['buy_signal'] >= 65:
                signal_emoji = "✅"
                signal_text = "BUY"
            elif row['buy_signal'] >= 50:
                signal_emoji = "⚡"
                signal_text = "CONSIDER"
            else:
                signal_emoji = "⏸️"
                signal_text = "WATCH"

            # Format text
            text = f"*#{idx + 1}: {row['ticker']}* ({row.get('name', 'N/A')})\n"
            text += f"{signal_emoji} *{signal_text}* - Buy Signal: *{row['buy_signal']:.1f}*/100\n"
            text += f"• Value: {row['value_score']:.1f} | Support: {row['support_score']:.1f}\n"
            text += f"• Price: ${row['current_price']:.2f}"

            if row.get('rsi') is not None:
                rsi_status = "Oversold" if row['rsi'] < 30 else "Neutral" if row['rsi'] < 70 else "Overbought"
                text += f" | RSI: {row['rsi']:.1f} ({rsi_status})"

            if row.get('pe_ratio'):
                text += f"\n• P/E: {row['pe_ratio']:.2f}"
            if row.get('pb_ratio'):
                text += f" | P/B: {row['pb_ratio']:.2f}"

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            })

        # Add legend
        blocks.extend([
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "*Legend:* 🔥 Strong Buy (80+) | ✅ Buy (65-79) | ⚡ Consider (50-64) | ⏸️ Watch (<50)"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "⚠️ _Not financial advice. Always do your own research._"
                    }
                ]
            }
        ])

        return blocks

    def send_screening_results(
        self,
        results: pd.DataFrame,
        top_n: int = 5
    ) -> bool:
        """Send screening results to Slack.

        Args:
            results: DataFrame with screening results (from screen_candidates).
            top_n: Number of top candidates to include.

        Returns:
            True if message sent successfully, False otherwise.

        Example:
            >>> notifier = SlackNotifier()
            >>> results = screen_candidates(db, tickers)
            >>> notifier.send_screening_results(results, top_n=5)
        """
        if results.empty:
            logger.warning("No screening results to send")
            return False

        try:
            blocks = self._format_slack_blocks(results, top_n)

            # Try webhook first (simpler)
            if self.webhook_url:
                return self._send_via_webhook(blocks)
            elif self.client:
                return self._send_via_bot(blocks)
            else:
                logger.error("Slack not configured properly")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False

    def _send_via_webhook(self, blocks: List[Dict]) -> bool:
        """Send message via webhook URL.

        Args:
            blocks: Slack block elements.

        Returns:
            True if successful, False otherwise.
        """
        try:
            import requests

            payload = {"blocks": blocks}
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("✓ Slack message sent successfully via webhook")
                return True
            else:
                logger.error(f"Slack webhook failed: {response.status_code} - {response.text}")
                return False

        except ImportError:
            logger.error("requests library not installed. Install with: pip install requests")
            return False
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False

    def _send_via_bot(self, blocks: List[Dict]) -> bool:
        """Send message via bot token.

        Args:
            blocks: Slack block elements.

        Returns:
            True if successful, False otherwise.
        """
        try:
            response = self.client.chat_postMessage(
                channel=self.channel,
                blocks=blocks,
                text="Daily Stock Screening Results"  # Fallback text
            )

            if response['ok']:
                logger.info(f"✓ Slack message sent successfully to {self.channel}")
                return True
            else:
                logger.error(f"Slack bot send failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Bot send failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Slack connection.

        Returns:
            True if connection successful, False otherwise.

        Example:
            >>> notifier = SlackNotifier()
            >>> if notifier.test_connection():
            ...     print("Slack configuration is valid!")
        """
        try:
            if self.webhook_url:
                import requests
                response = requests.post(
                    self.webhook_url,
                    json={"text": "Test message from Stock Screener"},
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info("✓ Slack webhook test successful")
                    return True
                else:
                    logger.error(f"✗ Webhook test failed: {response.status_code}")
                    return False

            elif self.client:
                response = self.client.auth_test()
                if response['ok']:
                    logger.info(f"✓ Slack bot authenticated as {response['user']}")
                    return True
                else:
                    logger.error("✗ Bot authentication failed")
                    return False

            else:
                logger.error("✗ Slack not configured")
                return False

        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            return False
