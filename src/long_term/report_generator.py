"""
Report Generator for Quarterly Ownership Reports.

Converts PortfolioAllocation objects into human-readable quarterly reports,
CSV allocations, and thesis invalidation tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import csv
import json

logger = logging.getLogger(__name__)


@dataclass
class InvalidationTrigger:
    """Container for a thesis invalidation trigger."""

    trigger_name: str
    category: str  # "Fundamental" | "Technical" | "Time-based" | "Portfolio"
    description: str
    action_threshold: str  # "Critical Exit" | "Reduce 50%" | "Monitor"


@dataclass
class AssetThesis:
    """Investment thesis for a single asset."""

    ticker: str
    asset_type: str  # "Stock" | "ETF"
    score: float
    allocation_pct: float
    sector_or_theme: str
    regime_or_bucket: str
    thesis_summary: str
    key_metrics: Dict[str, float]
    invalidation_triggers: List[InvalidationTrigger]


class InvalidationTracker:
    """Track thesis invalidation triggers for portfolio assets."""

    def __init__(self):
        """Initialize invalidation tracker with default triggers."""
        self.stock_triggers = {
            "roic_wacc_inversion": InvalidationTrigger(
                trigger_name="ROIC < WACC",
                category="Fundamental",
                description="Return on invested capital falls below cost of capital",
                action_threshold="Critical Exit",
            ),
            "revenue_decline": InvalidationTrigger(
                trigger_name="Revenue Decline 2+ Quarters",
                category="Fundamental",
                description="Top-line growth turns negative for consecutive quarters",
                action_threshold="Critical Exit",
            ),
            "margin_compression": InvalidationTrigger(
                trigger_name="Gross Margin Compression >200 bps",
                category="Fundamental",
                description="Gross margin declines by more than 200 basis points",
                action_threshold="Reduce 50%",
            ),
            "debt_explosion": InvalidationTrigger(
                trigger_name="Debt/EBITDA > 4.0",
                category="Fundamental",
                description="Balance sheet deterioration, debt burden unsustainable",
                action_threshold="Critical Exit",
            ),
            "price_below_40w_ma": InvalidationTrigger(
                trigger_name="Price < 40-Week MA for 3+ Months",
                category="Technical",
                description="Long-term trend broken, downside momentum",
                action_threshold="Critical Exit",
            ),
            "negative_3y_rs": InvalidationTrigger(
                trigger_name="3-Year RS vs SPY Turns Negative",
                category="Technical",
                description="Multi-year relative strength deterioration",
                action_threshold="Reduce 50%",
            ),
            "negative_5y_rs": InvalidationTrigger(
                trigger_name="5-Year RS vs SPY Turns Negative",
                category="Technical",
                description="Multi-year underperformance persists",
                action_threshold="Critical Exit",
            ),
            "score_drop_20": InvalidationTrigger(
                trigger_name="Quality Score Drops >20 Points",
                category="Fundamental",
                description="Deterioration in business quality metrics",
                action_threshold="Monitor",
            ),
            "holding_patience": InvalidationTrigger(
                trigger_name="Minimum 12-Month Holding Period",
                category="Time-based",
                description="Do not exit within 12 months unless critical violations",
                action_threshold="Monitor",
            ),
        }

        self.etf_triggers = {
            "negative_3y_rs": InvalidationTrigger(
                trigger_name="3-Year RS vs SPY Turns Negative",
                category="Technical",
                description="ETF underperformance persists",
                action_threshold="Reduce 50%",
            ),
            "theme_divergence": InvalidationTrigger(
                trigger_name="Theme Holdings Diverge (Sector <80%)",
                category="Fundamental",
                description="ETF strays from thematic focus",
                action_threshold="Monitor",
            ),
            "price_below_40w_ma": InvalidationTrigger(
                trigger_name="Price < 40-Week MA for 6+ Months",
                category="Technical",
                description="Long-term ETF downtrend",
                action_threshold="Reduce 50%",
            ),
            "expense_ratio_increase": InvalidationTrigger(
                trigger_name="Expense Ratio Increases >0.10%",
                category="Fundamental",
                description="Cost structure deterioration",
                action_threshold="Monitor",
            ),
            "score_drop_15": InvalidationTrigger(
                trigger_name="Quality Score Drops >15 Points",
                category="Fundamental",
                description="Deterioration in ETF quality",
                action_threshold="Monitor",
            ),
        }

    def get_triggers_for_stock(self, ticker: str) -> List[InvalidationTrigger]:
        """Get invalidation triggers for a stock."""
        return list(self.stock_triggers.values())

    def get_triggers_for_etf(self, ticker: str) -> List[InvalidationTrigger]:
        """Get invalidation triggers for an ETF."""
        return list(self.etf_triggers.values())


class ReportGenerator:
    """Generate quarterly ownership reports and allocations."""

    def __init__(self):
        """Initialize report generator."""
        self.invalidation_tracker = InvalidationTracker()

    def generate_ownership_report(
        self,
        portfolio,
        stocks: Dict,
        etfs: Dict,
        quarter_date: Optional[datetime] = None,
    ) -> str:
        """
        Generate human-readable quarterly ownership report.

        Args:
            portfolio: PortfolioAllocation object
            stocks: Dict of {ticker: stock_data}
            etfs: Dict of {ticker: etf_data}
            quarter_date: Report date (defaults to today)

        Returns:
            Formatted ownership report string
        """
        if quarter_date is None:
            quarter_date = datetime.now()

        # Calculate quarter designation
        q = (quarter_date.month - 1) // 3 + 1
        year = quarter_date.year

        lines = [
            "=" * 80,
            f"LONG-TERM COMPOUNDER REPORT - Q{q} {year}",
            f"Generated: {quarter_date.strftime('%Y-%m-%d')}",
            "Investment Horizon: 5-10 Years",
            "=" * 80,
            "",
            "PORTFOLIO SUMMARY",
            "-" * 80,
        ]

        # Portfolio stats
        lines.extend([
            f"Total Stocks: {portfolio.stock_count}",
            f"Total ETFs: {portfolio.etf_count}",
            f"Total Positions: {portfolio.total_positions}",
            f"Portfolio Score: {portfolio.total_score:.1f}/100",
            f"Concentration (Herfindahl): {portfolio.sector_concentration:.3f}",
            f"Rebalance Cadence: {portfolio.rebalance_cadence}",
            "",
            "TOP 5 CONVICTION POSITIONS",
            "-" * 80,
        ])

        for rank, (ticker, allocation) in enumerate(portfolio.highest_conviction, 1):
            asset_type = "ETF" if ticker in etfs else "Stock"
            lines.append(
                f"{rank}. {ticker:6} → {allocation:6.2%} ({asset_type})"
            )

        # Sector allocation
        lines.extend([
            "",
            "SECTOR ALLOCATION (Stocks)",
            "-" * 80,
        ])

        for sector, allocation in portfolio.sector_breakdown.items():
            lines.append(f"  {sector:20} → {allocation:6.2%}")

        # Theme allocation
        lines.extend([
            "",
            "THEME ALLOCATION (ETFs)",
            "-" * 80,
        ])

        for theme, allocation in portfolio.theme_breakdown.items():
            lines.append(f"  {theme:30} → {allocation:6.2%}")

        # Core vs satellite
        lines.extend([
            "",
            "CORE vs SATELLITE",
            "-" * 80,
            f"Core (60%):              {len(portfolio.core_allocations)} positions",
            f"Satellite (40%):         {len(portfolio.satellite_allocations)} positions",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    def generate_allocation_csv(
        self,
        portfolio,
        stocks: Dict,
        etfs: Dict,
        filepath: str,
    ) -> bool:
        """
        Generate allocation CSV for portfolio managers.

        Args:
            portfolio: PortfolioAllocation object
            stocks: Dict of {ticker: stock_data}
            etfs: Dict of {ticker: etf_data}
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "Rank",
                        "Ticker",
                        "Type",
                        "Score",
                        "Allocation (%)",
                        "Sector/Theme",
                        "Regime/Bucket",
                        "Position Size ($1M portfolio)",
                    ],
                )
                writer.writeheader()

                sorted_allocations = sorted(
                    portfolio.allocations.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for rank, (ticker, allocation) in enumerate(sorted_allocations, 1):
                    if ticker in etfs:
                        asset_type = "ETF"
                        sector_theme = (
                            portfolio.theme_breakdown.get(ticker, "Other")
                            if hasattr(portfolio, "theme_breakdown")
                            else "Thematic"
                        )
                        regime_bucket = "Core" if ticker in portfolio.core_allocations else "Satellite"
                        score = etfs[ticker].get("score", 0)
                    else:
                        asset_type = "Stock"
                        sector_theme = stocks[ticker].get("sector", "Unknown")
                        regime_bucket = "Core" if ticker in portfolio.core_allocations else "Satellite"
                        score = stocks[ticker].get("score", 0)

                    writer.writerow({
                        "Rank": rank,
                        "Ticker": ticker,
                        "Type": asset_type,
                        "Score": f"{score:.1f}",
                        "Allocation (%)": f"{allocation * 100:.2f}",
                        "Sector/Theme": sector_theme,
                        "Regime/Bucket": regime_bucket,
                        "Position Size ($1M portfolio)": f"${allocation * 1_000_000:,.0f}",
                    })

            logger.info(f"✓ Allocation CSV written to {filepath}")
            return True

        except Exception as e:
            logger.error(f"✗ Error writing allocation CSV: {e}")
            return False

    def generate_invalidation_summary(
        self,
        portfolio,
        stocks: Dict,
        etfs: Dict,
    ) -> Dict[str, List[Dict]]:
        """
        Generate thesis invalidation summary for tracking.

        Args:
            portfolio: PortfolioAllocation object
            stocks: Dict of {ticker: stock_data}
            etfs: Dict of {ticker: etf_data}

        Returns:
            Dict with invalidation tracking data
        """
        invalidation_summary = {
            "critical_monitors": [],
            "reduce_positions": [],
            "all_triggers": {},
        }

        # Add stock invalidation triggers
        for ticker in stocks.keys():
            if ticker in portfolio.allocations:
                triggers = self.invalidation_tracker.get_triggers_for_stock(ticker)
                invalidation_summary["all_triggers"][ticker] = [
                    {
                        "name": t.trigger_name,
                        "category": t.category,
                        "description": t.description,
                        "action": t.action_threshold,
                    }
                    for t in triggers
                ]

        # Add ETF invalidation triggers
        for ticker in etfs.keys():
            if ticker in portfolio.allocations:
                triggers = self.invalidation_tracker.get_triggers_for_etf(ticker)
                invalidation_summary["all_triggers"][ticker] = [
                    {
                        "name": t.trigger_name,
                        "category": t.category,
                        "description": t.description,
                        "action": t.action_threshold,
                    }
                    for t in triggers
                ]

        return invalidation_summary

    def generate_rebalance_summary(
        self,
        rebalance_actions: Dict[str, Dict],
    ) -> str:
        """
        Generate human-readable rebalance action summary.

        Args:
            rebalance_actions: Dict with buy/sell/hold actions

        Returns:
            Formatted rebalance summary
        """
        lines = [
            "=" * 80,
            "REBALANCE ACTIONS (2% Threshold)",
            "=" * 80,
            "",
        ]

        # Buy actions
        buy_actions = rebalance_actions.get("buy", {})
        lines.append(f"BUY RECOMMENDATIONS ({len(buy_actions)} positions):")
        lines.append("-" * 80)
        if buy_actions:
            for ticker, action in list(buy_actions.items())[:10]:
                lines.append(
                    f"  {ticker:6} → {action['action_size']:6.2%} "
                    f"(current: {action['current']:.2%}, target: {action['target']:.2%})"
                )
            if len(buy_actions) > 10:
                lines.append(f"  ... and {len(buy_actions) - 10} more")
        else:
            lines.append("  None - portfolio at target allocations")

        # Sell actions
        sell_actions = rebalance_actions.get("sell", {})
        lines.append("")
        lines.append(f"SELL RECOMMENDATIONS ({len(sell_actions)} positions):")
        lines.append("-" * 80)
        if sell_actions:
            for ticker, action in list(sell_actions.items())[:10]:
                lines.append(
                    f"  {ticker:6} → {action['action_size']:6.2%} "
                    f"(current: {action['current']:.2%}, target: {action['target']:.2%})"
                )
            if len(sell_actions) > 10:
                lines.append(f"  ... and {len(sell_actions) - 10} more")
        else:
            lines.append("  None - no positions exceed sell threshold")

        # Hold actions
        hold_actions = rebalance_actions.get("hold", {})
        lines.append("")
        lines.append(f"HOLD RECOMMENDATIONS ({len(hold_actions)} positions):")
        lines.append("-" * 80)
        if hold_actions:
            sorted_holds = sorted(
                hold_actions.items(),
                key=lambda x: x[1]["drift"],
                reverse=True,
            )
            for ticker, action in sorted_holds[:10]:
                lines.append(
                    f"  {ticker:6} → drift: {action['drift']:6.2%} "
                    f"(current: {action['current']:.2%}, target: {action['target']:.2%})"
                )
            if len(hold_actions) > 10:
                lines.append(f"  ... and {len(hold_actions) - 10} more")
        else:
            lines.append("  None")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def get_next_review_date(self, quarter_date: Optional[datetime] = None) -> str:
        """
        Calculate next quarterly review date.

        Args:
            quarter_date: Current report date (defaults to today)

        Returns:
            Formatted next review date (YYYY-MM-DD)
        """
        if quarter_date is None:
            quarter_date = datetime.now()

        # Determine next quarter
        current_q = (quarter_date.month - 1) // 3 + 1
        if current_q == 4:
            next_date = datetime(quarter_date.year + 1, 1, 15)
        else:
            next_month = (current_q + 1) * 3 - 2  # First month of next quarter
            next_date = datetime(quarter_date.year, next_month + 2, 15)

        return next_date.strftime("%Y-%m-%d")
