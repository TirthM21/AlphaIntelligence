"""
Portfolio Constructor.

Combines scored stocks and ETFs into a cohesive long-term allocation
with concentration rules, sector balancing, and conviction-based sizing.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Container for final portfolio allocation."""

    total_score: float
    stock_count: int
    etf_count: int
    total_positions: int

    allocations: Dict[str, float]  # Ticker → allocation % (0-1)
    core_allocations: Dict[str, float]  # Top 50%, 60% of portfolio
    satellite_allocations: Dict[str, float]  # Bottom 50%, 40% of portfolio

    sector_breakdown: Dict[str, float]  # Sector → allocation %
    theme_breakdown: Dict[str, float]  # Theme → allocation %

    # Metadata
    highest_conviction: List[Tuple[str, float]]  # Top 5 by allocation
    sector_concentration: float  # Herfindahl index
    rebalance_cadence: str  # "Annual" or "Quarterly"


class PortfolioConstructor:
    """Construct long-term allocation from scored stocks and ETFs."""

    def __init__(self):
        """Initialize portfolio constructor."""
        from .concentration_rules import ConcentrationRules, AllocationOptimizer
        self.rules = ConcentrationRules()
        self.optimizer = AllocationOptimizer(self.rules)

    def build_portfolio(
        self,
        stocks: List[Dict],
        etfs: List[Dict],
        sector_map: Dict[str, str],
        theme_map: Dict[str, str]
    ) -> Optional[PortfolioAllocation]:
        """
        Build optimal portfolio from scored assets.

        Args:
            stocks: List of stocks with 'ticker', 'score', 'name'
            etfs: List of ETFs with 'ticker', 'score', 'theme_id'
            sector_map: Dict mapping stock ticker to sector
            theme_map: Dict mapping ETF ticker to theme name

        Returns:
            PortfolioAllocation object, or None if invalid
        """
        try:
            logger.info(
                f"Building portfolio: {len(stocks)} stocks, {len(etfs)} ETFs"
            )

            # Validate input counts
            if len(stocks) < self.rules.min_stock_count:
                logger.warning(
                    f"Stock count {len(stocks)} < minimum {self.rules.min_stock_count}"
                )

            if len(etfs) < self.rules.min_etf_count:
                logger.warning(
                    f"ETF count {len(etfs)} < minimum {self.rules.min_etf_count}"
                )

            # Sort by score (descending) and take top candidates
            stocks_ranked = sorted(
                stocks,
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:self.rules.max_stock_count]

            etfs_ranked = sorted(
                etfs,
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:self.rules.max_etf_count]

            # Optimize allocations
            allocations = self.optimizer.optimize_allocations(
                stocks_ranked, etfs_ranked, sector_map
            )

            # Tier into core/satellite
            tiers = self.optimizer.tier_allocations(
                allocations, stocks_ranked, etfs_ranked
            )

            # Calculate sector breakdown
            sector_breakdown = self._calculate_sector_breakdown(
                stocks_ranked, allocations, sector_map
            )

            # Calculate theme breakdown
            theme_breakdown = self._calculate_theme_breakdown(
                etfs_ranked, allocations, theme_map
            )

            # Calculate highest conviction positions
            highest_conviction = sorted(
                allocations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # Calculate Herfindahl index (concentration metric)
            herfindahl = sum(a ** 2 for a in allocations.values())

            # Total portfolio score (weighted average)
            all_assets = stocks_ranked + etfs_ranked
            total_score = sum(
                asset.get("score", 0) * allocations.get(asset.get("ticker", ""), 0)
                for asset in all_assets
            )

            portfolio = PortfolioAllocation(
                total_score=total_score,
                stock_count=len(stocks_ranked),
                etf_count=len(etfs_ranked),
                total_positions=len(stocks_ranked) + len(etfs_ranked),
                allocations=allocations,
                core_allocations=tiers["core"],
                satellite_allocations=tiers["satellite"],
                sector_breakdown=sector_breakdown,
                theme_breakdown=theme_breakdown,
                highest_conviction=highest_conviction,
                sector_concentration=herfindahl,
                rebalance_cadence="Annual"
            )

            logger.info(
                f"✓ Portfolio built: {portfolio.total_positions} positions, "
                f"score {portfolio.total_score:.1f}, "
                f"concentration {portfolio.sector_concentration:.3f}"
            )

            return portfolio

        except Exception as e:
            logger.error(f"Error building portfolio: {e}")
            return None

    def _calculate_sector_breakdown(
        self,
        stocks: List[Dict],
        allocations: Dict[str, float],
        sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate sector allocation breakdown."""
        sector_breakdown = {}

        for stock in stocks:
            ticker = stock.get("ticker", "")
            sector = sector_map.get(ticker, "Unknown")
            allocation = allocations.get(ticker, 0)

            if sector not in sector_breakdown:
                sector_breakdown[sector] = 0
            sector_breakdown[sector] += allocation

        # Sort by allocation
        return dict(sorted(
            sector_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def _calculate_theme_breakdown(
        self,
        etfs: List[Dict],
        allocations: Dict[str, float],
        theme_map: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate theme allocation breakdown."""
        theme_breakdown = {}

        for etf in etfs:
            ticker = etf.get("ticker", "")
            theme = theme_map.get(ticker, "Other")
            allocation = allocations.get(ticker, 0)

            if theme not in theme_breakdown:
                theme_breakdown[theme] = 0
            theme_breakdown[theme] += allocation

        # Sort by allocation
        return dict(sorted(
            theme_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def generate_rebalance_actions(
        self,
        portfolio: PortfolioAllocation,
        current_holdings: Dict[str, float],
        threshold: float = 0.02
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate rebalance actions (buy/sell/hold).

        Args:
            portfolio: Current PortfolioAllocation
            current_holdings: Dict mapping ticker to current allocation
            threshold: Rebalance if drift > 2% (default)

        Returns:
            Dict with 'buy', 'sell', 'hold' actions
        """
        actions = {
            "buy": {},
            "sell": {},
            "hold": {}
        }

        for ticker, target_allocation in portfolio.allocations.items():
            current_allocation = current_holdings.get(ticker, 0)
            drift = abs(target_allocation - current_allocation)

            if drift > threshold:
                if target_allocation > current_allocation:
                    actions["buy"][ticker] = {
                        "current": current_allocation,
                        "target": target_allocation,
                        "action_size": target_allocation - current_allocation
                    }
                else:
                    actions["sell"][ticker] = {
                        "current": current_allocation,
                        "target": target_allocation,
                        "action_size": current_allocation - target_allocation
                    }
            else:
                actions["hold"][ticker] = {
                    "current": current_allocation,
                    "target": target_allocation,
                    "drift": drift
                }

        return actions

    def get_portfolio_summary(
        self,
        portfolio: PortfolioAllocation,
        stocks: Dict[str, Dict],
        etfs: Dict[str, Dict]
    ) -> str:
        """Generate human-readable portfolio summary."""
        lines = [
            "=" * 80,
            "LONG-TERM PORTFOLIO CONSTRUCTION",
            "=" * 80,
            "",
            f"Total Score:                 {portfolio.total_score:.1f}/100",
            f"Total Positions:             {portfolio.total_positions} "
            f"({portfolio.stock_count} stocks + {portfolio.etf_count} ETFs)",
            f"Sector Concentration (H):    {portfolio.sector_concentration:.3f} "
            f"(0-1 scale, lower = more diversified)",
            f"Rebalance Cadence:           {portfolio.rebalance_cadence}",
            "",
            "TOP 5 CONVICTION POSITIONS:",
            "-" * 80,
        ]

        for rank, (ticker, allocation) in enumerate(portfolio.highest_conviction, 1):
            lines.append(f"{rank}. {ticker:6} → {allocation:6.2%}")

        lines.extend([
            "",
            "SECTOR ALLOCATION:",
            "-" * 80,
        ])

        for sector, allocation in portfolio.sector_breakdown.items():
            lines.append(f"  {sector:20} → {allocation:6.2%}")

        lines.extend([
            "",
            "THEME ALLOCATION (ETFs):",
            "-" * 80,
        ])

        for theme, allocation in portfolio.theme_breakdown.items():
            lines.append(f"  {theme:30} → {allocation:6.2%}")

        lines.extend([
            "",
            "CORE vs SATELLITE:",
            "-" * 80,
            f"  Core (60%):              {len(portfolio.core_allocations)} positions",
            f"  Satellite (40%):         {len(portfolio.satellite_allocations)} positions",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)
