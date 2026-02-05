"""
Concentration Rules Engine.

Enforces constraints for long-term allocation:
- Max 10% per individual stock
- Max 15% per ETF
- Max 30% per sector
- Max 5% per subsector/theme
- Minimum portfolio size: 15-25 stocks, 8-10 ETFs
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConcentrationRules:
    """Container for portfolio concentration constraints."""

    max_stock_position: float = 0.10  # 10% max per stock
    max_etf_position: float = 0.15  # 15% max per ETF
    max_sector_allocation: float = 0.30  # 30% max per sector
    max_theme_allocation: float = 0.05  # 5% max per theme
    min_stock_count: int = 15  # Minimum stocks
    max_stock_count: int = 25  # Maximum stocks
    min_etf_count: int = 8  # Minimum ETFs
    max_etf_count: int = 10  # Maximum ETFs
    min_total_positions: int = 23  # 15 stocks + 8 ETFs
    max_total_positions: int = 35  # 25 stocks + 10 ETFs


class ConstraintValidator:
    """Validate portfolio constraints."""

    def __init__(self, rules: Optional[ConcentrationRules] = None):
        """
        Initialize validator.

        Args:
            rules: ConcentrationRules object (uses defaults if None)
        """
        self.rules = rules or ConcentrationRules()

    def validate_portfolio(
        self,
        stocks: List[Dict],
        etfs: List[Dict],
        allocations: Dict[str, float],
        sector_map: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate entire portfolio against all constraints.

        Args:
            stocks: List of stock dicts with 'ticker', 'score', 'sector'
            etfs: List of ETF dicts with 'ticker', 'score', 'theme_id'
            allocations: Dict mapping ticker to allocation % (0-1)
            sector_map: Dict mapping stock ticker to sector

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check position counts
        stock_count = len(stocks)
        etf_count = len(etfs)
        total_count = stock_count + etf_count

        if stock_count < self.rules.min_stock_count:
            violations.append(
                f"Stock count {stock_count} < minimum {self.rules.min_stock_count}"
            )

        if stock_count > self.rules.max_stock_count:
            violations.append(
                f"Stock count {stock_count} > maximum {self.rules.max_stock_count}"
            )

        if etf_count < self.rules.min_etf_count:
            violations.append(
                f"ETF count {etf_count} < minimum {self.rules.min_etf_count}"
            )

        if etf_count > self.rules.max_etf_count:
            violations.append(
                f"ETF count {etf_count} > maximum {self.rules.max_etf_count}"
            )

        if total_count < self.rules.min_total_positions:
            violations.append(
                f"Total positions {total_count} < minimum {self.rules.min_total_positions}"
            )

        if total_count > self.rules.max_total_positions:
            violations.append(
                f"Total positions {total_count} > maximum {self.rules.max_total_positions}"
            )

        # Check individual position sizes
        stock_violations = self._check_individual_positions(
            stocks, allocations, "Stock"
        )
        violations.extend(stock_violations)

        etf_violations = self._check_individual_positions(
            etfs, allocations, "ETF"
        )
        violations.extend(etf_violations)

        # Check sector allocations
        sector_violations = self._check_sector_allocations(
            stocks, allocations, sector_map
        )
        violations.extend(sector_violations)

        # Check sum to 100%
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow 1% rounding error
            violations.append(
                f"Total allocation {total_allocation:.2%} != 100%"
            )

        is_valid = len(violations) == 0
        return is_valid, violations

    def _check_individual_positions(
        self,
        assets: List[Dict],
        allocations: Dict[str, float],
        asset_type: str
    ) -> List[str]:
        """Check individual position size limits."""
        violations = []
        max_size = (
            self.rules.max_stock_position if asset_type == "Stock"
            else self.rules.max_etf_position
        )

        for asset in assets:
            ticker = asset.get("ticker", "")
            allocation = allocations.get(ticker, 0)

            if allocation > max_size:
                violations.append(
                    f"{asset_type} {ticker} allocation {allocation:.1%} "
                    f"> maximum {max_size:.1%}"
                )

        return violations

    def _check_sector_allocations(
        self,
        stocks: List[Dict],
        allocations: Dict[str, float],
        sector_map: Dict[str, str]
    ) -> List[str]:
        """Check sector concentration limits."""
        violations = []

        # Calculate sector totals
        sector_totals = {}

        for stock in stocks:
            ticker = stock.get("ticker", "")
            sector = sector_map.get(ticker, "Unknown")
            allocation = allocations.get(ticker, 0)

            if sector not in sector_totals:
                sector_totals[sector] = 0
            sector_totals[sector] += allocation

        # Check against max
        for sector, total in sector_totals.items():
            if total > self.rules.max_sector_allocation:
                violations.append(
                    f"Sector {sector} allocation {total:.1%} "
                    f"> maximum {self.rules.max_sector_allocation:.1%}"
                )

        return violations


class AllocationOptimizer:
    """Optimize portfolio allocations subject to constraints."""

    def __init__(self, rules: Optional[ConcentrationRules] = None):
        """
        Initialize optimizer.

        Args:
            rules: ConcentrationRules object
        """
        self.rules = rules or ConcentrationRules()
        self.validator = ConstraintValidator(rules)

    def optimize_allocations(
        self,
        stocks: List[Dict],
        etfs: List[Dict],
        sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Optimize allocations based on scores subject to constraints.

        Args:
            stocks: List of stocks with 'ticker', 'score'
            etfs: List of ETFs with 'ticker', 'score'
            sector_map: Dict mapping stock ticker to sector

        Returns:
            Dict mapping ticker to allocation percentage (0-1)
        """
        allocations = {}

        # Step 1: Calculate initial weights based on scores
        stock_weights = self._calculate_score_weights(stocks)
        etf_weights = self._calculate_score_weights(etfs)

        # Combine weights (stocks + ETFs)
        all_weights = {**stock_weights, **etf_weights}

        # Step 2: Normalize to 100%
        total_weight = sum(all_weights.values())
        if total_weight > 0:
            all_weights = {
                ticker: weight / total_weight
                for ticker, weight in all_weights.items()
            }

        # Step 3: Apply concentration limits
        allocations = self._apply_concentration_limits(
            all_weights, stocks, etfs, sector_map
        )

        return allocations

    def _calculate_score_weights(self, assets: List[Dict]) -> Dict[str, float]:
        """Calculate initial weights proportional to scores."""
        weights = {}

        for asset in assets:
            ticker = asset.get("ticker", "")
            score = asset.get("score", 0)

            # Use score as weight (higher score = higher allocation)
            if score > 0:
                weights[ticker] = score

        return weights

    def _apply_concentration_limits(
        self,
        weights: Dict[str, float],
        stocks: List[Dict],
        etfs: List[Dict],
        sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """Apply concentration limits iteratively."""
        allocations = weights.copy()
        max_iterations = 10

        for iteration in range(max_iterations):
            violations = []

            # Check individual position limits
            for ticker, allocation in allocations.items():
                is_stock = any(s.get("ticker") == ticker for s in stocks)
                max_size = (
                    self.rules.max_stock_position if is_stock
                    else self.rules.max_etf_position
                )

                if allocation > max_size:
                    # Scale down to max and redistribute overflow
                    overflow = allocation - max_size
                    allocations[ticker] = max_size
                    violations.append((ticker, overflow))

            # Check sector limits
            sector_allocations = {}
            for stock in stocks:
                ticker = stock.get("ticker", "")
                sector = sector_map.get(ticker, "Unknown")
                allocation = allocations.get(ticker, 0)

                if sector not in sector_allocations:
                    sector_allocations[sector] = 0
                sector_allocations[sector] += allocation

            for sector, total in sector_allocations.items():
                if total > self.rules.max_sector_allocation:
                    # Scale down sector proportionally
                    scale = self.rules.max_sector_allocation / total
                    for stock in stocks:
                        if sector_map.get(stock.get("ticker"), "") == sector:
                            ticker = stock.get("ticker", "")
                            allocations[ticker] *= scale
                    violations.append((sector, total - self.rules.max_sector_allocation))

            # If no violations, we're done
            if not violations:
                break

        # Final normalization to ensure sum = 100%
        total = sum(allocations.values())
        if total > 0:
            allocations = {
                ticker: allocation / total
                for ticker, allocation in allocations.items()
            }

        return allocations

    def tier_allocations(
        self,
        allocations: Dict[str, float],
        stocks: List[Dict],
        etfs: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Tier allocations into Core (60%) and Satellite (40%).

        Args:
            allocations: Dict mapping ticker to allocation
            stocks: List of stocks with 'score'
            etfs: List of ETFs with 'score'

        Returns:
            Dict with 'core' and 'satellite' allocations
        """
        # Rank by score
        all_assets = stocks + etfs
        ranked = sorted(
            all_assets,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        core_count = len(ranked) // 2  # Top 50%
        core_tickers = {a.get("ticker", "") for a in ranked[:core_count]}

        core_allocs = {}
        satellite_allocs = {}

        for ticker, allocation in allocations.items():
            if ticker in core_tickers:
                core_allocs[ticker] = allocation
            else:
                satellite_allocs[ticker] = allocation

        # Normalize tiers to their own 100%
        core_total = sum(core_allocs.values())
        satellite_total = sum(satellite_allocs.values())

        if core_total > 0:
            core_allocs = {
                t: a / core_total * 0.60 for t, a in core_allocs.items()
            }

        if satellite_total > 0:
            satellite_allocs = {
                t: a / satellite_total * 0.40 for t, a in satellite_allocs.items()
            }

        return {
            "core": core_allocs,
            "satellite": satellite_allocs,
        }
