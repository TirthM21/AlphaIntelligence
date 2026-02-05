"""
ETF Scoring Engine for Thematic Identification.

Scores thematic ETFs on:
- 30% Theme Purity (concentration, holdings focus)
- 40% Long-Term RS (1Y, 3Y, 5Y returns vs SPY)
- 20% Efficiency (expense ratio, turnover)
- 10% Structural Tailwind (theme strength)

Returns 0-100 point scores with detailed breakdown.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ETFScore:
    """Container for ETF scoring results."""

    ticker: str
    name: str
    theme_id: str
    theme_name: str
    total_score: float  # 0-100

    # Component scores
    theme_purity_score: float  # 0-30
    rs_persistence_score: float  # 0-40
    efficiency_score: float  # 0-20
    tailwind_score: float  # 0-10

    # Detailed breakdown
    top_10_concentration_score: float  # 0-15
    sector_concentration_score: float  # 0-15
    rs_1yr_score: float  # 0-12
    rs_3yr_score: float  # 0-16
    rs_5yr_score: float  # 0-12
    expense_ratio_score: float  # 0-10
    turnover_score: float  # 0-10

    # Key metrics
    aum_millions: float
    expense_ratio: float
    turnover: float
    top_10_concentration: float
    return_1yr: Optional[float] = None
    return_3yr: Optional[float] = None
    return_5yr: Optional[float] = None

    # Thesis
    thesis_drivers: list = None


class ETFEngine:
    """Score thematic ETFs for long-term allocation."""

    def __init__(self, universe=None):
        """
        Initialize ETF engine.

        Args:
            universe: ETFUniverse instance (optional)
        """
        self.universe = universe

    def score_etf(
        self,
        etf_metadata: Dict[str, Any],
        price_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ETFScore]:
        """
        Score an ETF for long-term allocation.

        Args:
            etf_metadata: ETF metadata dict with theme, holdings, etc.
            price_data: Dict with returns {
                'return_1yr': float,
                'return_3yr': float,
                'return_5yr': float,
                'spy_return_1yr': float,
                'spy_return_3yr': float,
                'spy_return_5yr': float,
            }

        Returns:
            ETFScore object, or None if insufficient data
        """
        if not etf_metadata:
            return None

        try:
            score = ETFScore(
                ticker=etf_metadata.get("ticker", ""),
                name=etf_metadata.get("name", ""),
                theme_id=etf_metadata.get("theme_id", ""),
                theme_name=etf_metadata.get("theme_name", ""),
                total_score=0.0,
                theme_purity_score=0.0,
                rs_persistence_score=0.0,
                efficiency_score=0.0,
                tailwind_score=0.0,
                top_10_concentration_score=0.0,
                sector_concentration_score=0.0,
                rs_1yr_score=0.0,
                rs_3yr_score=0.0,
                rs_5yr_score=0.0,
                expense_ratio_score=0.0,
                turnover_score=0.0,
                aum_millions=etf_metadata.get("aum_millions", 0),
                expense_ratio=etf_metadata.get("expense_ratio", 0.5),
                turnover=etf_metadata.get("turnover", 50),
                top_10_concentration=etf_metadata.get("top_10_concentration", 0),
                thesis_drivers=[],
            )

            # Score theme purity (30 points)
            self._score_theme_purity(score, etf_metadata)

            # Score RS persistence (40 points)
            if price_data:
                self._score_rs_persistence(score, price_data)

            # Score efficiency (20 points)
            self._score_efficiency(score, etf_metadata)

            # Score structural tailwind (10 points)
            self._score_tailwind(score, etf_metadata)

            # Total score
            score.total_score = (
                score.theme_purity_score +
                score.rs_persistence_score +
                score.efficiency_score +
                score.tailwind_score
            )

            # Generate thesis
            self._generate_thesis(score)

            return score

        except Exception as e:
            logger.error(f"Error scoring {etf_metadata.get('ticker', 'unknown')}: {e}")
            return None

    def _score_theme_purity(
        self,
        score: ETFScore,
        etf_metadata: Dict[str, Any]
    ) -> None:
        """Score theme purity (30 points max)."""
        from .metrics import MetricsCalculator

        top_10_conc = etf_metadata.get("top_10_concentration", 0)
        sector_conc = etf_metadata.get("sector_concentration", 0)

        # Top 10 concentration: 0-15 points (30% = 0, 60%+ = 15)
        score.top_10_concentration_score = MetricsCalculator.scale_linear(
            top_10_conc, 0.30, 0.70, 0.0, 15.0
        )

        # Sector concentration: 0-15 points (70% = 0, 95%+ = 15)
        score.sector_concentration_score = MetricsCalculator.scale_linear(
            sector_conc, 0.70, 0.95, 0.0, 15.0
        )

        score.theme_purity_score = (
            score.top_10_concentration_score +
            score.sector_concentration_score
        )

        score.theme_purity_score = min(30.0, score.theme_purity_score)

    def _score_rs_persistence(
        self,
        score: ETFScore,
        price_data: Dict[str, Any]
    ) -> None:
        """Score multi-year RS persistence (40 points max)."""
        from .metrics import MetricsCalculator

        # Calculate returns vs SPY
        etf_return_1yr = price_data.get("return_1yr", 0.0)
        etf_return_3yr = price_data.get("return_3yr", 0.0)
        etf_return_5yr = price_data.get("return_5yr", 0.0)

        spy_return_1yr = price_data.get("spy_return_1yr", 0.0)
        spy_return_3yr = price_data.get("spy_return_3yr", 0.0)
        spy_return_5yr = price_data.get("spy_return_5yr", 0.0)

        # RS = ETF return - SPY return
        rs_1yr = etf_return_1yr - spy_return_1yr
        rs_3yr = etf_return_3yr - spy_return_3yr
        rs_5yr = etf_return_5yr - spy_return_5yr

        score.return_1yr = etf_return_1yr
        score.return_3yr = etf_return_3yr
        score.return_5yr = etf_return_5yr

        # 1-year RS: 0-12 points (-10% = 0, +20% = 12)
        score.rs_1yr_score = MetricsCalculator.scale_linear(
            rs_1yr, -0.10, 0.20, 0.0, 12.0
        )

        # 3-year RS: 0-16 points (-5% = 0, +15% = 16)
        score.rs_3yr_score = MetricsCalculator.scale_linear(
            rs_3yr, -0.05, 0.15, 0.0, 16.0
        )

        # 5-year RS: 0-12 points (-3% = 0, +12% = 12)
        score.rs_5yr_score = MetricsCalculator.scale_linear(
            rs_5yr, -0.03, 0.12, 0.0, 12.0
        )

        score.rs_persistence_score = (
            score.rs_1yr_score +
            score.rs_3yr_score +
            score.rs_5yr_score
        )

        score.rs_persistence_score = max(0, min(40.0, score.rs_persistence_score))

    def _score_efficiency(
        self,
        score: ETFScore,
        etf_metadata: Dict[str, Any]
    ) -> None:
        """Score cost and turnover efficiency (20 points max)."""
        from .metrics import MetricsCalculator

        expense_ratio = etf_metadata.get("expense_ratio", 0.5)
        turnover = etf_metadata.get("turnover", 50)

        # Expense ratio: 0-10 points (0.75% = 0, 0.05% = 10)
        score.expense_ratio_score = MetricsCalculator.scale_linear(
            expense_ratio, 0.0075, 0.0005, 0.0, 10.0, invert=False
        )
        score.expense_ratio_score = max(0, score.expense_ratio_score)

        # Turnover: 0-10 points (200% = 0, <20% = 10)
        score.turnover_score = MetricsCalculator.scale_linear(
            turnover, 2.0, 0.20, 0.0, 10.0, invert=False
        )
        score.turnover_score = max(0, min(10.0, score.turnover_score))

        score.efficiency_score = (
            score.expense_ratio_score +
            score.turnover_score
        )

        score.efficiency_score = min(20.0, score.efficiency_score)

    def _score_tailwind(
        self,
        score: ETFScore,
        etf_metadata: Dict[str, Any]
    ) -> None:
        """Score structural tailwind (10 points max)."""
        if self.universe:
            theme_id = etf_metadata.get("theme_id", "")
            tailwind = self.universe.get_tailwind_score(theme_id)
            score.tailwind_score = min(10.0, tailwind)
        else:
            # Default tailwinds if no universe provided
            theme_id = etf_metadata.get("theme_id", "")
            tailwind_map = {
                "ai_cloud": 10.0,
                "defense": 7.0,
                "energy_transition": 6.0,
                "healthcare_innovation": 6.0,
                "cybersecurity": 7.0,
            }
            score.tailwind_score = tailwind_map.get(theme_id, 5.0)

    def _generate_thesis(self, score: ETFScore) -> None:
        """Generate key thesis drivers for the score."""
        score.thesis_drivers = []

        # Theme purity drivers
        if score.top_10_concentration > 50:
            score.thesis_drivers.append(
                f"Concentrated top-10 holdings ({score.top_10_concentration:.0f}%)"
            )

        if score.sector_concentration_score > 10:
            score.thesis_drivers.append(
                f"Pure-play theme exposure (sector concentration score: {score.sector_concentration_score:.1f})"
            )

        # RS drivers
        if score.rs_3yr_score > 10:
            score.thesis_drivers.append(
                "Strong 3-year outperformance vs SPY"
            )

        if score.rs_5yr_score > 8:
            score.thesis_drivers.append(
                "Sustained 5-year relative strength"
            )

        # Cost drivers
        if score.expense_ratio < 0.30:
            score.thesis_drivers.append(
                f"Low-cost structure ({score.expense_ratio:.2%} ER)"
            )

        if score.turnover < 30:
            score.thesis_drivers.append(
                f"Tax-efficient ({score.turnover:.0f}% turnover)"
            )

        # Size drivers
        if score.aum_millions > 5000:
            score.thesis_drivers.append(
                f"Large AUM (${score.aum_millions:,.0f}M)"
            )

        # Tailwind driver
        if score.tailwind_score >= 8:
            score.thesis_drivers.append(
                "Strong structural tailwind"
            )

    def rank_etfs(self, etf_scores: list) -> list:
        """
        Rank ETFs by score.

        Args:
            etf_scores: List of ETFScore objects

        Returns:
            Sorted list (highest score first)
        """
        return sorted(etf_scores, key=lambda x: x.total_score, reverse=True)

    def split_by_bucket(
        self,
        etf_scores: list,
        core_count: int = 5
    ) -> tuple:
        """
        Split ETFs into Core and Satellite buckets.

        Args:
            etf_scores: List of ranked ETFScore objects
            core_count: Number of core ETFs (default 5)

        Returns:
            Tuple of (core_etfs, satellite_etfs)
        """
        ranked = self.rank_etfs(etf_scores)
        core = ranked[:core_count]
        satellite = ranked[core_count:core_count + 5]

        return core, satellite
