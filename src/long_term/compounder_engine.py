"""
Long-term compounder identification engine.

Scores stocks on:
- 60% Fundamental Dominance (growth, capital efficiency, reinvestment, balance sheet)
- 25% Long-Horizon RS Persistence (multi-year outperformance vs SPY)
- 15% Structural Trend Durability (40-week MA trends, smooth equity curves)

Returns deterministic, continuous scores 0-110+ with detailed breakdowns.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompounderScore:
    """Container for compounder scoring results."""

    ticker: str
    total_score: float  # 0-110+
    regime: str  # STRUCTURAL_GROWTH, MATURE_HOLD, CAPITAL_DESTRUCTION

    # Component scores (out of max)
    fundamental_score: float  # 0-60
    rs_persistence_score: float  # 0-25
    trend_durability_score: float  # 0-15
    moat_bonus: float  # 0-10

    # Detailed breakdown
    growth_quality_score: float  # 0-20
    capital_efficiency_score: float  # 0-20
    reinvestment_quality_score: float  # 0-10
    balance_sheet_score: float  # 0-10

    rs_1yr_score: float  # 0-8
    rs_3yr_score: float  # 0-10
    rs_5yr_score: float  # 0-7
    volatility_adjustment: float  # -5 to 0

    trend_strength_score: float  # 0-10
    trend_consistency_score: float  # 0-5

    # Key metrics (for reporting)
    revenue_cagr_3yr: Optional[float] = None
    revenue_cagr_5yr: Optional[float] = None
    eps_cagr_3yr: Optional[float] = None
    roic: Optional[float] = None
    roic_wacc_spread: Optional[float] = None
    fcf_margin: Optional[float] = None
    debt_to_ebitda: Optional[float] = None

    # Thesis drivers (for reporting)
    thesis_drivers: list = None  # List of key reasons for the score


class CompounderEngine:
    """Score stocks for long-term compounding potential."""

    def __init__(self):
        """Initialize scoring engine."""
        self.min_price = 5.0  # Exclude penny stocks
        self.max_price = 10000.0  # Exclude extreme prices

    def score_stock(
        self,
        ticker: str,
        fundamentals: Optional[Dict[str, Any]] = None,
        price_data: Optional[Dict[str, Any]] = None
    ) -> Optional[CompounderScore]:
        """
        Score a stock for long-term compounding potential.

        Args:
            ticker: Stock ticker
            fundamentals: LongTermFundamentals object or dict with metrics
            price_data: Dict with price/RS data {
                'current_price': float,
                'returns_1yr': float (decimal),
                'returns_3yr': float (annualized),
                'returns_5yr': float (annualized),
                'max_drawdown_3yr': float,
                'price_40w_ma': float,
                'price_200d_ma': float,
                'volatility': float,
                ...
            }

        Returns:
            CompounderScore object, or None if insufficient data
        """
        if not fundamentals or not price_data:
            logger.warning(f"Insufficient data for {ticker}")
            return None

        try:
            score = CompounderScore(
                ticker=ticker,
                total_score=0.0,
                regime="UNKNOWN",
                fundamental_score=0.0,
                rs_persistence_score=0.0,
                trend_durability_score=0.0,
                moat_bonus=0.0,
                growth_quality_score=0.0,
                capital_efficiency_score=0.0,
                reinvestment_quality_score=0.0,
                balance_sheet_score=0.0,
                rs_1yr_score=0.0,
                rs_3yr_score=0.0,
                rs_5yr_score=0.0,
                volatility_adjustment=0.0,
                trend_strength_score=0.0,
                trend_consistency_score=0.0,
                thesis_drivers=[]
            )

            # Price validation
            current_price = price_data.get("current_price", 0)
            if current_price < self.min_price or current_price > self.max_price:
                logger.debug(f"Price {current_price} outside range for {ticker}")
                return None

            # Calculate fundamental dominance (60 points)
            self._score_fundamentals(score, fundamentals)

            # Calculate RS persistence (25 points)
            self._score_rs_persistence(score, price_data)

            # Calculate trend durability (15 points)
            self._score_trend_durability(score, price_data)

            # Calculate moat bonus (0-10 points)
            self._score_moat_bonus(score, fundamentals)

            # Total score
            score.total_score = (
                score.fundamental_score +
                score.rs_persistence_score +
                score.trend_durability_score +
                score.moat_bonus
            )

            # Classify regime
            self._classify_regime(score, fundamentals, price_data)

            # Generate thesis drivers
            self._generate_thesis(score)

            return score

        except Exception as e:
            logger.error(f"Error scoring {ticker}: {e}")
            return None

    def _score_fundamentals(
        self,
        score: CompounderScore,
        fundamentals: Dict[str, Any]
    ) -> None:
        """Score fundamental quality (60 points max)."""
        from .metrics import MetricsCalculator

        # Growth Quality (20 points)
        rev_cagr_3yr = fundamentals.get("revenue_cagr_3yr", 0)
        rev_cagr_5yr = fundamentals.get("revenue_cagr_5yr", 0)
        eps_cagr_3yr = fundamentals.get("eps_cagr_3yr", 0)

        score.revenue_cagr_3yr = rev_cagr_3yr
        score.revenue_cagr_5yr = rev_cagr_5yr
        score.eps_cagr_3yr = eps_cagr_3yr

        # 3-year revenue CAGR: 0-8 points
        score_3yr_rev = MetricsCalculator.scale_linear(
            rev_cagr_3yr, 0.0, 0.15, 0.0, 8.0
        )

        # 5-year revenue CAGR: 0-7 points
        score_5yr_rev = MetricsCalculator.scale_linear(
            rev_cagr_5yr, 0.0, 0.15, 0.0, 7.0
        )

        # 3-year EPS CAGR: 0-5 points
        score_eps = MetricsCalculator.scale_linear(
            eps_cagr_3yr, 0.0, 0.20, 0.0, 5.0
        )

        score.growth_quality_score = score_3yr_rev + score_5yr_rev + score_eps

        # Capital Efficiency (20 points)
        roic = fundamentals.get("roic", 0)
        roic_wacc_spread = fundamentals.get("roic_wacc_spread", 0)
        fcf_margin = fundamentals.get("fcf_margin", 0)

        score.roic = roic
        score.roic_wacc_spread = roic_wacc_spread
        score.fcf_margin = fcf_margin

        # ROIC: 0-10 points (10% = 0, 25%+ = 10)
        score_roic = MetricsCalculator.scale_linear(
            roic, 0.10, 0.25, 0.0, 10.0
        )

        # ROIC-WACC spread: 0-5 points (0% = 0, 15%+ = 5)
        score_spread = MetricsCalculator.scale_linear(
            roic_wacc_spread, 0.0, 0.15, 0.0, 5.0
        )

        # FCF margin: 0-5 points (0% = 0, 20%+ = 5)
        score_fcf = MetricsCalculator.scale_linear(
            fcf_margin, 0.0, 0.20, 0.0, 5.0
        )

        score.capital_efficiency_score = score_roic + score_spread + score_fcf

        # Reinvestment Quality (10 points) - Placeholder
        # TODO: Calculate from R&D/capex metrics
        score.reinvestment_quality_score = 5.0

        # Balance Sheet Strength (10 points)
        debt_to_ebitda = fundamentals.get("debt_to_ebitda", 5.0)
        interest_coverage = fundamentals.get("interest_coverage", 3.0)

        score.debt_to_ebitda = debt_to_ebitda

        # Debt/EBITDA: 0-5 points (3.0 = 0, <1.0 = 5)
        score_debt = MetricsCalculator.scale_linear(
            debt_to_ebitda, 3.0, 1.0, 0.0, 5.0, invert=False
        )
        score_debt = max(0, score_debt)  # Can't be negative

        # Interest coverage: 0-3 points
        score_interest = MetricsCalculator.scale_linear(
            interest_coverage, 3.0, 10.0, 0.0, 3.0
        )

        # Cash proxy: 0-2 points (placeholder)
        score_cash = 1.0

        score.balance_sheet_score = score_debt + score_interest + score_cash

        # Total fundamental score
        score.fundamental_score = (
            score.growth_quality_score +
            score.capital_efficiency_score +
            score.reinvestment_quality_score +
            score.balance_sheet_score
        )

        # Cap at 60 points
        score.fundamental_score = min(60.0, score.fundamental_score)

    def _score_rs_persistence(
        self,
        score: CompounderScore,
        price_data: Dict[str, Any]
    ) -> None:
        """Score multi-year relative strength persistence (25 points max)."""
        from .metrics import MetricsCalculator

        # 1-year total return vs SPY: 0-8 points (-10% = 0, +20% = 8)
        returns_1yr = price_data.get("returns_1yr", 0.0)
        score.rs_1yr_score = MetricsCalculator.scale_linear(
            returns_1yr, -0.10, 0.20, 0.0, 8.0
        )

        # 3-year annualized return vs SPY: 0-10 points (-5% = 0, +15% = 10)
        returns_3yr = price_data.get("returns_3yr", 0.0)
        score.rs_3yr_score = MetricsCalculator.scale_linear(
            returns_3yr, -0.05, 0.15, 0.0, 10.0
        )

        # 5-year annualized return vs SPY: 0-7 points (-3% = 0, +12% = 7)
        returns_5yr = price_data.get("returns_5yr", 0.0)
        score.rs_5yr_score = MetricsCalculator.scale_linear(
            returns_5yr, -0.03, 0.12, 0.0, 7.0
        )

        # Drawdown-adjusted RS penalty: -5 to 0 points
        max_dd_3yr = price_data.get("max_drawdown_3yr", 0.0)
        spy_max_dd_3yr = price_data.get("spy_max_drawdown_3yr", -0.15)

        dd_relative = max_dd_3yr - spy_max_dd_3yr  # More negative = worse
        score.volatility_adjustment = MetricsCalculator.scale_linear(
            dd_relative, -0.20, 0.0, -5.0, 0.0
        )

        score.rs_persistence_score = (
            score.rs_1yr_score +
            score.rs_3yr_score +
            score.rs_5yr_score +
            score.volatility_adjustment
        )

        # Cap at 25 points
        score.rs_persistence_score = max(0, min(25.0, score.rs_persistence_score))

    def _score_trend_durability(
        self,
        score: CompounderScore,
        price_data: Dict[str, Any]
    ) -> None:
        """Score structural trend durability (15 points max)."""
        from .metrics import MetricsCalculator

        current_price = price_data.get("current_price", 0)
        price_40w_ma = price_data.get("price_40w_ma", 0)
        price_200d_ma = price_data.get("price_200d_ma", 0)

        # Price > 40-week MA: 0-5 points (0% = 0, 20%+ above = 5)
        if price_40w_ma > 0:
            distance_40w = (current_price - price_40w_ma) / price_40w_ma
            score.trend_strength_score = MetricsCalculator.scale_linear(
                distance_40w, 0.0, 0.20, 0.0, 5.0
            )
        else:
            score.trend_strength_score = 0.0

        # 40-week MA slope: 0-5 points
        ma_slope_40w = price_data.get("ma_slope_40w", 0.0)
        score_slope = MetricsCalculator.scale_linear(
            ma_slope_40w, 0.0, 0.15, 0.0, 5.0
        )

        # Months in uptrend: 0-5 points (12 months = 0, 36+ = 5)
        months_uptrend = price_data.get("months_uptrend", 12)
        score.trend_consistency_score = MetricsCalculator.scale_linear(
            months_uptrend, 12.0, 36.0, 0.0, 5.0
        )

        score.trend_durability_score = (
            score.trend_strength_score +
            score_slope +
            score.trend_consistency_score
        )

        # Cap at 15 points
        score.trend_durability_score = min(15.0, score.trend_durability_score)

    def _score_moat_bonus(
        self,
        score: CompounderScore,
        fundamentals: Dict[str, Any]
    ) -> None:
        """Score business moat proxies (0-10 bonus points)."""
        # Placeholder: Full moat scoring in Phase 3
        score.moat_bonus = 0.0

    def _classify_regime(
        self,
        score: CompounderScore,
        fundamentals: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> None:
        """Classify into one of 3 long-cycle regimes."""
        # Structural Growth regime conditions
        conditions_growth = [
            price_data.get("current_price", 0) > price_data.get("price_40w_ma", 0),
            score.rs_3yr_score > 5.0,  # Positive multi-year RS
            fundamentals.get("revenue_cagr_3yr", 0) > 0,
        ]

        if sum(conditions_growth) >= 2:
            score.regime = "STRUCTURAL_GROWTH"
        else:
            # Check for capital destruction
            conditions_destruction = [
                price_data.get("current_price", 0) < price_data.get("price_40w_ma", 0),
                score.rs_3yr_score < 0.0,
                fundamentals.get("debt_to_ebitda", 0) > 4.0,
            ]

            if sum(conditions_destruction) >= 2:
                score.regime = "CAPITAL_DESTRUCTION"
            else:
                score.regime = "MATURE_HOLD"

    def _generate_thesis(self, score: CompounderScore) -> None:
        """Generate key thesis drivers for the score."""
        score.thesis_drivers = []

        # Growth drivers
        if score.revenue_cagr_3yr and score.revenue_cagr_3yr > 0.10:
            score.thesis_drivers.append(
                f"Strong revenue growth ({score.revenue_cagr_3yr:.1%} CAGR, 3Y)"
            )

        # Capital efficiency drivers
        if score.roic and score.roic > 0.20:
            score.thesis_drivers.append(
                f"Capital efficiency ({score.roic:.1%} ROIC)"
            )

        # RS drivers
        if score.rs_3yr_score > 8.0:
            score.thesis_drivers.append(
                "Strong multi-year relative strength vs SPY"
            )

        # Trend drivers
        if score.trend_durability_score > 10.0:
            score.thesis_drivers.append(
                "Durable long-term trend structure"
            )

        # Balance sheet drivers
        if score.debt_to_ebitda and score.debt_to_ebitda < 1.5:
            score.thesis_drivers.append(
                f"Strong balance sheet ({score.debt_to_ebitda:.1f}x debt/EBITDA)"
            )

        # Regime drivers
        if score.regime == "STRUCTURAL_GROWTH":
            score.thesis_drivers.append("Regime: Structural Growth ✅")
        elif score.regime == "CAPITAL_DESTRUCTION":
            score.thesis_drivers.append("Regime: Capital Destruction ❌")
        else:
            score.thesis_drivers.append("Regime: Mature / Hold ⏸️")
