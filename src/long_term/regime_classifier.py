"""
Long-cycle regime classifier.

Classifies stocks into one of 3 long-term investment regimes:
1. Structural Growth - Eligible for new capital
2. Mature / Hold - Hold existing positions, no new capital
3. Capital Destruction - Exit signal
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Long-cycle investment regime."""
    STRUCTURAL_GROWTH = "STRUCTURAL_GROWTH"      # ✅ Buy
    MATURE_HOLD = "MATURE_HOLD"                 # ⏸️ Hold
    CAPITAL_DESTRUCTION = "CAPITAL_DESTRUCTION"  # ❌ Sell


class RegimeClassifier:
    """Classify stocks into long-cycle investment regimes."""

    def classify(
        self,
        ticker: str,
        fundamentals: Optional[Dict[str, Any]],
        price_data: Optional[Dict[str, Any]],
        detailed: bool = False
    ) -> Regime:
        """
        Classify stock into regime.

        Args:
            ticker: Stock ticker
            fundamentals: Dict with growth, profitability, balance sheet metrics
            price_data: Dict with price, moving averages, RS metrics
            detailed: If True, return regime details

        Returns:
            Regime enum value
        """
        if not fundamentals or not price_data:
            logger.warning(f"Insufficient data for regime classification of {ticker}")
            return Regime.MATURE_HOLD

        try:
            return self._classify_regime(ticker, fundamentals, price_data, detailed)
        except Exception as e:
            logger.error(f"Error classifying {ticker}: {e}")
            return Regime.MATURE_HOLD

    def _classify_regime(
        self,
        ticker: str,
        fundamentals: Dict[str, Any],
        price_data: Dict[str, Any],
        detailed: bool = False
    ) -> Regime:
        """Perform regime classification."""

        # Structural Growth Regime Conditions
        structural_growth_conditions = self._check_structural_growth(
            fundamentals, price_data
        )

        if len(structural_growth_conditions) >= 3:  # Need 3+ conditions
            if detailed:
                logger.info(
                    f"{ticker} → STRUCTURAL_GROWTH "
                    f"({len(structural_growth_conditions)}/5 conditions met)"
                )
            return Regime.STRUCTURAL_GROWTH

        # Capital Destruction Regime Conditions
        capital_destruction_conditions = self._check_capital_destruction(
            fundamentals, price_data
        )

        if len(capital_destruction_conditions) >= 2:  # Need 2+ conditions
            if detailed:
                logger.info(
                    f"{ticker} → CAPITAL_DESTRUCTION "
                    f"({len(capital_destruction_conditions)}/5 conditions met)"
                )
            return Regime.CAPITAL_DESTRUCTION

        # Default to Mature/Hold
        if detailed:
            logger.info(f"{ticker} → MATURE_HOLD (insufficient growth or destruction)")

        return Regime.MATURE_HOLD

    def _check_structural_growth(
        self,
        fundamentals: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> list:
        """
        Check Structural Growth regime conditions.

        Conditions:
        1. Price > 40-week MA AND > 200-day MA
        2. 40-week MA slope > 0 for ≥12 consecutive months
        3. Revenue + EPS CAGR (3-year) both > 0%
        4. RS vs SPY positive over 1Y, 3Y, and 5Y windows
        5. ROIC > WACC

        Returns:
            List of met conditions
        """
        met_conditions = []

        # Condition 1: Price > moving averages
        current_price = price_data.get("current_price", 0)
        price_40w_ma = price_data.get("price_40w_ma", 0)
        price_200d_ma = price_data.get("price_200d_ma", 0)

        if (current_price > price_40w_ma > 0 and
            current_price > price_200d_ma > 0):
            met_conditions.append("price > 40W MA and 200D MA")

        # Condition 2: 40-week MA slope positive for 12+ months
        ma_slope_40w = price_data.get("ma_slope_40w", 0.0)
        months_uptrend = price_data.get("months_uptrend", 0)

        if ma_slope_40w > 0 and months_uptrend >= 12:
            met_conditions.append("40W MA slope positive for 12+ months")

        # Condition 3: Revenue and EPS CAGR both positive
        rev_cagr_3yr = fundamentals.get("revenue_cagr_3yr", 0)
        eps_cagr_3yr = fundamentals.get("eps_cagr_3yr", 0)

        if rev_cagr_3yr > 0 and eps_cagr_3yr > 0:
            met_conditions.append("positive revenue and EPS CAGR")

        # Condition 4: Positive multi-year RS
        returns_1yr = price_data.get("returns_1yr", 0)
        returns_3yr = price_data.get("returns_3yr", 0)
        returns_5yr = price_data.get("returns_5yr", 0)

        rs_positive = sum([
            returns_1yr > 0,
            returns_3yr > 0,
            returns_5yr > 0
        ])

        if rs_positive >= 2:  # At least 2 of 3 windows positive
            met_conditions.append("positive RS over multiple timeframes")

        # Condition 5: ROIC > WACC
        roic = fundamentals.get("roic", 0)
        wacc = fundamentals.get("wacc", 0.08)

        if roic > wacc:
            met_conditions.append("ROIC > WACC (value creation)")

        return met_conditions

    def _check_capital_destruction(
        self,
        fundamentals: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> list:
        """
        Check Capital Destruction regime conditions.

        Conditions:
        1. Price < 40-week MA for 3+ consecutive months
        2. Multi-year RS deterioration (3Y and 5Y RS negative)
        3. Margin compression (gross margin declining 3 consecutive quarters)
        4. Balance sheet decay (debt/EBITDA > 4.0)
        5. Revenue decline (2+ consecutive quarters)

        Returns:
            List of met conditions
        """
        met_conditions = []

        # Condition 1: Price < 40-week MA
        current_price = price_data.get("current_price", 0)
        price_40w_ma = price_data.get("price_40w_ma", 0)
        months_below_40w = price_data.get("months_below_40w", 0)

        if price_40w_ma > 0 and current_price < price_40w_ma and months_below_40w >= 3:
            met_conditions.append("price < 40W MA for 3+ months")

        # Condition 2: Multi-year RS deterioration
        returns_3yr = price_data.get("returns_3yr", 0)
        returns_5yr = price_data.get("returns_5yr", 0)

        if returns_3yr < 0 and returns_5yr < 0:
            met_conditions.append("negative 3Y and 5Y RS")

        # Condition 3: Margin compression
        gross_margin_trend = fundamentals.get("gross_margin_trend", 0)
        consecutive_margin_decline = fundamentals.get("consecutive_margin_decline", 0)

        if gross_margin_trend < 0 and consecutive_margin_decline >= 3:
            met_conditions.append("gross margin declining 3+ quarters")

        # Condition 4: Balance sheet decay
        debt_to_ebitda = fundamentals.get("debt_to_ebitda", 0)

        if debt_to_ebitda > 4.0:
            met_conditions.append("debt/EBITDA > 4.0")

        # Condition 5: Revenue decline
        consecutive_revenue_decline = fundamentals.get("consecutive_revenue_decline", 0)

        if consecutive_revenue_decline >= 2:
            met_conditions.append("revenue declining 2+ quarters")

        return met_conditions

    def get_regime_description(self, regime: Regime) -> str:
        """Get human-readable description of regime."""
        descriptions = {
            Regime.STRUCTURAL_GROWTH:
                "✅ Structural Growth - Eligible for new capital allocation",
            Regime.MATURE_HOLD:
                "⏸️ Mature / Hold - Hold existing positions, no new capital",
            Regime.CAPITAL_DESTRUCTION:
                "❌ Capital Destruction - Exit signal, thesis broken",
        }
        return descriptions.get(regime, "Unknown regime")

    def get_regime_color(self, regime: Regime) -> str:
        """Get terminal color code for regime."""
        colors = {
            Regime.STRUCTURAL_GROWTH: "\033[92m",      # Green
            Regime.MATURE_HOLD: "\033[93m",             # Yellow
            Regime.CAPITAL_DESTRUCTION: "\033[91m",     # Red
        }
        colors_reset = "\033[0m"
        return colors.get(regime, "") + regime.value + colors_reset
