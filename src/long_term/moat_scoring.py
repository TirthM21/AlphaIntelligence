"""
Business moat signal scoring.

Converts qualitative moat concepts into quantifiable proxies:
- Pricing power (gross margin stability)
- Customer lock-in (revenue retention)
- Platform effects (revenue per employee growth)
- Operating leverage
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MoatScorer:
    """Score business moat proxies."""

    def score_moat(
        self,
        ticker: str,
        fundamentals: Dict[str, Any]
    ) -> float:
        """
        Score business moat strength (0-10 bonus points).

        Args:
            ticker: Stock ticker
            fundamentals: Dict with margin, revenue, employee data

        Returns:
            Moat bonus score (0-10)
        """
        try:
            score = 0.0

            # Pricing power (0-3 points)
            score += self._score_pricing_power(fundamentals)

            # Customer lock-in (0-3 points)
            score += self._score_customer_lock_in(fundamentals)

            # Platform effects (0-3 points)
            score += self._score_platform_effects(fundamentals)

            # Operating leverage (0-2 points)
            score += self._score_operating_leverage(fundamentals)

            # Cap at 10 points
            return min(10.0, score)

        except Exception as e:
            logger.debug(f"Error scoring moat for {ticker}: {e}")
            return 0.0

    def _score_pricing_power(self, fundamentals: Dict[str, Any]) -> float:
        """
        Score pricing power via gross margin stability.

        Stable or expanding margins indicate pricing power.

        Returns:
            Score 0-3
        """
        gross_margin_current = fundamentals.get("gross_margin_current", 0)
        gross_margin_std_dev = fundamentals.get("gross_margin_std_dev", 0)
        gross_margin_trend = fundamentals.get("gross_margin_trend", 0)

        score = 0.0

        # Margin stability (0-2 points)
        # Low std dev (<2%) = strong pricing power
        if gross_margin_std_dev < 0.02:
            score += 2.0
        elif gross_margin_std_dev < 0.05:
            score += 1.0

        # Margin expansion (0-1 point)
        if gross_margin_trend > 0:
            score += 1.0

        return min(3.0, score)

    def _score_customer_lock_in(self, fundamentals: Dict[str, Any]) -> float:
        """
        Score customer lock-in via retention proxies.

        Low revenue volatility and high subscription revenue indicate lock-in.

        Returns:
            Score 0-3
        """
        revenue_volatility = fundamentals.get("revenue_volatility", 1.0)
        subscription_revenue_pct = fundamentals.get("subscription_revenue_pct", 0)

        score = 0.0

        # Revenue stability (0-2 points)
        # CV < 0.05 (very stable) = strong lock-in
        if revenue_volatility < 0.05:
            score += 2.0
        elif revenue_volatility < 0.10:
            score += 1.0
        elif revenue_volatility < 0.15:
            score += 0.5

        # Subscription revenue (0-1 point)
        # >50% = strong recurring revenue
        if subscription_revenue_pct > 0.50:
            score += 1.0
        elif subscription_revenue_pct > 0.30:
            score += 0.5

        return min(3.0, score)

    def _score_platform_effects(self, fundamentals: Dict[str, Any]) -> float:
        """
        Score platform effects via productivity metrics.

        Revenue per employee growth indicates network effects and scalability.

        Returns:
            Score 0-3
        """
        revenue_per_employee_growth = fundamentals.get(
            "revenue_per_employee_growth", 0
        )
        employee_count_trend = fundamentals.get("employee_count_trend", 0)

        score = 0.0

        # Revenue per employee growth (0-2 points)
        # 10%+ annual growth = strong platform effects
        if revenue_per_employee_growth > 0.10:
            score += 2.0
        elif revenue_per_employee_growth > 0.05:
            score += 1.0

        # Employee efficiency (0-1 point)
        # Headcount growth < revenue growth = operating leverage
        if revenue_per_employee_growth > 0:
            score += 1.0

        return min(3.0, score)

    def _score_operating_leverage(self, fundamentals: Dict[str, Any]) -> float:
        """
        Score operating leverage.

        Revenue growth > opex growth indicates improving efficiency.

        Returns:
            Score 0-2
        """
        revenue_growth_rate = fundamentals.get("revenue_growth_rate", 0)
        opex_growth_rate = fundamentals.get("opex_growth_rate", 0)

        score = 0.0

        # Operating leverage
        if opex_growth_rate > 0:
            leverage_ratio = revenue_growth_rate / opex_growth_rate
            if leverage_ratio > 1.5:  # Revenue growing 50%+ faster
                score += 2.0
            elif leverage_ratio > 1.0:  # Revenue growing faster
                score += 1.0
        elif revenue_growth_rate > 0:
            # Revenue growing with flat or declining opex = max leverage
            score += 2.0

        return min(2.0, score)

    def get_moat_description(self, score: float) -> str:
        """Get moat strength description."""
        if score >= 8:
            return "🏰 Elite moat (strong competitive advantage)"
        elif score >= 5:
            return "🛡️ Solid moat (sustainable competitive advantage)"
        elif score >= 3:
            return "🔒 Modest moat (some competitive protection)"
        elif score > 0:
            return "⚠️ Weak moat (limited protection)"
        else:
            return "❌ No moat (commodity business)"
