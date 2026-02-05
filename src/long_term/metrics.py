"""
Long-term metric calculations for compounder identification.

Provides helpers for calculating:
- ROIC (Return on Invested Capital)
- WACC (Weighted Average Cost of Capital)
- CAGR (Compound Annual Growth Rate)
- Other capital efficiency metrics
"""

from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import statistics
import math


class MetricsCalculator:
    """Calculate long-term fundamental metrics for investment quality scoring."""

    @staticmethod
    def calculate_cagr(
        starting_value: float,
        ending_value: float,
        periods: int
    ) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            starting_value: Initial value
            ending_value: Final value
            periods: Number of years

        Returns:
            CAGR as decimal (0.05 = 5%)
        """
        if starting_value <= 0 or ending_value <= 0 or periods <= 0:
            return 0.0

        try:
            cagr = (ending_value / starting_value) ** (1 / periods) - 1
            return max(cagr, -0.99)  # Cap at -99% to prevent extreme values
        except (ZeroDivisionError, ValueError):
            return 0.0

    @staticmethod
    def calculate_roic(
        nopat: float,
        invested_capital: float
    ) -> float:
        """
        Calculate Return on Invested Capital.

        ROIC = NOPAT / Invested Capital

        Args:
            nopat: Net Operating Profit After Tax
            invested_capital: Shareholders' equity + Total debt - Cash

        Returns:
            ROIC as decimal (0.15 = 15%)
        """
        if invested_capital <= 0:
            return 0.0

        roic = nopat / invested_capital
        return max(min(roic, 2.0), -0.5)  # Cap between -50% and 200%

    @staticmethod
    def calculate_wacc(
        cost_of_equity: float,
        cost_of_debt: float,
        market_value_equity: float,
        market_value_debt: float,
        tax_rate: float = 0.21
    ) -> float:
        """
        Calculate Weighted Average Cost of Capital.

        WACC = (E/V × Re) + (D/V × Rd × (1 - Tc))

        Args:
            cost_of_equity: Expected return on equity (CAPM)
            cost_of_debt: Interest rate on debt
            market_value_equity: Market value of equity
            market_value_debt: Market value of debt
            tax_rate: Corporate tax rate (default 21% for US)

        Returns:
            WACC as decimal (0.08 = 8%)
        """
        total_value = market_value_equity + market_value_debt

        if total_value <= 0:
            return cost_of_equity  # Return cost of equity if no debt

        weight_equity = market_value_equity / total_value
        weight_debt = market_value_debt / total_value

        wacc = (weight_equity * cost_of_equity) + \
               (weight_debt * cost_of_debt * (1 - tax_rate))

        return max(wacc, 0.01)  # Minimum 1%

    @staticmethod
    def calculate_fcf_margin(
        free_cash_flow: float,
        revenue: float
    ) -> float:
        """
        Calculate Free Cash Flow Margin.

        FCF Margin = FCF / Revenue

        Args:
            free_cash_flow: Annual free cash flow
            revenue: Annual revenue

        Returns:
            FCF Margin as decimal (0.20 = 20%)
        """
        if revenue <= 0:
            return 0.0

        margin = free_cash_flow / revenue
        return max(min(margin, 1.0), -0.5)  # Cap between -50% and 100%

    @staticmethod
    def calculate_roic_wacc_spread(
        roic: float,
        wacc: float
    ) -> float:
        """
        Calculate ROIC - WACC spread (measure of competitive advantage).

        Higher spread = stronger moat and better capital allocation.

        Args:
            roic: Return on Invested Capital
            wacc: Weighted Average Cost of Capital

        Returns:
            Spread as decimal (0.10 = 10%)
        """
        return roic - wacc

    @staticmethod
    def calculate_gross_margin_stability(
        gross_margins: List[float],
        periods: int = 12
    ) -> Tuple[float, float]:
        """
        Calculate gross margin stability over period.

        Args:
            gross_margins: List of quarterly gross margins
            periods: Number of quarters to analyze (default 12 = 3 years)

        Returns:
            Tuple of (current_margin, std_dev)
        """
        if not gross_margins or len(gross_margins) < periods:
            return (0.0, 0.0)

        recent_margins = gross_margins[-periods:]
        current = recent_margins[-1]

        # Calculate standard deviation (pure Python)
        if len(recent_margins) > 1:
            std_dev = statistics.stdev(recent_margins)
        else:
            std_dev = 0.0

        return (current, std_dev)

    @staticmethod
    def calculate_revenue_retention(
        revenues: List[float],
        periods: int = 12
    ) -> float:
        """
        Calculate revenue volatility as proxy for customer retention.

        Lower volatility = better retention.

        Args:
            revenues: List of quarterly revenues
            periods: Number of quarters to analyze (default 12 = 3 years)

        Returns:
            Coefficient of variation (std dev / mean)
        """
        if not revenues or len(revenues) < periods:
            return 1.0  # Max volatility if insufficient data

        recent_revenues = revenues[-periods:]
        if not recent_revenues:
            return 1.0

        mean_revenue = statistics.mean(recent_revenues)
        if mean_revenue <= 0:
            return 1.0

        if len(recent_revenues) > 1:
            std_dev = statistics.stdev(recent_revenues)
        else:
            std_dev = 0.0
        cv = std_dev / mean_revenue  # Coefficient of variation

        return min(cv, 2.0)  # Cap at 200% variation

    @staticmethod
    def calculate_operating_leverage(
        revenue_growth_rate: float,
        opex_growth_rate: float
    ) -> float:
        """
        Calculate operating leverage.

        Operating leverage = Revenue growth / Opex growth
        >1.0 indicates improving operational efficiency.

        Args:
            revenue_growth_rate: YoY revenue growth
            opex_growth_rate: YoY opex growth

        Returns:
            Operating leverage ratio
        """
        if opex_growth_rate <= 0:
            return 1.0 if revenue_growth_rate > 0 else 0.0

        leverage = revenue_growth_rate / opex_growth_rate
        return max(min(leverage, 5.0), 0.0)  # Cap between 0 and 5x

    @staticmethod
    def calculate_revenue_per_employee_growth(
        revenues: List[float],
        employees: List[float]
    ) -> Optional[float]:
        """
        Calculate revenue per employee growth.

        Indicator of improving productivity and scalability.

        Args:
            revenues: List of annual revenues
            employees: List of employee counts

        Returns:
            CAGR of revenue per employee, or None if insufficient data
        """
        if (not revenues or not employees or
            len(revenues) < 2 or len(employees) < 2 or
            employees[0] <= 0 or employees[-1] <= 0):
            return None

        rpe_start = revenues[0] / employees[0]
        rpe_end = revenues[-1] / employees[-1]

        if rpe_start <= 0:
            return None

        years = len(revenues) - 1
        if years <= 0:
            return None

        return MetricsCalculator.calculate_cagr(rpe_start, rpe_end, years)

    @staticmethod
    def calculate_debt_ratios(
        total_debt: float,
        ebitda: float,
        interest_expense: float
    ) -> Tuple[float, float]:
        """
        Calculate leverage ratios.

        Args:
            total_debt: Total debt (short-term + long-term)
            ebitda: Earnings Before Interest, Tax, Depreciation, Amortization
            interest_expense: Annual interest expense

        Returns:
            Tuple of (debt_to_ebitda, interest_coverage)
        """
        # Debt/EBITDA
        if ebitda > 0:
            debt_to_ebitda = total_debt / ebitda
            debt_to_ebitda = min(debt_to_ebitda, 10.0)  # Cap at 10x
        else:
            debt_to_ebitda = 10.0

        # Interest Coverage
        if interest_expense > 0:
            interest_coverage = ebitda / interest_expense
            interest_coverage = max(interest_coverage, 0.0)
        else:
            interest_coverage = 100.0  # Assume excellent if no interest expense

        return (debt_to_ebitda, interest_coverage)

    @staticmethod
    def calculate_net_margin_trend(
        net_margins: List[float],
        periods: int = 12
    ) -> float:
        """
        Calculate net margin trend.

        Positive trend indicates improving profitability.

        Args:
            net_margins: List of quarterly net margins
            periods: Number of quarters to analyze

        Returns:
            Linear regression slope (quarterly change rate)
        """
        if not net_margins or len(net_margins) < periods:
            return 0.0

        recent_margins = net_margins[-periods:]
        if len(recent_margins) < 2:
            return 0.0

        # Calculate linear regression slope using least-squares method
        try:
            n = len(recent_margins)
            x = list(range(n))

            # Calculate sums needed for linear regression
            sum_x = sum(x)
            sum_y = sum(recent_margins)
            sum_xy = sum(xi * yi for xi, yi in zip(x, recent_margins))
            sum_x2 = sum(xi * xi for xi in x)

            # Slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
            denominator = (n * sum_x2) - (sum_x * sum_x)
            if denominator == 0:
                return 0.0

            slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
            return slope
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def scale_linear(
        value: float,
        min_val: float,
        max_val: float,
        min_score: float = 0.0,
        max_score: float = 10.0,
        invert: bool = False
    ) -> float:
        """
        Linear scaling utility for converting metrics to scores.

        Args:
            value: Metric value
            min_val: Minimum metric value (maps to min_score)
            max_val: Maximum metric value (maps to max_score)
            min_score: Minimum output score
            max_score: Maximum output score
            invert: If True, reverse the mapping

        Returns:
            Scaled score between min_score and max_score
        """
        if min_val == max_val:
            return (min_score + max_score) / 2

        # Normalize to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(min(normalized, 1.0), 0.0)  # Clamp to 0-1

        if invert:
            normalized = 1.0 - normalized

        # Scale to output range
        score = min_score + (normalized * (max_score - min_score))
        return score
