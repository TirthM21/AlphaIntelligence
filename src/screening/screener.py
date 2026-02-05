"""Stock screening module for identifying undervalued stocks at support levels.

This module combines fundamental analysis (value scoring) with technical analysis
(support level detection) to identify high-probability buying opportunities.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.storage import StockDatabase
from .indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    detect_volume_spike,
    find_swing_lows
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_value_score(fundamentals: Dict[str, any]) -> float:
    """Calculate value score based on fundamental metrics.

    The value score combines P/E ratio, P/B ratio, and FCF yield to assess
    whether a stock is undervalued. Higher scores indicate better value.

    Scoring criteria:
    - P/E ratio: < 15 gets max points, scaled up to 30
    - P/B ratio: < 1.5 gets max points, scaled up to 3.0
    - FCF yield: > 5% gets max points, scaled down to 0%
    - Debt/Equity: < 50% gets bonus points

    Args:
        fundamentals: Dictionary containing fundamental metrics with keys:
                     'pe_ratio', 'pb_ratio', 'fcf_yield', 'debt_equity'.

    Returns:
        Float score between 0 and 100. Higher is better.
        - 80-100: Excellent value
        - 60-79: Good value
        - 40-59: Fair value
        - 0-39: Poor value

    Example:
        >>> fundamentals = {
        ...     'pe_ratio': 12.5,
        ...     'pb_ratio': 1.2,
        ...     'fcf_yield': 6.5,
        ...     'debt_equity': 45.0
        ... }
        >>> score = calculate_value_score(fundamentals)
        >>> print(f"Value score: {score:.1f}")
    """
    score = 0.0
    max_score = 100.0

    # Extract metrics with defaults
    pe_ratio = fundamentals.get('pe_ratio')
    pb_ratio = fundamentals.get('pb_ratio')
    fcf_yield = fundamentals.get('fcf_yield')
    debt_equity = fundamentals.get('debt_equity')

    # P/E Ratio Score (40 points max)
    # Best: P/E < 15, Acceptable: P/E < 30
    if pe_ratio is not None and pe_ratio > 0:
        if pe_ratio <= 15:
            score += 40
        elif pe_ratio <= 30:
            # Linear scale from 40 down to 20
            score += 40 - ((pe_ratio - 15) / 15) * 20
        elif pe_ratio <= 50:
            # Linear scale from 20 down to 0
            score += 20 - ((pe_ratio - 30) / 20) * 20
        else:
            score += 0
    else:
        logger.debug("P/E ratio missing or invalid")

    # P/B Ratio Score (30 points max)
    # Best: P/B < 1.5, Acceptable: P/B < 3.0
    if pb_ratio is not None and pb_ratio > 0:
        if pb_ratio <= 1.5:
            score += 30
        elif pb_ratio <= 3.0:
            # Linear scale from 30 down to 10
            score += 30 - ((pb_ratio - 1.5) / 1.5) * 20
        elif pb_ratio <= 5.0:
            # Linear scale from 10 down to 0
            score += 10 - ((pb_ratio - 3.0) / 2.0) * 10
        else:
            score += 0
    else:
        logger.debug("P/B ratio missing or invalid")

    # FCF Yield Score (20 points max)
    # Best: FCF yield > 5%, Acceptable: FCF yield > 0%
    if fcf_yield is not None:
        if fcf_yield >= 5.0:
            score += 20
        elif fcf_yield > 0:
            # Linear scale from 0 to 20
            score += (fcf_yield / 5.0) * 20
        else:
            score += 0
    else:
        logger.debug("FCF yield missing")

    # Debt/Equity Ratio Bonus (10 points max)
    # Best: Debt/Equity < 50%, Acceptable: < 100%
    if debt_equity is not None:
        if debt_equity <= 50:
            score += 10
        elif debt_equity <= 100:
            # Linear scale from 10 down to 5
            score += 10 - ((debt_equity - 50) / 50) * 5
        elif debt_equity <= 200:
            # Linear scale from 5 down to 0
            score += 5 - ((debt_equity - 100) / 100) * 5
        else:
            score += 0
    else:
        logger.debug("Debt/Equity ratio missing")

    # Normalize to 0-100
    score = min(max(score, 0.0), max_score)

    logger.debug(
        f"Value score: {score:.1f} (PE: {pe_ratio}, PB: {pb_ratio}, "
        f"FCF: {fcf_yield}, D/E: {debt_equity})"
    )

    return round(score, 2)


def detect_support_levels(price_df: pd.DataFrame) -> List[float]:
    """Detect support levels from price history.

    Support levels are identified using:
    1. Swing lows (local minimums over 30-day window)
    2. 50-day and 200-day moving averages
    3. Recent significant lows

    Args:
        price_df: DataFrame with columns: Date, Open, High, Low, Close, Volume.
                 Must have at least 200 days of data for full analysis.

    Returns:
        List of support price levels, sorted from lowest to highest.
        Returns empty list if insufficient data or errors occur.

    Example:
        >>> price_df = pd.DataFrame({
        ...     'Date': pd.date_range('2023-01-01', periods=250),
        ...     'Close': [100 + i * 0.1 for i in range(250)],
        ...     'High': [102 + i * 0.1 for i in range(250)],
        ...     'Low': [98 + i * 0.1 for i in range(250)]
        ... })
        >>> supports = detect_support_levels(price_df)
        >>> print(supports)
    """
    if price_df.empty:
        logger.warning("Empty price DataFrame provided")
        return []

    required_cols = ['Close', 'High', 'Low']
    missing_cols = [col for col in required_cols if col not in price_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return []

    try:
        support_levels = []
        close_prices = price_df['Close']

        # 1. Find swing lows (local minimums)
        if len(close_prices) >= 61:  # Need at least 30 days on each side + 1
            swing_lows = find_swing_lows(close_prices, window=30)
            support_levels.extend(swing_lows)
            logger.debug(f"Found {len(swing_lows)} swing lows")

        # 2. Calculate 50-day moving average
        if len(close_prices) >= 50:
            sma_50 = calculate_sma(close_prices, period=50)
            if not sma_50.isna().all():
                support_levels.append(float(sma_50.iloc[-1]))
                logger.debug(f"Added SMA-50: {sma_50.iloc[-1]:.2f}")

        # 3. Calculate 200-day moving average
        if len(close_prices) >= 200:
            sma_200 = calculate_sma(close_prices, period=200)
            if not sma_200.isna().all():
                support_levels.append(float(sma_200.iloc[-1]))
                logger.debug(f"Added SMA-200: {sma_200.iloc[-1]:.2f}")

        # 4. Add significant recent lows (last 90 days)
        if len(close_prices) >= 90:
            recent_low = close_prices.iloc[-90:].min()
            support_levels.append(float(recent_low))
            logger.debug(f"Added recent low: {recent_low:.2f}")

        # 5. Add 52-week low if available
        if len(close_prices) >= 252:  # Trading days in a year
            week_52_low = close_prices.iloc[-252:].min()
            support_levels.append(float(week_52_low))
            logger.debug(f"Added 52-week low: {week_52_low:.2f}")

        # Remove duplicates and sort
        # Group similar levels (within 0.5% of each other)
        if support_levels:
            support_levels = sorted(set(support_levels))
            consolidated = [support_levels[0]]

            for level in support_levels[1:]:
                # If level is more than 0.5% different from last, add it
                if abs(level - consolidated[-1]) / consolidated[-1] > 0.005:
                    consolidated.append(level)

            logger.info(f"Detected {len(consolidated)} support levels")
            return consolidated

        return []

    except Exception as e:
        logger.error(f"Error detecting support levels: {e}")
        return []


def calculate_support_score(
    current_price: float,
    support_levels: List[float],
    rsi: Optional[float] = None,
    volume_spike: bool = False,
    price_history: Optional[pd.DataFrame] = None
) -> float:
    """Calculate support score based on technical factors.

    The support score assesses how close the stock is to support levels and
    whether technical indicators suggest a buying opportunity.

    Scoring criteria:
    - Distance from nearest support: Closer is better (max 40 points)
    - RSI oversold (< 40): Bonus points (max 30 points)
    - Volume spike on dip: Bonus points (20 points)
    - Price at support level: Bonus points (10 points)

    Args:
        current_price: Current stock price.
        support_levels: List of identified support price levels.
        rsi: Current RSI value (0-100). Optional.
        volume_spike: Whether volume is significantly elevated. Optional.
        price_history: DataFrame with price data for additional analysis. Optional.

    Returns:
        Float score between 0 and 100. Higher indicates better technical setup.
        - 80-100: Excellent technical setup
        - 60-79: Good technical setup
        - 40-59: Fair technical setup
        - 0-39: Poor technical setup

    Example:
        >>> score = calculate_support_score(
        ...     current_price=95.5,
        ...     support_levels=[90, 95, 100],
        ...     rsi=35,
        ...     volume_spike=True
        ... )
        >>> print(f"Support score: {score:.1f}")
    """
    if current_price <= 0:
        logger.warning("Invalid current price")
        return 0.0

    score = 0.0

    # 1. Distance from nearest support (40 points max)
    if support_levels:
        # Find nearest support below current price
        supports_below = [s for s in support_levels if s <= current_price]

        if supports_below:
            nearest_support = max(supports_below)
            distance_pct = ((current_price - nearest_support) / nearest_support) * 100

            if distance_pct <= 1.0:
                # Within 1% of support - excellent
                score += 40
                logger.debug(f"At support level: {distance_pct:.2f}% away")
            elif distance_pct <= 3.0:
                # Within 3% of support - very good
                score += 40 - ((distance_pct - 1.0) / 2.0) * 15
                logger.debug(f"Near support: {distance_pct:.2f}% away")
            elif distance_pct <= 5.0:
                # Within 5% of support - good
                score += 25 - ((distance_pct - 3.0) / 2.0) * 10
                logger.debug(f"Approaching support: {distance_pct:.2f}% away")
            elif distance_pct <= 10.0:
                # Within 10% of support - fair
                score += 15 - ((distance_pct - 5.0) / 5.0) * 15
                logger.debug(f"Distant from support: {distance_pct:.2f}% away")
            else:
                logger.debug(f"Far from support: {distance_pct:.2f}% away")
        else:
            # Current price below all support levels
            logger.debug("Price below all known support levels")
            score += 20  # Potential new support forming

    # 2. RSI Score (30 points max)
    if rsi is not None:
        if rsi <= 30:
            # Deeply oversold - excellent buying opportunity
            score += 30
            logger.debug(f"Deeply oversold: RSI {rsi:.1f}")
        elif rsi <= 40:
            # Oversold - good buying opportunity
            score += 30 - ((rsi - 30) / 10) * 10
            logger.debug(f"Oversold: RSI {rsi:.1f}")
        elif rsi <= 50:
            # Approaching oversold
            score += 20 - ((rsi - 40) / 10) * 10
            logger.debug(f"Neutral RSI: {rsi:.1f}")
        elif rsi >= 70:
            # Overbought - negative signal
            score -= 10
            logger.debug(f"Overbought: RSI {rsi:.1f}")

    # 3. Volume Spike Bonus (20 points)
    if volume_spike:
        score += 20
        logger.debug("Volume spike detected - accumulation signal")

    # 4. Multiple support confluence (10 points bonus)
    if support_levels:
        # Check if multiple support levels are clustered near current price
        supports_nearby = [
            s for s in support_levels
            if abs(current_price - s) / current_price <= 0.03
        ]
        if len(supports_nearby) >= 2:
            score += 10
            logger.debug(f"Multiple support confluence: {len(supports_nearby)} levels")

    # 5. Additional analysis with price history
    if price_history is not None and not price_history.empty:
        try:
            # Check if price is in lower 25% of recent range
            if len(price_history) >= 90:
                recent_high = price_history['High'].iloc[-90:].max()
                recent_low = price_history['Low'].iloc[-90:].min()
                price_range = recent_high - recent_low

                if price_range > 0:
                    position = (current_price - recent_low) / price_range

                    if position <= 0.25:
                        score += 10
                        logger.debug(f"Price in lower 25% of range: {position:.2%}")
        except Exception as e:
            logger.warning(f"Error in price range analysis: {e}")

    # Normalize to 0-100
    score = min(max(score, 0.0), 100.0)

    logger.debug(f"Support score: {score:.1f}")
    return round(score, 2)


def screen_candidates(
    db: StockDatabase,
    tickers_list: List[str],
    value_weight: float = 0.7,
    support_weight: float = 0.3,
    min_data_days: int = 200
) -> pd.DataFrame:
    """Screen stocks and rank by combined value and technical scores.

    This is the main screening function that combines fundamental value analysis
    with technical support analysis to identify high-probability buying opportunities.

    Args:
        db: StockDatabase instance for data retrieval.
        tickers_list: List of stock ticker symbols to screen.
        value_weight: Weight for value score in final ranking (default: 0.7).
        support_weight: Weight for support score in final ranking (default: 0.3).
        min_data_days: Minimum days of price data required (default: 200).

    Returns:
        DataFrame with columns: ticker, name, sector, current_price,
        value_score, support_score, buy_signal, rsi, nearest_support.
        Sorted by buy_signal descending (best opportunities first).
        Returns empty DataFrame if no valid candidates found.

    Example:
        >>> from src.data.storage import StockDatabase
        >>> db = StockDatabase()
        >>> results = screen_candidates(db, ["AAPL", "MSFT", "GOOGL"])
        >>> print(results.head())
        >>> print(f"Top candidate: {results.iloc[0]['ticker']}")
    """
    if not tickers_list:
        logger.warning("Empty ticker list provided")
        return pd.DataFrame()

    if value_weight + support_weight != 1.0:
        logger.warning(f"Weights don't sum to 1.0: {value_weight} + {support_weight}")
        # Normalize weights
        total = value_weight + support_weight
        value_weight /= total
        support_weight /= total

    logger.info(f"Screening {len(tickers_list)} candidates...")

    results = []

    for ticker in tickers_list:
        try:
            logger.debug(f"Processing {ticker}...")

            # Get fundamental data
            fundamentals = db.get_latest_fundamentals(ticker)
            if not fundamentals:
                logger.warning(f"No fundamentals found for {ticker}")
                continue

            # Get price history
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=min_data_days + 100)  # Extra buffer

            price_history = db.get_price_history(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if price_history.empty or len(price_history) < min_data_days:
                logger.warning(
                    f"Insufficient price data for {ticker}: {len(price_history)} days"
                )
                continue

            # Calculate value score
            value_score = calculate_value_score(fundamentals)

            # Detect support levels
            support_levels = detect_support_levels(price_history)

            # Calculate technical indicators
            current_price = fundamentals.get('current_price')
            if current_price is None or current_price <= 0:
                logger.warning(f"Invalid current price for {ticker}")
                continue

            # Calculate RSI
            rsi = None
            if len(price_history) >= 15:
                rsi_series = calculate_rsi(price_history['Close'], period=14)
                if not rsi_series.isna().all():
                    rsi = float(rsi_series.iloc[-1])

            # Detect volume spike
            volume_spike = False
            if 'Volume' in price_history.columns and len(price_history) >= 20:
                current_volume = price_history['Volume'].iloc[-1]
                volume_spike = detect_volume_spike(
                    price_history['Volume'],
                    current_volume,
                    threshold=1.5
                )

            # Calculate support score
            support_score = calculate_support_score(
                current_price=current_price,
                support_levels=support_levels,
                rsi=rsi,
                volume_spike=volume_spike,
                price_history=price_history
            )

            # Calculate combined buy signal
            buy_signal = (value_score * value_weight) + (support_score * support_weight)

            # Find nearest support
            nearest_support = None
            if support_levels:
                supports_below = [s for s in support_levels if s <= current_price]
                if supports_below:
                    nearest_support = max(supports_below)

            # Compile results
            results.append({
                'ticker': ticker,
                'name': fundamentals.get('name', ticker),
                'sector': fundamentals.get('sector', 'Unknown'),
                'current_price': current_price,
                'value_score': value_score,
                'support_score': support_score,
                'buy_signal': round(buy_signal, 2),
                'rsi': round(rsi, 2) if rsi is not None else None,
                'nearest_support': round(nearest_support, 2) if nearest_support else None,
                'pe_ratio': fundamentals.get('pe_ratio'),
                'pb_ratio': fundamentals.get('pb_ratio'),
                'data_points': len(price_history)
            })

            logger.info(
                f"{ticker}: Buy Signal={buy_signal:.1f} "
                f"(Value={value_score:.1f}, Support={support_score:.1f})"
            )

        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
            continue

    if not results:
        logger.warning("No valid screening results")
        return pd.DataFrame()

    # Create DataFrame and sort by buy signal
    df = pd.DataFrame(results)
    df = df.sort_values('buy_signal', ascending=False).reset_index(drop=True)

    logger.info(f"Successfully screened {len(df)} candidates")

    return df
