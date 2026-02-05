"""Technical indicators module for stock analysis.

This module provides implementations of common technical indicators used in
stock screening and analysis, including RSI, SMA, EMA, and volume analysis.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI).

    RSI is a momentum oscillator that measures the speed and magnitude of
    directional price changes. RSI oscillates between 0 and 100.

    Args:
        prices: Series of closing prices.
        period: Number of periods for RSI calculation (default: 14).

    Returns:
        Series of RSI values. Values range from 0 to 100.
        - RSI > 70: Overbought condition
        - RSI < 30: Oversold condition

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> rsi = calculate_rsi(prices, period=6)
        >>> print(rsi.iloc[-1])
    """
    if len(prices) < period + 1:
        logger.warning(f"Insufficient data for RSI calculation: {len(prices)} < {period + 1}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate exponential moving averages
    avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gains / avg_losses.replace(0, np.nan)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    SMA is the arithmetic mean of prices over a specified period.

    Args:
        prices: Series of closing prices.
        period: Number of periods for SMA calculation.

    Returns:
        Series of SMA values.

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> sma = calculate_sma(prices, period=3)
        >>> print(sma.iloc[-1])
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for SMA calculation: {len(prices)} < {period}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive to
    new information compared to SMA.

    Args:
        prices: Series of closing prices.
        period: Number of periods for EMA calculation.

    Returns:
        Series of EMA values.

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> ema = calculate_ema(prices, period=3)
        >>> print(ema.iloc[-1])
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for EMA calculation: {len(prices)} < {period}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

    return prices.ewm(span=period, min_periods=period, adjust=False).mean()


def find_swing_lows(prices: pd.Series, window: int = 30) -> List[float]:
    """Find swing lows (local minimums) in price data.

    A swing low is identified when a price point is lower than prices within
    a specified window on both sides.

    Args:
        prices: Series of closing prices.
        window: Number of periods on each side to check (default: 30).

    Returns:
        List of price levels representing swing lows, sorted ascending.

    Example:
        >>> prices = pd.Series([100, 95, 90, 92, 95, 93, 88, 90, 95])
        >>> lows = find_swing_lows(prices, window=2)
        >>> print(lows)  # [88.0, 90.0]
    """
    if len(prices) < window * 2 + 1:
        logger.warning(f"Insufficient data for swing low detection: {len(prices)} < {window * 2 + 1}")
        return []

    swing_lows = []

    # Find local minimums
    for i in range(window, len(prices) - window):
        current_price = prices.iloc[i]

        # Check if current price is lower than all prices in the window
        left_window = prices.iloc[i - window:i]
        right_window = prices.iloc[i + 1:i + window + 1]

        if (current_price <= left_window.min()) and (current_price <= right_window.min()):
            swing_lows.append(float(current_price))

    # Remove duplicates and sort
    swing_lows = sorted(list(set(swing_lows)))

    logger.debug(f"Found {len(swing_lows)} swing lows")
    return swing_lows


def detect_volume_spike(
    volumes: pd.Series,
    current_volume: float,
    threshold: float = 1.5
) -> bool:
    """Detect if current volume is significantly higher than average.

    A volume spike indicates increased trading activity, which can signal
    important price movements or accumulation/distribution.

    Args:
        volumes: Series of historical volume data.
        current_volume: Current period's volume.
        threshold: Multiplier for average volume (default: 1.5 = 150% of average).

    Returns:
        True if current volume exceeds threshold * average volume, False otherwise.

    Example:
        >>> volumes = pd.Series([1000000, 1100000, 950000, 1050000])
        >>> is_spike = detect_volume_spike(volumes, 1600000, threshold=1.5)
        >>> print(is_spike)  # True
    """
    if len(volumes) < 20:
        logger.warning(f"Insufficient volume data: {len(volumes)} < 20")
        return False

    # Calculate average volume (last 20 periods)
    avg_volume = volumes.iloc[-20:].mean()

    if avg_volume == 0:
        return False

    # Check if current volume exceeds threshold
    is_spike = current_volume >= (avg_volume * threshold)

    if is_spike:
        logger.debug(f"Volume spike detected: {current_volume:.0f} vs avg {avg_volume:.0f}")

    return is_spike


def calculate_support_strength(
    prices: pd.Series,
    support_level: float,
    tolerance: float = 0.02
) -> int:
    """Calculate the strength of a support level.

    Support strength is measured by the number of times the price has
    touched and bounced off a support level.

    Args:
        prices: Series of closing prices.
        support_level: The support price level to test.
        tolerance: Percentage tolerance for considering a touch (default: 2%).

    Returns:
        Integer representing the number of times support was tested.

    Example:
        >>> prices = pd.Series([100, 95, 96, 94, 97, 95, 98])
        >>> strength = calculate_support_strength(prices, 95.0, tolerance=0.02)
        >>> print(strength)
    """
    if len(prices) == 0:
        return 0

    # Calculate price range for support level
    lower_bound = support_level * (1 - tolerance)
    upper_bound = support_level * (1 + tolerance)

    # Count touches
    touches = ((prices >= lower_bound) & (prices <= upper_bound)).sum()

    return int(touches)


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and two outer bands
    (standard deviations away from the middle band).

    Args:
        prices: Series of closing prices.
        period: Number of periods for SMA (default: 20).
        num_std: Number of standard deviations for bands (default: 2.0).

    Returns:
        Tuple of (middle_band, upper_band, lower_band).

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> middle, upper, lower = calculate_bollinger_bands(prices, period=5)
        >>> print(f"Middle: {middle.iloc[-1]:.2f}")
    """
    if len(prices) < period:
        empty = pd.Series([np.nan] * len(prices), index=prices.index)
        return empty, empty, empty

    # Calculate middle band (SMA)
    middle_band = calculate_sma(prices, period)

    # Calculate standard deviation
    rolling_std = prices.rolling(window=period, min_periods=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)

    return middle_band, upper_band, lower_band


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages.

    Args:
        prices: Series of closing prices.
        fast_period: Fast EMA period (default: 12).
        slow_period: Slow EMA period (default: 26).
        signal_period: Signal line EMA period (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> macd, signal, histogram = calculate_macd(prices)
        >>> print(f"MACD: {macd.iloc[-1]:.2f}")
    """
    if len(prices) < slow_period + signal_period:
        empty = pd.Series([np.nan] * len(prices), index=prices.index)
        return empty, empty, empty

    # Calculate EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of an
    asset price for that period.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Number of periods for ATR calculation (default: 14).

    Returns:
        Series of ATR values.

    Example:
        >>> high = pd.Series([102, 104, 103])
        >>> low = pd.Series([98, 99, 100])
        >>> close = pd.Series([100, 102, 101])
        >>> atr = calculate_atr(high, low, close, period=2)
    """
    if len(high) < period + 1:
        return pd.Series([np.nan] * len(high), index=high.index)

    # Calculate True Range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR (SMA of True Range)
    atr = true_range.rolling(window=period, min_periods=period).mean()

    return atr
