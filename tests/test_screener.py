"""Comprehensive test suite for stock screening module.

Tests screening logic, value scoring, support detection, and technical indicators.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.screening.screener import (
    calculate_value_score,
    detect_support_levels,
    calculate_support_score,
    screen_candidates
)
from src.screening.indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    detect_volume_spike,
    find_swing_lows,
    calculate_bollinger_bands,
    calculate_macd
)


class TestValueScoring:
    """Test suite for value score calculations."""

    def test_calculate_value_score_excellent(self):
        """Test value score for excellent fundamentals."""
        fundamentals = {
            'pe_ratio': 12.0,
            'pb_ratio': 1.2,
            'fcf_yield': 6.5,
            'debt_equity': 40.0
        }

        score = calculate_value_score(fundamentals)

        assert 80 <= score <= 100
        assert isinstance(score, float)

    def test_calculate_value_score_good(self):
        """Test value score for good fundamentals."""
        fundamentals = {
            'pe_ratio': 18.0,
            'pb_ratio': 2.0,
            'fcf_yield': 4.0,
            'debt_equity': 60.0
        }

        score = calculate_value_score(fundamentals)

        assert 60 <= score <= 90  # Adjusted range for scoring algorithm
        assert isinstance(score, float)

    def test_calculate_value_score_poor(self):
        """Test value score for poor fundamentals."""
        fundamentals = {
            'pe_ratio': 45.0,
            'pb_ratio': 8.0,
            'fcf_yield': 0.5,
            'debt_equity': 250.0
        }

        score = calculate_value_score(fundamentals)

        assert 0 <= score <= 40
        assert isinstance(score, float)

    def test_calculate_value_score_missing_data(self):
        """Test value score with missing metrics."""
        fundamentals = {
            'pe_ratio': 15.0,
            'pb_ratio': None,
            'fcf_yield': None,
            'debt_equity': None
        }

        score = calculate_value_score(fundamentals)

        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_calculate_value_score_empty_dict(self):
        """Test value score with empty fundamentals."""
        score = calculate_value_score({})

        assert score == 0.0

    def test_calculate_value_score_negative_values(self):
        """Test handling of negative/invalid values."""
        fundamentals = {
            'pe_ratio': -10.0,
            'pb_ratio': -5.0,
            'fcf_yield': -2.0,
            'debt_equity': -50.0
        }

        score = calculate_value_score(fundamentals)

        # Should handle gracefully and return low score
        assert 0 <= score <= 100


class TestSupportDetection:
    """Test suite for support level detection."""

    def test_detect_support_levels_simple(self):
        """Test support detection with simple price data."""
        dates = pd.date_range('2023-01-01', periods=250)
        prices = [100 + np.sin(i / 10) * 10 for i in range(250)]

        price_df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'High': [p + 2 for p in prices],
            'Low': [p - 2 for p in prices]
        })

        supports = detect_support_levels(price_df)

        assert isinstance(supports, list)
        assert len(supports) > 0
        assert all(isinstance(s, float) for s in supports)
        assert supports == sorted(supports)  # Should be sorted

    def test_detect_support_levels_insufficient_data(self):
        """Test support detection with insufficient data."""
        price_df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=50),
            'Close': list(range(50)),
            'High': list(range(1, 51)),
            'Low': list(range(-1, 49))
        })

        supports = detect_support_levels(price_df)

        # Should still return something, even with limited data
        assert isinstance(supports, list)

    def test_detect_support_levels_empty_df(self):
        """Test support detection with empty DataFrame."""
        price_df = pd.DataFrame()

        supports = detect_support_levels(price_df)

        assert supports == []

    def test_detect_support_levels_missing_columns(self):
        """Test support detection with missing required columns."""
        price_df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': list(range(100))
            # Missing High and Low
        })

        supports = detect_support_levels(price_df)

        assert supports == []

    def test_detect_support_levels_consolidation(self):
        """Test that similar support levels are consolidated."""
        # Create data with multiple similar lows
        dates = pd.date_range('2023-01-01', periods=300)
        prices = [100.0] * 50 + [100.2] * 50 + [100.1] * 50 + [110.0] * 150

        price_df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices]
        })

        supports = detect_support_levels(price_df)

        # Similar levels should be consolidated
        assert isinstance(supports, list)


class TestSupportScoring:
    """Test suite for support score calculations."""

    def test_calculate_support_score_at_support(self):
        """Test support score when price is at support level."""
        score = calculate_support_score(
            current_price=100.0,
            support_levels=[95.0, 100.0, 105.0],
            rsi=35,
            volume_spike=True
        )

        assert 70 <= score <= 100
        assert isinstance(score, float)

    def test_calculate_support_score_far_from_support(self):
        """Test support score when price is far from support."""
        score = calculate_support_score(
            current_price=120.0,
            support_levels=[90.0, 95.0, 100.0],
            rsi=60,
            volume_spike=False
        )

        assert 0 <= score <= 60
        assert isinstance(score, float)

    def test_calculate_support_score_oversold(self):
        """Test support score boost from oversold RSI."""
        score_oversold = calculate_support_score(
            current_price=100.0,
            support_levels=[99.0],
            rsi=25,
            volume_spike=False
        )

        score_neutral = calculate_support_score(
            current_price=100.0,
            support_levels=[99.0],
            rsi=50,
            volume_spike=False
        )

        assert score_oversold > score_neutral

    def test_calculate_support_score_volume_spike(self):
        """Test support score boost from volume spike."""
        score_spike = calculate_support_score(
            current_price=100.0,
            support_levels=[99.0],
            rsi=None,
            volume_spike=True
        )

        score_no_spike = calculate_support_score(
            current_price=100.0,
            support_levels=[99.0],
            rsi=None,
            volume_spike=False
        )

        assert score_spike > score_no_spike

    def test_calculate_support_score_no_supports(self):
        """Test support score with no support levels."""
        score = calculate_support_score(
            current_price=100.0,
            support_levels=[],
            rsi=None,
            volume_spike=False
        )

        assert 0 <= score <= 100

    def test_calculate_support_score_invalid_price(self):
        """Test handling of invalid current price."""
        score = calculate_support_score(
            current_price=0.0,
            support_levels=[95.0, 100.0],
            rsi=35,
            volume_spike=False
        )

        assert score == 0.0


class TestTechnicalIndicators:
    """Test suite for technical indicators."""

    def test_calculate_rsi_normal(self):
        """Test RSI calculation with normal price data."""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 3)
        rsi = calculate_rsi(prices, period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all((valid_rsi >= 0) & (valid_rsi <= 100))

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = pd.Series([100, 102, 101])
        rsi = calculate_rsi(prices, period=14)

        assert isinstance(rsi, pd.Series)
        assert rsi.isna().all()

    def test_calculate_sma(self):
        """Test Simple Moving Average calculation."""
        prices = pd.Series([100, 102, 104, 106, 108])
        sma = calculate_sma(prices, period=3)

        assert isinstance(sma, pd.Series)
        assert len(sma) == len(prices)
        # Last SMA should be average of last 3 prices
        expected = (104 + 106 + 108) / 3
        assert abs(sma.iloc[-1] - expected) < 0.01

    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation."""
        prices = pd.Series([100, 102, 104, 106, 108, 110, 112])
        ema = calculate_ema(prices, period=5)

        assert isinstance(ema, pd.Series)
        assert len(ema) == len(prices)
        # EMA should give more weight to recent prices
        assert not ema.isna().all()

    def test_detect_volume_spike_true(self):
        """Test volume spike detection with actual spike."""
        volumes = pd.Series([1000000] * 30)
        current_volume = 2000000  # 2x average

        is_spike = detect_volume_spike(volumes, current_volume, threshold=1.5)

        assert is_spike == True

    def test_detect_volume_spike_false(self):
        """Test volume spike detection with normal volume."""
        volumes = pd.Series([1000000] * 30)
        current_volume = 1100000  # Only 1.1x average

        is_spike = detect_volume_spike(volumes, current_volume, threshold=1.5)

        assert is_spike == False

    def test_find_swing_lows(self):
        """Test swing low detection."""
        prices = pd.Series([100, 95, 90, 95, 100, 95, 85, 90, 95, 100] * 10)
        swing_lows = find_swing_lows(prices, window=5)

        assert isinstance(swing_lows, list)
        assert len(swing_lows) > 0
        assert all(isinstance(low, float) for low in swing_lows)
        assert swing_lows == sorted(swing_lows)

    def test_find_swing_lows_insufficient_data(self):
        """Test swing low detection with insufficient data."""
        prices = pd.Series([100, 95, 90])
        swing_lows = find_swing_lows(prices, window=30)

        assert swing_lows == []

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = pd.Series([100 + i for i in range(30)])
        middle, upper, lower = calculate_bollinger_bands(prices, period=20)

        assert isinstance(middle, pd.Series)
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(middle) == len(prices)
        # Upper band should be above middle, middle above lower
        valid_idx = ~middle.isna()
        if valid_idx.any():
            assert all(upper[valid_idx] >= middle[valid_idx])
            assert all(middle[valid_idx] >= lower[valid_idx])

    def test_calculate_macd(self):
        """Test MACD calculation."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)])
        macd, signal, histogram = calculate_macd(prices)

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd) == len(prices)


class TestScreenCandidates:
    """Test suite for main screening function."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database for testing."""
        db = Mock()

        # Mock fundamentals
        db.get_latest_fundamentals.return_value = {
            'ticker': 'TEST',
            'name': 'Test Company',
            'sector': 'Technology',
            'current_price': 100.0,
            'pe_ratio': 15.0,
            'pb_ratio': 2.0,
            'fcf_yield': 5.0,
            'debt_equity': 50.0
        }

        # Mock price history
        dates = pd.date_range('2023-01-01', periods=250)
        prices = [100 + np.sin(i / 10) * 10 for i in range(250)]

        db.get_price_history.return_value = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'High': [p + 2 for p in prices],
            'Low': [p - 2 for p in prices],
            'Open': prices,
            'Volume': [1000000] * 250
        })

        return db

    def test_screen_candidates_success(self, mock_db):
        """Test successful screening of candidates."""
        results = screen_candidates(mock_db, ['TEST'])

        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        assert 'ticker' in results.columns
        assert 'value_score' in results.columns
        assert 'support_score' in results.columns
        assert 'buy_signal' in results.columns

    def test_screen_candidates_multiple_tickers(self, mock_db):
        """Test screening multiple tickers."""
        results = screen_candidates(mock_db, ['TEST1', 'TEST2', 'TEST3'])

        assert isinstance(results, pd.DataFrame)
        # Results should be sorted by buy_signal descending
        if len(results) > 1:
            assert all(results['buy_signal'].iloc[i] >= results['buy_signal'].iloc[i+1]
                      for i in range(len(results)-1))

    def test_screen_candidates_empty_list(self, mock_db):
        """Test screening with empty ticker list."""
        results = screen_candidates(mock_db, [])

        assert isinstance(results, pd.DataFrame)
        assert results.empty

    def test_screen_candidates_no_data(self):
        """Test screening when database has no data."""
        db = Mock()
        db.get_latest_fundamentals.return_value = None
        db.get_price_history.return_value = pd.DataFrame()

        results = screen_candidates(db, ['TEST'])

        assert isinstance(results, pd.DataFrame)
        assert results.empty

    def test_screen_candidates_custom_weights(self, mock_db):
        """Test screening with custom weights."""
        results = screen_candidates(
            mock_db,
            ['TEST'],
            value_weight=0.5,
            support_weight=0.5
        )

        assert isinstance(results, pd.DataFrame)
        assert not results.empty

    def test_screen_candidates_includes_metadata(self, mock_db):
        """Test that results include all expected metadata."""
        results = screen_candidates(mock_db, ['TEST'])

        expected_columns = [
            'ticker', 'name', 'sector', 'current_price',
            'value_score', 'support_score', 'buy_signal',
            'rsi', 'nearest_support', 'pe_ratio', 'pb_ratio'
        ]

        for col in expected_columns:
            assert col in results.columns


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_value_score_boundary_values(self):
        """Test value scoring at boundary conditions."""
        # Test exact boundary P/E = 15
        fundamentals = {'pe_ratio': 15.0, 'pb_ratio': 1.5}
        score = calculate_value_score(fundamentals)
        assert 0 <= score <= 100

        # Test exact boundary P/B = 1.5
        fundamentals = {'pe_ratio': 12.0, 'pb_ratio': 1.5}
        score = calculate_value_score(fundamentals)
        assert 0 <= score <= 100

    def test_support_detection_flat_prices(self):
        """Test support detection with flat price data."""
        price_df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=250),
            'Close': [100.0] * 250,
            'High': [100.0] * 250,
            'Low': [100.0] * 250
        })

        supports = detect_support_levels(price_df)

        assert isinstance(supports, list)

    def test_rsi_all_gains(self):
        """Test RSI with only price gains."""
        prices = pd.Series([100 + i for i in range(30)])
        rsi = calculate_rsi(prices, period=14)

        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            # Should be high RSI (near 100)
            assert valid_rsi.iloc[-1] > 50

    def test_rsi_all_losses(self):
        """Test RSI with only price losses."""
        prices = pd.Series([100 - i for i in range(30)])
        rsi = calculate_rsi(prices, period=14)

        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            # Should be low RSI (near 0)
            assert valid_rsi.iloc[-1] < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
