"""Data fetching module for retrieving stock data from Yahoo Finance.

This module provides a robust interface for fetching stock fundamentals and
price history with caching, error handling, and retry logic.
"""

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """Fetches stock data from Yahoo Finance with caching and error handling.

    This class provides methods to fetch fundamental data and price history
    for stocks, with automatic caching to reduce API calls and retry logic
    for network failures.

    Attributes:
        cache_dir: Directory path for storing cached data files.
        cache_expiry_hours: Number of hours before cache expires (default: 24).
        max_retries: Maximum number of retry attempts for API calls (default: 3).
        retry_delay: Delay in seconds between retry attempts (default: 2).

    Example:
        >>> fetcher = YahooFinanceFetcher()
        >>> fundamentals = fetcher.fetch_fundamentals("AAPL")
        >>> prices = fetcher.fetch_price_history("AAPL", period="5y")
        >>> all_data = fetcher.fetch_multiple(["AAPL", "MSFT", "GOOGL"])
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        cache_expiry_hours: int = 24,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> None:
        """Initialize the YahooFinanceFetcher.

        Args:
            cache_dir: Directory path for caching data. Created if it doesn't exist.
            cache_expiry_hours: Hours before cached data expires.
            max_retries: Maximum retry attempts for failed API calls.
            retry_delay: Seconds to wait between retries.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"YahooFinanceFetcher initialized with cache_dir: {cache_dir}")

    def _get_cache_path(self, ticker: str, data_type: str) -> Path:
        """Get the cache file path for a given ticker and data type.

        Args:
            ticker: Stock ticker symbol.
            data_type: Type of data ('fundamentals' or 'prices').

        Returns:
            Path object for the cache file.
        """
        return self.cache_dir / f"{ticker}_{data_type}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid.

        Args:
            cache_path: Path to the cache file.

        Returns:
            True if cache exists and hasn't expired, False otherwise.
        """
        if not cache_path.exists():
            return False

        file_modified = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)

        is_valid = file_modified > expiry_time
        if is_valid:
            logger.debug(f"Cache valid for {cache_path.name}")
        else:
            logger.debug(f"Cache expired for {cache_path.name}")

        return is_valid

    def _load_from_cache(self, cache_path: Path) -> Optional[any]:
        """Load data from cache file.

        Args:
            cache_path: Path to the cache file.

        Returns:
            Cached data if successful, None otherwise.
        """
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded data from cache: {cache_path.name}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path.name}: {e}")
            return None

    def _save_to_cache(self, data: any, cache_path: Path) -> None:
        """Save data to cache file.

        Args:
            data: Data to cache.
            cache_path: Path to save the cache file.
        """
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path.name}: {e}")

    def _fetch_with_retry(self, ticker: str) -> Optional[yf.Ticker]:
        """Fetch ticker data with recursive exponential backoff.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            yfinance Ticker object if successful, None otherwise.
        """
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker)
                # DO NOT access .info here - it's a separate, slow, and heavily throttled API call.
                # yfinance will fetch price data just fine without it.
                return stock
            except Exception as e:
                # Exponential backoff: 2s, 4s, 8s...
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {ticker}: {e}. Retrying in {delay}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {ticker} after {self.max_retries} attempts")
                    return None

    def fetch_fundamentals(self, ticker: str) -> Dict[str, any]:
        """Fetch fundamental data for a stock.

        Retrieves key fundamental metrics including current price, 52-week high/low,
        P/E ratio, P/B ratio, debt-to-equity, and free cash flow.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            Dictionary containing fundamental metrics. Returns empty dict on failure.

        Example:
            >>> fetcher = YahooFinanceFetcher()
            >>> data = fetcher.fetch_fundamentals("AAPL")
            >>> print(data['pe_ratio'])
        """
        cache_path = self._get_cache_path(ticker, 'fundamentals')

        cached_data = None

        # Check cache first
        if cache_path.exists():
            cached_data = self._load_from_cache(cache_path)
            if self._is_cache_valid(cache_path) and cached_data is not None:
                return cached_data

        # Fetch from API
        logger.info(f"Fetching fundamentals for {ticker}")
        stock = self._fetch_with_retry(ticker)

        if stock is None:
            logger.error(f"Could not fetch fundamentals for {ticker}")
            return {}

        try:
            fundamentals = self._build_or_update_fundamentals(
                ticker=ticker,
                stock=stock,
                cached_data=cached_data if isinstance(cached_data, dict) else None
            )

            # Log warnings for missing data
            missing_fields = [k for k, v in fundamentals.items() if v is None and k not in ['ticker', 'name', 'sector', 'fetch_date']]
            if missing_fields:
                logger.warning(f"Missing data for {ticker}: {', '.join(missing_fields)}")

            # Cache the results
            self._save_to_cache(fundamentals, cache_path)

            logger.info(f"Successfully fetched fundamentals for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error extracting fundamentals for {ticker}: {e}")
            return {}

    def _build_or_update_fundamentals(
        self,
        ticker: str,
        stock: yf.Ticker,
        cached_data: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """Build fundamentals snapshot, updating cached values incrementally when possible."""
        fundamentals = dict(cached_data or {})

        # Start with fast_info because it's cheaper than a full .info refresh.
        try:
            fast_info = stock.fast_info
        except Exception as e:
            logger.debug(f"{ticker}: fast_info unavailable: {e}")
            fast_info = {}

        price_now = fast_info.get('last_price') or fast_info.get('regular_market_price')
        if price_now is not None:
            fundamentals['current_price'] = price_now

        week_52_high = fast_info.get('year_high')
        week_52_low = fast_info.get('year_low')
        if week_52_high is not None:
            fundamentals['week_52_high'] = week_52_high
        if week_52_low is not None:
            fundamentals['week_52_low'] = week_52_low

        market_cap = fast_info.get('market_cap')
        if market_cap is not None:
            fundamentals['market_cap'] = market_cap

        # Pull full info only when needed to fill missing static fundamentals.
        required_fields = [
            'name', 'sector', 'pe_ratio', 'pb_ratio', 'debt_to_equity',
            'free_cash_flow', 'trailing_eps', 'forward_eps', 'dividend_yield'
        ]
        missing_required = [field for field in required_fields if fundamentals.get(field) is None]

        if missing_required:
            info = stock.info
            fundamentals.update({
                'name': fundamentals.get('name') or info.get('longName', ticker),
                'sector': fundamentals.get('sector') or info.get('sector', 'Unknown'),
                'current_price': fundamentals.get('current_price') or info.get('currentPrice') or info.get('regularMarketPrice'),
                'week_52_high': fundamentals.get('week_52_high') or info.get('fiftyTwoWeekHigh'),
                'week_52_low': fundamentals.get('week_52_low') or info.get('fiftyTwoWeekLow'),
                'pe_ratio': fundamentals.get('pe_ratio') or info.get('trailingPE') or info.get('forwardPE'),
                'pb_ratio': fundamentals.get('pb_ratio') or info.get('priceToBook'),
                'debt_to_equity': fundamentals.get('debt_to_equity') or info.get('debtToEquity'),
                'free_cash_flow': fundamentals.get('free_cash_flow') or info.get('freeCashflow'),
                'market_cap': fundamentals.get('market_cap') or info.get('marketCap'),
                'trailing_eps': fundamentals.get('trailing_eps') or info.get('trailingEps'),
                'forward_eps': fundamentals.get('forward_eps') or info.get('forwardEps'),
                'dividend_yield': fundamentals.get('dividend_yield') or info.get('dividendYield')
            })

        fundamentals['ticker'] = ticker
        fundamentals['fetch_date'] = datetime.now().isoformat()
        fundamentals.setdefault('name', ticker)
        fundamentals.setdefault('sector', 'Unknown')

        return fundamentals

    def fetch_price_history(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical price data for a stock.

        Retrieves OHLCV (Open, High, Low, Close, Volume) data for the specified period.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            period: Time period for historical data. Valid values: 1d, 5d, 1mo, 3mo,
                   6mo, 1y, 2y, 5y, 10y, ytd, max. Default is '5y'.
            interval: Data interval. Valid values: 1m, 2m, 5m, 15m, 30m, 60m, 90m,
                     1h, 1d, 5d, 1wk, 1mo, 3mo. Default is '1d'.

        Returns:
            DataFrame with columns: Date (index), Open, High, Low, Close, Volume.
            Returns empty DataFrame on failure.

        Example:
            >>> fetcher = YahooFinanceFetcher()
            >>> prices = fetcher.fetch_price_history("AAPL", period="1y")
            >>> print(prices.head())
        """
        cache_path = self._get_cache_path(ticker, f'prices_{period}_{interval}')

        cached_data = None

        # Check cache first
        if cache_path.exists():
            cached_data = self._load_from_cache(cache_path)
            if self._is_cache_valid(cache_path) and cached_data is not None and isinstance(cached_data, pd.DataFrame):
                return cached_data

            if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                refreshed = self._refresh_price_history_incremental(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    cached_data=cached_data
                )
                if not refreshed.empty:
                    self._save_to_cache(refreshed, cache_path)
                    logger.info(f"Updated cached price history for {ticker} with incremental yfinance data")
                    return refreshed

        # Fetch from API
        logger.info(f"Fetching price history for {ticker} (period={period}, interval={interval})")
        stock = self._fetch_with_retry(ticker)

        if stock is None:
            logger.error(f"Could not fetch price history for {ticker}")
            return pd.DataFrame()

        try:
            # Fetch historical data
            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No price history data available for {ticker}")
                return pd.DataFrame()

            # Clean up the DataFrame - keep DatetimeIndex for consistency with git_fetcher
            # DO NOT reset_index() - we want to preserve the DatetimeIndex from yfinance
            hist.columns = [col.capitalize() for col in hist.columns]

            # Ensure index is DatetimeIndex (yfinance should provide this)
            if not isinstance(hist.index, pd.DatetimeIndex):
                logger.warning(f"{ticker}: yfinance returned non-DatetimeIndex: {type(hist.index)}")
                return pd.DataFrame()

            # Select only OHLCV columns (no 'Date' column - it's the index)
            available_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            hist = hist[[col for col in available_cols if col in hist.columns]]

            # Cache the results
            self._save_to_cache(hist, cache_path)

            logger.info(f"Successfully fetched {len(hist)} price records for {ticker}")
            return hist

        except Exception as e:
            logger.error(f"Error fetching price history for {ticker}: {e}")
            return pd.DataFrame()

    def _refresh_price_history_incremental(
        self,
        ticker: str,
        period: str,
        interval: str,
        cached_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Update cached price history by requesting only the missing tail window."""
        if cached_data.empty or not isinstance(cached_data.index, pd.DatetimeIndex):
            return pd.DataFrame()

        try:
            last_cached_dt = pd.Timestamp(cached_data.index.max())
            start_dt = (last_cached_dt - timedelta(days=7)).date().isoformat()

            logger.info(
                f"Incremental price refresh for {ticker} from {start_dt} (period={period}, interval={interval})"
            )

            stock = self._fetch_with_retry(ticker)
            if stock is None:
                return pd.DataFrame()

            incremental = stock.history(start=start_dt, interval=interval)
            if incremental.empty:
                return cached_data

            incremental.columns = [col.capitalize() for col in incremental.columns]
            available_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            incremental = incremental[[col for col in available_cols if col in incremental.columns]]

            merged = pd.concat([cached_data, incremental])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.sort_index()

            # Trim oversized caches to requested period where supported.
            period_to_days = {
                '1mo': 31, '3mo': 93, '6mo': 186, '1y': 366, '2y': 731,
                '5y': 1827, '10y': 3653
            }
            if period in period_to_days:
                cutoff = pd.Timestamp.now(tz=merged.index.tz) - pd.Timedelta(days=period_to_days[period])
                merged = merged[merged.index >= cutoff]

            return merged
        except Exception as e:
            logger.warning(f"Failed incremental price refresh for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: List[str],
        period: str = "5y"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch fundamentals and price history for multiple stocks.

        Args:
            tickers: List of stock ticker symbols.
            period: Time period for historical data (default: '5y').

        Returns:
            Tuple of (fundamentals_df, prices_df):
                - fundamentals_df: DataFrame with one row per ticker
                - prices_df: DataFrame with all price data, includes 'ticker' column

        Example:
            >>> fetcher = YahooFinanceFetcher()
            >>> fundamentals, prices = fetcher.fetch_multiple(["AAPL", "MSFT", "GOOGL"])
            >>> print(fundamentals[['ticker', 'pe_ratio', 'pb_ratio']])
        """
        logger.info(f"Fetching data for {len(tickers)} tickers")

        all_fundamentals = []
        all_prices = []

        for ticker in tickers:
            # Fetch fundamentals
            fundamentals = self.fetch_fundamentals(ticker)
            if fundamentals:
                all_fundamentals.append(fundamentals)

            # Fetch price history
            prices = self.fetch_price_history(ticker, period=period)
            if not prices.empty:
                prices['ticker'] = ticker
                all_prices.append(prices)

        # Combine all data
        fundamentals_df = pd.DataFrame(all_fundamentals) if all_fundamentals else pd.DataFrame()
        prices_df = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()

        logger.info(
            f"Fetched {len(all_fundamentals)}/{len(tickers)} fundamentals, "
            f"{len(all_prices)}/{len(tickers)} price histories"
        )

        return fundamentals_df, prices_df

    def fetch_batch_prices(
        self,
        tickers: List[str],
        period: str = "5y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price history for multiple tickers in a single batch request.
        
        This is MUCH more efficient and less likely to trigger rate limits than
        fetching tickers individually.

        Args:
            tickers: List of stock ticker symbols.
            period: Time period (default: '5y').
            interval: Data interval (default: '1d').

        Returns:
            Dictionary mapping ticker to its price DataFrame.
        """
        if not tickers:
            return {}

        logger.info(f"Batch fetching prices for {len(tickers)} tickers (period={period})")
        
        # Filter out tickers already in cache for this specific period/interval
        cache_ready = {}
        to_fetch = []
        
        for ticker in tickers:
            cache_path = self._get_cache_path(ticker, f'prices_{period}_{interval}')
            if self._is_cache_valid(cache_path):
                cached = self._load_from_cache(cache_path)
                if cached is not None and isinstance(cached, pd.DataFrame):
                    cache_ready[ticker] = cached
                else:
                    to_fetch.append(ticker)
            else:
                to_fetch.append(ticker)

        if not to_fetch:
            logger.info(f"All {len(tickers)} tickers found in cache.")
            return cache_ready

        results = cache_ready.copy()
        
        # Process in chunks of 50 to avoid URL length limits and very large responses
        chunk_size = 50
        for i in range(0, len(to_fetch), chunk_size):
            chunk = to_fetch[i:i + chunk_size]
            logger.info(f"Requesting batch chunk {i//chunk_size + 1}: {len(chunk)} tickers")
            
            try:
                # yf.download is the optimized batch method
                data = yf.download(
                    tickers=" ".join(chunk),
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    threads=True,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"Batch request for chunk {i} returned no data")
                    continue

                for ticker in chunk:
                    try:
                        if len(chunk) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker]
                        
                        if ticker_data.empty:
                            continue
                            
                        # Clean up
                        ticker_data.columns = [col.capitalize() for col in ticker_data.columns]
                        available_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        ticker_data = ticker_data[[col for col in available_cols if col in ticker_data.columns]]
                        ticker_data = ticker_data.dropna(subset=['Close'])
                        
                        if not ticker_data.empty:
                            results[ticker] = ticker_data
                            # Save individual cache for future use
                            cache_path = self._get_cache_path(ticker, f'prices_{period}_{interval}')
                            self._save_to_cache(ticker_data, cache_path)
                            
                    except Exception as e:
                        logger.debug(f"Failed to extract {ticker} from batch: {e}")
                        
                # Small courtesy delay between chunks
                if i + chunk_size < len(to_fetch):
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Batch fetch failed for chunk starting with {chunk[0]}: {e}")
                # Fallback to individual fetches for this chunk if batch fails
                for ticker in chunk:
                    res = self.fetch_price_history(ticker, period=period, interval=interval)
                    if not res.empty:
                        results[ticker] = res

        return results

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cached data.

        Args:
            ticker: If provided, clears cache only for this ticker.
                   If None, clears all cached data.

        Example:
            >>> fetcher = YahooFinanceFetcher()
            >>> fetcher.clear_cache("AAPL")  # Clear only AAPL cache
            >>> fetcher.clear_cache()  # Clear all cache
        """
        if ticker:
            # Clear cache for specific ticker
            pattern = f"{ticker}_*.pkl"
            removed = 0
            for cache_file in self.cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
            logger.info(f"Cleared {removed} cache file(s) for {ticker}")
        else:
            # Clear all cache
            removed = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
            logger.info(f"Cleared {removed} cache file(s)")
