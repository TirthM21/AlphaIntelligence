"""Fetch and maintain the universe of all publicly traded US stocks.

This module fetches the complete list of US-listed stocks from multiple sources
and maintains a daily-updated universe for screening.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import requests

from .fmp_fetcher import FMPFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USStockUniverseFetcher:
    """Fetches and maintains the universe of all US-listed stocks."""

    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize the universe fetcher.

        Args:
            cache_dir: Directory for caching universe data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "us_stock_universe.pkl"
        logger.info("USStockUniverseFetcher initialized")

    def _fetch_from_fmp(self) -> pd.DataFrame:
        """Fetch stock list from Financial Modeling Prep.
        
        Uses FMP's stock list endpoint which returns all US-listed stocks
        with symbol, name, price, and exchange info.
        
        Returns:
            DataFrame with symbols and names
        """
        try:
            fmp = FMPFetcher()
            if not fmp.api_key:
                logger.info("FMP API key not set, skipping FMP universe fetch")
                return pd.DataFrame()
            
            stocks = fmp.fetch_stock_list()
            if not stocks:
                logger.warning("FMP returned empty stock list")
                return pd.DataFrame()
            
            df = pd.DataFrame(stocks)
            if 'symbol' not in df.columns:
                return pd.DataFrame()
            
            # Rename to match expected format
            df = df[['symbol', 'name']].copy()
            
            # Filter out penny stocks (price < $1) if price data available
            if 'price' in pd.DataFrame(stocks).columns:
                prices = pd.DataFrame(stocks)['price']
                mask = prices.fillna(0) >= 1.0
                df = df[mask.values]
            
            logger.info(f"FMP universe: {len(df)} US stocks fetched")
            return df
        except Exception as e:
            logger.error(f"Error fetching FMP stock list: {e}")
            return pd.DataFrame()

    def _fetch_nasdaq_listed(self) -> pd.DataFrame:
        """Fetch NASDAQ-listed stocks from NASDAQ FTP.

        Returns:
            DataFrame with NASDAQ stocks
        """
        try:
            url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
            df = pd.read_csv(url, sep='|')
            df = df[df['Symbol'].notna()]
            df = df[df['Test Issue'] == 'N']  # Exclude test issues
            df = df[['Symbol', 'Security Name']].copy()
            df.columns = ['symbol', 'name']
            logger.info(f"Fetched {len(df)} NASDAQ stocks")
            return df
        except Exception as e:
            logger.error(f"Error fetching NASDAQ stocks: {e}")
            return pd.DataFrame()

    def _fetch_other_listed(self) -> pd.DataFrame:
        """Fetch non-NASDAQ listed stocks (NYSE, AMEX, etc).

        Returns:
            DataFrame with other exchange stocks
        """
        try:
            url = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"
            df = pd.read_csv(url, sep='|')
            df = df[df['ACT Symbol'].notna()]
            df = df[df['Test Issue'] == 'N']  # Exclude test issues
            df = df[['ACT Symbol', 'Security Name']].copy()
            df.columns = ['symbol', 'name']
            logger.info(f"Fetched {len(df)} NYSE/AMEX stocks")
            return df
        except Exception as e:
            logger.error(f"Error fetching NYSE/AMEX stocks: {e}")
            return pd.DataFrame()

    def _filter_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out unwanted tickers.

        Removes:
        - Tickers with special characters ($, ^, ., etc.)
        - Test symbols
        - Warrants, rights, units
        - Preferred shares
        - ETFs and funds (heuristic)

        Args:
            df: DataFrame with symbols

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        initial_count = len(df)

        # Remove symbols with special characters
        df = df[~df['symbol'].str.contains(r'[\$\^\.\-]', regex=True, na=False)]

        # Remove common suffixes for warrants, rights, units
        suffixes = ['W', 'R', 'U', 'WS', 'WT']
        for suffix in suffixes:
            df = df[~df['symbol'].str.endswith(suffix, na=False)]

        # Remove preferred shares (usually have letters after symbol)
        # Keep only symbols that are 1-5 uppercase letters
        df = df[df['symbol'].str.match(r'^[A-Z]{1,5}$', na=False)]

        # Remove obvious ETFs and funds (heuristic based on name)
        etf_keywords = [
            'ETF', 'FUND', 'TRUST', 'INDEX', 'PORTFOLIO',
            'SHARES', 'NOTES', 'BOND', 'TREASURY'
        ]
        name_upper = df['name'].str.upper()
        for keyword in etf_keywords:
            df = df[~name_upper.str.contains(keyword, na=False)]

        filtered_count = len(df)
        logger.info(f"Filtered {initial_count - filtered_count} stocks, kept {filtered_count}")

        return df

    def fetch_universe(self, force_refresh: bool = False) -> List[str]:
        """Fetch the complete universe of US-listed stocks.

        Args:
            force_refresh: Force refresh even if cached data is recent

        Returns:
            List of stock ticker symbols
        """
        # Check cache
        if not force_refresh and self.cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(
                self.cache_file.stat().st_mtime
            )

            if cache_age < timedelta(days=1):
                logger.info("Loading universe from cache")
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded {len(cached_data['symbols'])} symbols from cache")
                return cached_data['symbols']

        logger.info("Fetching fresh universe from exchanges...")

        nasdaq_count = 0
        other_count = 0
        fmp_count = 0
        source = "nasdaq_ftp"

        # Try FMP first (faster, more reliable, includes metadata)
        fmp_df = self._fetch_from_fmp()
        
        if not fmp_df.empty and len(fmp_df) > 1000:
            logger.info(f"Using FMP as primary universe source ({len(fmp_df)} stocks)")
            all_stocks = fmp_df
            source = "fmp"
            fmp_count = len(fmp_df)
        else:
            # Fallback to NASDAQ FTP
            logger.info("FMP unavailable or insufficient, falling back to NASDAQ FTP...")
            nasdaq_df = self._fetch_nasdaq_listed()
            other_df = self._fetch_other_listed()
            nasdaq_count = len(nasdaq_df)
            other_count = len(other_df)
            all_stocks = pd.concat([nasdaq_df, other_df], ignore_index=True)

        if all_stocks.empty:
            logger.error("Failed to fetch any stocks")
            return []

        # Remove duplicates
        all_stocks = all_stocks.drop_duplicates(subset=['symbol'])

        # Filter unwanted symbols
        all_stocks = self._filter_stocks(all_stocks)

        # Sort by symbol
        all_stocks = all_stocks.sort_values('symbol').reset_index(drop=True)

        symbols = all_stocks['symbol'].tolist()

        # Cache the results
        cache_data = {
            'symbols': symbols,
            'fetch_date': datetime.now().isoformat(),
            'count': len(symbols),
            'metadata': {
                'source': source,
                'fmp_count': fmp_count,
                'nasdaq_count': nasdaq_count,
                'other_count': other_count,
                'filtered_count': len(symbols)
            }
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"Cached {len(symbols)} symbols")
        logger.info(f"Universe composition: {cache_data['metadata']}")

        return symbols

    def get_universe_info(self) -> Dict:
        """Get information about the cached universe.

        Returns:
            Dict with universe metadata
        """
        if not self.cache_file.exists():
            return {
                'cached': False,
                'count': 0
            }

        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        cache_age = datetime.now() - datetime.fromtimestamp(
            self.cache_file.stat().st_mtime
        )

        return {
            'cached': True,
            'count': cached_data['count'],
            'fetch_date': cached_data['fetch_date'],
            'cache_age_hours': cache_age.total_seconds() / 3600,
            'metadata': cached_data.get('metadata', {})
        }
