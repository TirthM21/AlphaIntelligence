"""Finnhub API fetcher for market data, crypto, forex, and news.

Implements rate limiting and caching to respect Finnhub API constraints.
"""

import logging
import os
import time
import requests
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class FinnhubFetcher:
    """Fetcher for Finnhub API with rate limiting and caching."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./data/cache/finnhub"):
        """Initialize Finnhub fetcher.

        Args:
            api_key: Finnhub API key (or set FINNHUB_API_KEY env variable)
            cache_dir: Directory for caching responses
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        self.base_url = "https://finnhub.io/api/v1"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # User requested 5 seconds rate limit
        self.last_call_time = 0
        self.min_delay = 5.0 

        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not set. Some features may be disabled.")

    def _wait_for_rate_limit(self):
        """Ensure minimum delay between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            wait_time = self.min_delay - elapsed
            logger.debug(f"Finnhub Rate Limit: Sleeping {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_call_time = time.time()

    def _fetch(self, endpoint: str, params: Dict = None, cache_hours: int = 24) -> Optional[Union[Dict, List]]:
        """Fetch data from Finnhub with caching and rate limiting."""
        if not self.api_key:
            return None

        # Sort params for stable cache key
        import hashlib
        import json
        param_dict = params.copy() if params else {}
        if 'token' in param_dict: del param_dict['token']
        
        stable_params = json.dumps(param_dict, sort_keys=True)
        param_hash = hashlib.md5(stable_params.encode()).hexdigest()[:10]
        cache_key = f"{endpoint.replace('/', '_')}_{param_hash}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        # Check cache
        if cache_path.exists():
            file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - file_time < timedelta(hours=cache_hours):
                logger.debug(f"Finnhub Cache Hit: {endpoint}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        # Rate limiting
        self._wait_for_rate_limit()

        # Call API
        try:
            url = f"{self.base_url}/{endpoint}"
            request_params = params.copy() if params else {}
            request_params['token'] = self.api_key
            
            logger.info(f"Finnhub Fetching: {endpoint} {param_dict}")
            response = requests.get(url, params=request_params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            return data
        except Exception as e:
            logger.error(f"Finnhub API Error ({endpoint}): {e}")
            return None

    # 1. Market Data - Stocks
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote."""
        return self._fetch("quote", {"symbol": symbol}, cache_hours=1)

    def get_candles(self, symbol: str, resolution: str = "D", start: int = None, end: int = None) -> Optional[Dict]:
        """Get historical OHLCV."""
        if not start:
            start = int((datetime.now() - timedelta(days=365)).timestamp())
        if not end:
            end = int(datetime.now().timestamp())
            
        return self._fetch("stock/candle", {
            "symbol": symbol,
            "resolution": resolution,
            "from": start,
            "to": end
        })

    def get_symbols(self, exchange: str = "US") -> Optional[List[Dict]]:
        """Get list of symbols."""
        return self._fetch("stock/symbol", {"exchange": exchange}, cache_hours=168)

    def get_peers(self, symbol: str) -> Optional[List[str]]:
        """Get industry peers."""
        return self._fetch("stock/peers", {"symbol": symbol}, cache_hours=24)

    def get_earnings(self, symbol: str) -> Optional[List[Dict]]:
        """Get earnings history."""
        return self._fetch("stock/earnings", {"symbol": symbol}, cache_hours=24)

    def get_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile."""
        return self._fetch("stock/profile2", {"symbol": symbol}, cache_hours=168)

    def get_metrics(self, symbol: str, metric: str = "all") -> Optional[Dict]:
        """Get financial metrics."""
        return self._fetch("stock/metric", {"symbol": symbol, "metric": metric}, cache_hours=24)

    # 2. Crypto
    def get_crypto_candles(self, symbol: str, resolution: str = "D", start: int = None, end: int = None) -> Optional[Dict]:
        """Get historical crypto OHLCV."""
        if not start:
            start = int((datetime.now() - timedelta(days=30)).timestamp())
        if not end:
            end = int(datetime.now().timestamp())
            
        return self._fetch("crypto/candle", {
            "symbol": symbol,
            "resolution": resolution,
            "from": start,
            "to": end
        })

    # 3. Forex
    def get_forex_rates(self, base: str = "USD") -> Optional[Dict]:
        """Get FX rates."""
        return self._fetch("forex/rates", {"base": base}, cache_hours=1)

    # 4. News & Sentiment
    def get_company_news(self, symbol: str, start: str = None, end: str = None) -> Optional[List[Dict]]:
        """Get company news."""
        if not start:
            start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')
            
        return self._fetch("company-news", {
            "symbol": symbol,
            "from": start,
            "to": end
        }, cache_hours=1)

    def get_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get news sentiment."""
        return self._fetch("news-sentiment", {"symbol": symbol}, cache_hours=24)

    # 5. Calendars
    def get_earnings_calendar(self, start: str = None, end: str = None) -> Optional[List[Dict]]:
        """Get earnings calendar."""
        if not start:
            start = datetime.now().strftime('%Y-%m-%d')
        if not end:
            end = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            
        data = self._fetch("calendar/earnings", {"from": start, "to": end}, cache_hours=6)
        return data.get('earningsCalendar', []) if data else []

    # 6. Filings
    def get_filings(self, symbol: str) -> Optional[List[Dict]]:
        """Get SEC filings."""
        return self._fetch("stock/filings", {"symbol": symbol}, cache_hours=24)

    # 7. ETFs
    def get_etf_profile(self, symbol: str) -> Optional[Dict]:
        """Get ETF profile."""
        return self._fetch("etf/profile", {"symbol": symbol}, cache_hours=168)

    # 8. Insider Trading
    def get_insider_transactions(self, symbol: str) -> Optional[List[Dict]]:
        """Get company insider transactions."""
        data = self._fetch("stock/insider-transactions", {"symbol": symbol}, cache_hours=24)
        return data.get('data', []) if data else []

    def get_insider_sentiment(self, symbol: str, start: str = None, end: str = None) -> Optional[List[Dict]]:
        """Get insider sentiment (MSPR)."""
        if not start:
            start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')
            
        data = self._fetch("stock/insider-sentiment", {
            "symbol": symbol,
            "from": start,
            "to": end
        }, cache_hours=168)
        return data.get('data', []) if data else []

    # 9. Analyst Sentiment & Targets
    def get_recommendation_trends(self, symbol: str) -> Optional[List[Dict]]:
        """Get latest analyst recommendation trends."""
        return self._fetch("stock/recommendation", {"symbol": symbol}, cache_hours=24)

    def get_price_target(self, symbol: str) -> Optional[Dict]:
        """Get analyst price targets."""
        return self._fetch("stock/price-target", {"symbol": symbol}, cache_hours=24)

    # 10. Financial Statements (Standardized)
    def get_financials(self, symbol: str, statement: str = "bs", freq: str = "annual") -> Optional[List[Dict]]:
        """Get standardized financials (bs, ic, cf)."""
        data = self._fetch("stock/financials", {
            "symbol": symbol,
            "statement": statement,
            "freq": freq
        }, cache_hours=168)
        return data.get('financials', []) if data else []
