"""Finnhub API fetcher for high-quality market news and data.

Finnhub provides:
- Real-time market news
- Company news
- Sentiment analysis
- Earnings calendars
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pickle

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinnhubFetcher:
    """Fetch high-quality news and market data from Finnhub."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./data/cache"):
        """Initialize Finnhub fetcher."""
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            logger.warning("No FINNHUB_API_KEY found in .env!")
        
        self.base_url = "https://finnhub.io/api/v1"
        self.cache_dir = Path(cache_dir) / "finnhub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _is_cache_valid(self, cache_path: Path, hours: int = 1) -> bool:
        if not cache_path.exists():
            return False
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - file_time < timedelta(hours=hours)

    def fetch_market_news(self, category: str = "general", min_id: int = 0) -> List[Dict]:
        """Fetch general market news.
        
        Categories: general, forex, crypto, merger
        """
        cache_key = f"market_news_{category}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, hours=0.5): # 30 min cache for fresh news
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/news"
            params = {
                'category': category,
                'minId': min_id,
                'token': self.api_key
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching Finnhub market news: {e}")
            return []

    def fetch_company_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Fetch news for a specific company."""
        cache_key = f"company_news_{ticker}"
        cache_path = self._get_cache_path(cache_key)
        
        # 4 hour cache for company news
        if self._is_cache_valid(cache_path, hours=4):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if not self.api_key:
            return []

        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': ticker,
                'from': start_date,
                'to': end_date,
                'token': self.api_key
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {ticker}: {e}")
            return []

    def fetch_basic_financials(self, ticker: str) -> Dict:
        """Fetch basic financial metrics (margins, PE, etc)."""
        if not self.api_key:
            return {}
            
        try:
            url = f"{self.base_url}/stock/metric"
            params = {
                'symbol': ticker,
                'metric': 'all',
                'token': self.api_key
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching Finnhub metrics for {ticker}: {e}")
            return {}

    def fetch_earnings_calendar(self, days_forward: int = 7) -> List[Dict]:
        """Fetch coming earnings releases from Finnhub."""
        cache_key = f"earnings_cal_{days_forward}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, hours=12):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if not self.api_key:
            return []

        try:
            start_date = datetime.now().strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=days_forward)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/calendar/earnings"
            params = {
                'from': start_date,
                'to': end_date,
                'token': self.api_key
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get('earningsCalendar', [])
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching Finnhub earnings calendar: {e}")
            return []

    def fetch_economic_calendar(self) -> List[Dict]:
        """Fetch economic events from Finnhub (Premium Endpoint)."""
        # This is often premium, but we include it for completeness if the key allows
        cache_key = "econ_cal"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, hours=12):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/calendar/economic"
            params = {'token': self.api_key}
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 403:
                return []
            response.raise_for_status()
            data = response.json().get('economicCalendar', [])
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            return data
        except Exception as e:
            # Silent fail for premium-only
            return []
