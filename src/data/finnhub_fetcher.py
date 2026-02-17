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

    def _fetch_quote(self, symbol: str) -> Dict:
        """Fetch quote for a symbol from Finnhub."""
        if not self.api_key:
            return {}
        try:
            response = self.session.get(
                f"{self.base_url}/quote",
                params={"symbol": symbol, "token": self.api_key},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json() or {}
            if not data or float(data.get("c", 0) or 0) <= 0:
                return {}
            return data
        except Exception as exc:
            logger.error(f"Error fetching Finnhub quote for {symbol}: {exc}")
            return {}

    def fetch_major_index_snapshot(self) -> Dict[str, Dict]:
        """Fetch major index proxies and normalize into a compact snapshot."""
        cache_key = "major_index_snapshot"
        cache_path = self._get_cache_path(cache_key)
        if self._is_cache_valid(cache_path, hours=0.08):  # ~5 minutes
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        proxies = {
            "SPY": "S&P 500",
            "QQQ": "Nasdaq 100",
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
        }
        snapshot: Dict[str, Dict] = {}
        for symbol, label in proxies.items():
            quote = self._fetch_quote(symbol)
            if not quote:
                continue
            current = float(quote.get("c", 0) or 0)
            previous_close = float(quote.get("pc", 0) or 0)
            if current <= 0 or previous_close <= 0:
                continue
            pct_change = ((current / previous_close) - 1) * 100
            snapshot[symbol] = {
                "label": label,
                "symbol": symbol,
                "current": round(current, 2),
                "change": round(current - previous_close, 2),
                "change_pct": round(pct_change, 2),
                "timestamp": quote.get("t"),
            }

        if snapshot:
            with open(cache_path, "wb") as f:
                pickle.dump(snapshot, f)
        return snapshot

    def fetch_market_sentiment_proxy(self) -> Dict:
        """Aggregate a proxy sentiment score from broad ETF tape and market breadth proxies."""
        cache_key = "market_sentiment_proxy"
        cache_path = self._get_cache_path(cache_key)
        if self._is_cache_valid(cache_path, hours=0.25):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        snapshot = self.fetch_major_index_snapshot()
        if not snapshot:
            return {"score": 50.0, "label": "Neutral", "components": []}

        spy_move = snapshot.get("SPY", {}).get("change_pct", 0.0)
        qqq_move = snapshot.get("QQQ", {}).get("change_pct", 0.0)
        iwm_move = snapshot.get("IWM", {}).get("change_pct", 0.0)
        dia_move = snapshot.get("DIA", {}).get("change_pct", 0.0)

        avg_move = (spy_move + qqq_move + iwm_move + dia_move) / 4
        breadth_signal = iwm_move - spy_move
        growth_signal = qqq_move - dia_move

        score = 50.0
        score += max(-20, min(20, avg_move * 8))
        score += max(-10, min(10, breadth_signal * 6))
        score += max(-10, min(10, growth_signal * 4))
        score = max(0.0, min(100.0, score))

        if score >= 75:
            label = "Extreme Greed"
        elif score >= 60:
            label = "Greed"
        elif score <= 25:
            label = "Extreme Fear"
        elif score <= 40:
            label = "Fear"
        else:
            label = "Neutral"

        payload = {
            "score": round(score, 1),
            "label": label,
            "components": [
                {"name": "Average Index Move", "value": round(avg_move, 2)},
                {"name": "Small vs Large Cap", "value": round(breadth_signal, 2)},
                {"name": "Growth vs Value", "value": round(growth_signal, 2)},
            ],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
        return payload

    def fetch_notable_movers(self, symbols: Optional[List[str]] = None, limit: int = 6) -> List[Dict]:
        """Build a notable movers feed (top gainers/losers proxy) from tracked symbols."""
        cache_key = f"notable_movers_{limit}"
        cache_path = self._get_cache_path(cache_key)
        if self._is_cache_valid(cache_path, hours=0.17):  # ~10 minutes
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        universe = symbols or [
            "SPY", "QQQ", "DIA", "IWM", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP",
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "JPM", "UNH", "XOM",
        ]

        movers: List[Dict] = []
        for symbol in universe:
            quote = self._fetch_quote(symbol)
            if not quote:
                continue
            current = float(quote.get("c", 0) or 0)
            previous_close = float(quote.get("pc", 0) or 0)
            if current <= 0 or previous_close <= 0:
                continue
            pct_change = ((current / previous_close) - 1) * 100
            reason = "Momentum extension"
            if abs(pct_change) >= 3:
                reason = "High-volatility move"
            elif symbol in {"XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI"}:
                reason = "Sector rotation signal"

            movers.append(
                {
                    "symbol": symbol,
                    "price": round(current, 2),
                    "change_pct": round(pct_change, 2),
                    "direction": "gainer" if pct_change >= 0 else "loser",
                    "reason": reason,
                }
            )

        gainers = sorted([m for m in movers if m["change_pct"] >= 0], key=lambda x: x["change_pct"], reverse=True)
        losers = sorted([m for m in movers if m["change_pct"] < 0], key=lambda x: x["change_pct"])
        feed = (gainers[: max(1, limit // 2)] + losers[: max(1, limit - (limit // 2))])[:limit]

        if feed:
            with open(cache_path, "wb") as f:
                pickle.dump(feed, f)
        return feed

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
