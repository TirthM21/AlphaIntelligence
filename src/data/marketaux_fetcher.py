"""Marketaux News & Sentiment API Fetcher."""

import logging
import os
import requests
import json
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class MarketauxFetcher:
    """Fetcher for Marketaux API to get sentiment and global financial news."""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('MARKETAUX_API_KEY') or "ElkIqFLhx7BrOtLorkT7uDD2jo0AgqHVZuxaFyQ6"
        self.base_url = "https://api.marketaux.com/v1"
        self.cache_dir = Path("./data/cache/marketaux")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            logger.warning("MARKETAUX_API_KEY not set. Features will be limited.")

    def _get_cache_path(self, key: str) -> Path:
        """Helper to get a clean cache filename."""
        safe_key = "".join([c if c.isalnum() else "_" for c in key])
        return self.cache_dir / f"{safe_key}.pkl"

    def _is_cache_valid(self, cache_path: Path, hours: int = 1) -> bool:
        """Check if cache file exists and is recent."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=hours)

    def fetch_trending_entities(self, countries: str = "us", minutes: int = 1440) -> List[Dict]:
        """Fetch trending entities (stocks/indices) from Marketaux."""
        if not self.api_key:
            return []
            
        cache_key = f"trending_{countries}_{minutes}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, hours=2):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        params = {
            'api_token': self.api_key,
            'countries': countries,
            'language': 'en',
            'limit': 10
        }
        
        try:
            response = requests.get(f"{self.base_url}/entity/trending/aggregation", params=params, timeout=15)
            if response.status_code == 403:
                 logger.info("Marketaux trending endpoint restricted. Falling back to news aggregation...")
                 return self._fetch_trending_fallback(countries)
                 
            response.raise_for_status()
            data = response.json()
            entities = data.get('data', [])
            
            with open(cache_path, 'wb') as f:
                pickle.dump(entities, f)
            return entities
        except Exception as e:
            logger.error(f"Marketaux trending fetch failed: {e}")
            return self._fetch_trending_fallback(countries)

    def _fetch_trending_fallback(self, countries: str = "us") -> List[Dict]:
        """Fallback: Extract trending entities from recent news items."""
        params = {
            'api_token': self.api_key,
            'countries': countries,
            'language': 'en',
            'limit': 10,
            'must_have_entities': 'true'
        }
        try:
            response = requests.get(f"{self.base_url}/news/all", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get('data', [])
            
            entity_map = {}
            for art in articles:
                for ent in art.get('entities', []):
                    sym = ent.get('symbol')
                    if not sym: continue
                    if sym not in entity_map:
                        entity_map[sym] = {
                            'key': sym,
                            'sentiment_avg': ent.get('sentiment_score', 0),
                            'total_documents': 1,
                            'score': ent.get('match_score', 0)
                        }
                    else:
                        entity_map[sym]['total_documents'] += 1
                        entity_map[sym]['sentiment_avg'] = (entity_map[sym]['sentiment_avg'] + ent.get('sentiment_score', 0)) / 2
            
            # Sort by document count
            sorted_entities = sorted(entity_map.values(), key=lambda x: x['total_documents'], reverse=True)
            return sorted_entities[:10]
        except Exception as e:
            logger.error(f"Marketaux trending fallback failed: {e}")
            return []

    def fetch_stock_news(self, symbols: List[str], limit: int = 5) -> List[Dict]:
        """Fetch news and sentiment for specific symbols."""
        if not self.api_key or not symbols:
            return []
            
        sym_str = ",".join(symbols)
        cache_key = f"news_{sym_str}_{limit}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, hours=1):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        params = {
            'api_token': self.api_key,
            'symbols': sym_str,
            'language': 'en',
            'limit': limit,
            'filter_entities': 'true'
        }
        
        try:
            response = requests.get(f"{self.base_url}/news/all", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get('data', [])
            
            with open(cache_path, 'wb') as f:
                pickle.dump(articles, f)
            return articles
        except Exception as e:
            logger.error(f"Marketaux news fetch failed: {e}")
            return []

    def fetch_market_news(self, sentiment_gte: float = 0.1, limit: int = 5) -> List[Dict]:
        """Fetch general positive market news."""
        if not self.api_key:
            return []
            
        params = {
            'api_token': self.api_key,
            'language': 'en',
            'limit': limit,
            'sentiment_gte': sentiment_gte,
            'must_have_entities': 'true'
        }
        
        try:
            response = requests.get(f"{self.base_url}/news/all", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            logger.error(f"Marketaux macro news fetch failed: {e}")
            return []
