"""Financial Modeling Prep (FMP) API fetcher for detailed quarterly fundamentals.

FMP provides comprehensive financial data including:
- Quarterly income statements (net margins, operating margins)
- Balance sheets (inventory details)
- Cash flow statements
- Ratios and metrics

Free tier: 250 requests/day
Get free API key: https://site.financialmodelingprep.com/
"""

import logging
import os
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import hashlib
import json

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FMPFetcher:
    """Fetch detailed quarterly fundamentals from Financial Modeling Prep."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./data/cache"):
        """Initialize FMP fetcher.

        Args:
            api_key: FMP API key (or set FMP_API_KEY env variable)
            cache_dir: Directory for caching responses
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')

        if not self.api_key:
            logger.warning(
                "No FMP API key found! Set FMP_API_KEY environment variable or pass api_key parameter.\n"
                "Get free key at: https://site.financialmodelingprep.com/developer/docs"
            )

        self.base_url = "https://financialmodelingprep.com/stable"
        self.cache_dir = Path(cache_dir) / "fmp"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = Path("./data/state/fmp_usage.json")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Bandwidth tracking (30-day limit: 20 GB)
        self.bandwidth_used = 0
        self.bandwidth_limit = 20 * 1024 * 1024 * 1024  # 20 GB in bytes
        self.cooldown_seconds = 120
        self.max_retry_attempts = 4

        self.usage_state = self._load_usage_state()
        self.bandwidth_used = int(self.usage_state.get('bandwidth_used', 0))

        logger.info("FMPFetcher initialized")

    def _load_usage_state(self) -> Dict[str, any]:
        """Load persisted usage state from local JSON store."""
        default_state = {
            'attempted_calls': 0,
            'successful_calls': 0,
            'throttled_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'bandwidth_used': 0,
            'endpoint_cooldowns': {},
            'endpoint_global_cooldowns': {},
            'endpoint_429_streaks': {},
            'global_429_streak': 0,
            'request_log': []
        }
        if not self.state_path.exists():
            return default_state

        try:
            with open(self.state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            default_state.update(data)
            return default_state
        except Exception as e:
            logger.warning(f"Failed to load FMP usage state, using defaults: {e}")
            return default_state

    def _save_usage_state(self):
        """Persist usage state to disk."""
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.usage_state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist FMP usage state: {e}")

    def _build_request_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Build stable request key used for cache/cooldown tracking."""
        param_dict = params.copy() if params else {}
        param_dict.pop('apikey', None)
        stable_params = json.dumps(param_dict, sort_keys=True)
        return f"{endpoint}|{stable_params}"

    def _record_request_event(
        self,
        endpoint: str,
        request_key: str,
        status_code: Optional[int],
        cache_status: str,
        note: str = ""
    ):
        """Record endpoint-level request telemetry."""
        log = self.usage_state.setdefault('request_log', [])
        log.append({
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'request_key': request_key,
            'status_code': status_code,
            'cache': cache_status,
            'note': note
        })
        # Keep state file lightweight
        self.usage_state['request_log'] = log[-200:]
        self._save_usage_state()

    def is_cooldown_active(self, endpoint: str, params: Optional[Dict] = None) -> bool:
        """Return True when endpoint+params is currently in cooldown."""
        request_key = self._build_request_key(endpoint, params)
        cooldowns = self.usage_state.get('endpoint_cooldowns', {})
        until_raw = cooldowns.get(request_key)
        if not until_raw:
            return False
        try:
            return datetime.now() < datetime.fromisoformat(until_raw)
        except ValueError:
            return False

    def is_endpoint_cooldown_active(self, endpoint: str) -> bool:
        """Return True when any request variant for an endpoint is in cooldown."""
        cooldowns = self.usage_state.get('endpoint_cooldowns', {})
        now = datetime.now()
        for request_key, until_raw in cooldowns.items():
            if not request_key.startswith(f"{endpoint}|"):
                continue
            try:
                if now < datetime.fromisoformat(until_raw):
                    return True
            except ValueError:
                continue
        return False

    def _is_global_cooldown_active(self, endpoint: str) -> bool:
        """Return True when global or endpoint-level quota cooldown is active."""
        now = datetime.now()
        global_cooldowns = self.usage_state.setdefault('endpoint_global_cooldowns', {})

        for cooldown_key in ("__global__", endpoint):
            until_raw = global_cooldowns.get(cooldown_key)
            if not until_raw:
                continue
            try:
                if now < datetime.fromisoformat(until_raw):
                    return True
            except ValueError:
                continue

        return False

    def _request_json(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = 10,
        full_url: Optional[str] = None,
    ) -> Optional[Dict]:
        """Centralized HTTP request wrapper with cooldown + retry telemetry."""
        params = params.copy() if params else {}
        request_key = self._build_request_key(endpoint, params)
        global_cooldowns = self.usage_state.setdefault('endpoint_global_cooldowns', {})
        endpoint_429_streaks = self.usage_state.setdefault('endpoint_429_streaks', {})

        # Short-circuit when global/endpoint quota cooldown is active
        now = datetime.now()
        for cooldown_key in ("__global__", endpoint):
            until_raw = global_cooldowns.get(cooldown_key)
            if not until_raw:
                continue
            try:
                cooldown_until = datetime.fromisoformat(until_raw)
                if now < cooldown_until:
                    self.usage_state['attempted_calls'] += 1
                    self.usage_state['throttled_calls'] += 1
                    self._record_request_event(
                        endpoint,
                        request_key,
                        429,
                        'miss',
                        f'global_cooldown_short_circuit:{cooldown_key}'
                    )
                    logger.warning(
                        f"FMP global cooldown active for {cooldown_key}; skipping HTTP call and using fallback path."
                    )
                    return None
            except ValueError:
                pass

        # Short-circuit when endpoint+params variant is cooling down
        cooldowns = self.usage_state.setdefault('endpoint_cooldowns', {})
        cooldown_until_raw = cooldowns.get(request_key)
        if cooldown_until_raw:
            try:
                cooldown_until = datetime.fromisoformat(cooldown_until_raw)
                if now < cooldown_until:
                    self.usage_state['attempted_calls'] += 1
                    self.usage_state['throttled_calls'] += 1
                    self._record_request_event(endpoint, request_key, 429, 'miss', 'cooldown_short_circuit')
                    logger.warning(f"FMP cooldown active for {endpoint}; using fallback path.")
                    return None
            except ValueError:
                pass

        if not self.api_key:
            logger.error("Cannot fetch without API key")
            return None

        params['apikey'] = self.api_key
        url = full_url or f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retry_attempts):
            self.usage_state['attempted_calls'] += 1
            try:
                # Rate limiting baseline
                time.sleep(0.1)
                response = requests.get(url, params=params, timeout=timeout)
                status_code = response.status_code

                if status_code == 429:
                    self.usage_state['throttled_calls'] += 1
                    endpoint_streak = int(endpoint_429_streaks.get(endpoint, 0)) + 1
                    endpoint_429_streaks[endpoint] = endpoint_streak
                    global_streak = int(self.usage_state.get('global_429_streak', 0)) + 1
                    self.usage_state['global_429_streak'] = global_streak

                    long_cooldown_minutes = 0
                    if endpoint_streak >= 2:
                        long_cooldown_minutes = 15
                    if endpoint_streak >= 4:
                        long_cooldown_minutes = 60

                    if long_cooldown_minutes:
                        cooldown_until = datetime.now() + timedelta(minutes=long_cooldown_minutes)
                        global_cooldowns[endpoint] = cooldown_until.isoformat()
                        self._record_request_event(
                            endpoint,
                            request_key,
                            429,
                            'miss',
                            f'endpoint_global_cooldown_{long_cooldown_minutes}m'
                        )
                        logger.warning(
                            f"FMP endpoint quota exhaustion suspected for {endpoint}; "
                            f"setting {long_cooldown_minutes}m endpoint cooldown and returning fallback."
                        )
                        return None

                    if global_streak >= 6:
                        cooldown_until = datetime.now() + timedelta(minutes=30)
                        global_cooldowns["__global__"] = cooldown_until.isoformat()
                        self._record_request_event(
                            endpoint,
                            request_key,
                            429,
                            'miss',
                            'global_quota_cooldown_30m'
                        )
                        logger.warning(
                            "FMP global quota exhaustion suspected; setting 30m global cooldown and returning fallback."
                        )
                        return None

                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    cooldown_until = datetime.now() + timedelta(seconds=max(self.cooldown_seconds, int(delay)))
                    cooldowns[request_key] = cooldown_until.isoformat()
                    self._record_request_event(endpoint, request_key, 429, 'miss', f'retry_in_{delay:.2f}s')
                    logger.warning(
                        f"FMP returned 429 for {endpoint} (attempt {attempt + 1}/{self.max_retry_attempts}); "
                        f"backing off {delay:.2f}s."
                    )
                    time.sleep(delay)
                    continue

                if status_code == 403:
                    msg = response.json().get('Error Message', '') if response.text else 'Forbidden'
                    self._record_request_event(endpoint, request_key, 403, 'miss', msg)
                    logger.error(f"FMP API 403 Error: {msg}. (Endpoint: {endpoint})")
                    return {'__request_error__': True, 'status': 403, 'msg': msg}

                response.raise_for_status()
                self.bandwidth_used += len(response.content)
                self.usage_state['bandwidth_used'] = self.bandwidth_used
                self.usage_state['successful_calls'] += 1
                self.usage_state['global_429_streak'] = 0
                endpoint_429_streaks[endpoint] = 0
                cooldowns.pop(request_key, None)
                self._record_request_event(endpoint, request_key, status_code, 'miss', 'success')
                return response.json()
            except requests.exceptions.RequestException as e:
                self._record_request_event(endpoint, request_key, None, 'miss', f'error:{e}')
                logger.error(f"Error fetching from FMP ({endpoint}): {e}")
                break

        return None

    def _get_cache_path(self, ticker: str, endpoint: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{ticker}_{endpoint}.pkl"

    def _is_cache_valid(self, cache_path: Path, hours: int = 24) -> bool:
        """Check if cache is valid.

        Uses longer cache (7 days) for non-earnings periods,
        shorter cache (6 hours) during earnings season.
        """
        if not cache_path.exists():
            return False

        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)

        # Adjust cache duration based on earnings season proximity
        if self._is_earnings_season():
            # During earnings season (Jan 15-Feb 15, Apr 15-May 15, Jul 15-Aug 15, Oct 15-Nov 15)
            # Use shorter cache to catch new earnings
            cache_hours = 6
        else:
            # Outside earnings season, use longer cache to save bandwidth
            cache_hours = 168  # 7 days

        return datetime.now() - file_time < timedelta(hours=cache_hours)

    def _is_earnings_season(self) -> bool:
        """Check if currently in earnings season.

        Earnings seasons (roughly):
        - Q4: Jan 15 - Feb 15
        - Q1: Apr 15 - May 15
        - Q2: Jul 15 - Aug 15
        - Q3: Oct 15 - Nov 15
        """
        now = datetime.now()
        month = now.month
        day = now.day

        earnings_windows = [
            (1, 15, 2, 15),  # Q4 earnings: Jan 15 - Feb 15
            (4, 15, 5, 15),  # Q1 earnings: Apr 15 - May 15
            (7, 15, 8, 15),  # Q2 earnings: Jul 15 - Aug 15
            (10, 15, 11, 15) # Q3 earnings: Oct 15 - Nov 15
        ]

        for start_month, start_day, end_month, end_day in earnings_windows:
            if start_month == end_month:
                if month == start_month and start_day <= day <= end_day:
                    return True
            else:
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day):
                    return True

        return False

    def _fetch(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch from FMP API with rate limiting and disk caching.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response or None
        """
        # 1. Check Disk Cache
        param_dict = params.copy() if params else {}
        param_dict.pop('apikey', None)

        stable_params = json.dumps(param_dict, sort_keys=True)
        param_hash = hashlib.md5(stable_params.encode()).hexdigest()[:10]
        cache_key = f"{endpoint.replace('/', '_')}_{param_hash}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        request_key = self._build_request_key(endpoint, param_dict)

        # Cache duration: 30 days for fundamentals, 24h for news/others
        if cache_path.exists():
            try:
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                
                # Determine cache duration based on endpoint type
                fundamental_endpoints = {
                    'income-statement', 'balance-sheet-statement', 'cash-flow-statement', 
                    'key-metrics', 'discounted-cash-flow', 'financial-ratios'
                }
                is_fundamental = any(fe in endpoint for fe in fundamental_endpoints)
                
                news_endpoints = {'stock_news', 'press-releases'}
                is_news = any(ne in endpoint for ne in news_endpoints)
                
                if is_fundamental:
                    # User request: every month once only for fundamentals
                    success_duration = timedelta(days=30)
                elif is_news:
                    # User request: news daily
                    success_duration = timedelta(hours=24)
                else:
                    # Fallback success duration
                    success_duration = timedelta(days=7) if not self._is_earnings_season() else timedelta(hours=12)
                
                # Fail/Error duration
                error_duration = timedelta(hours=24)
                
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Check if it was a cached error
                if isinstance(cached_data, dict) and cached_data.get('__cached_error__'):
                    if datetime.now() - mtime < error_duration:
                        logger.warning(f"FMP CACHE HIT (Error): {endpoint} - Skipping to save limit.")
                        self.usage_state['cache_hits'] += 1
                        self._record_request_event(endpoint, request_key, cached_data.get('status'), 'hit', 'cached_error')
                        return None
                elif datetime.now() - mtime < success_duration:
                    logger.info(f"FMP CACHE HIT ({'Monthly' if is_fundamental else 'Daily'}): {endpoint}")
                    self.usage_state['cache_hits'] += 1
                    self._record_request_event(endpoint, request_key, 200, 'hit', 'cached_success')
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to read cache for {endpoint}: {e}")

        logger.info(f"FMP CACHE MISS: Fetching {endpoint} from API...")
        self.usage_state['cache_misses'] += 1
        self._record_request_event(endpoint, request_key, None, 'miss', 'cache_miss')

        # 2. Fetch from API if not cached or expired
        data = self._request_json(endpoint, params=param_dict, timeout=10)
        try:
            if isinstance(data, dict) and data.get('__request_error__') and data.get('status') == 403:
                msg = data.get('msg', 'Forbidden')
                # Cache the 403 error to avoid retrying this endpoint
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'__cached_error__': True, 'status': 403, 'msg': msg}, f)
                except Exception as cache_err:
                    logger.warning(f"Failed to cache error for {endpoint}: {cache_err}")

                if "Special Endpoint" in msg or "Legacy Endpoint" in msg:
                    logger.warning("This endpoint requires a higher FMP plan tier or is no longer supported for this key. System will fallback to yfinance.")
                return None

            if data is None:
                return None

            # Check bandwidth limit
            if self.bandwidth_used > self.bandwidth_limit:
                logger.warning(
                    f"FMP bandwidth limit exceeded! "
                    f"Used: {self.bandwidth_used / 1024 / 1024:.1f} MB / "
                    f"{self.bandwidth_limit / 1024 / 1024 / 1024:.1f} GB"
                )

            # Check for error in response
            if isinstance(data, dict) and 'Error Message' in data:
                logger.error(f"FMP API error: {data['Error Message']}")
                return None

            # 3. Save to Cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache for {endpoint}: {e}")

            return data

        except Exception as e:
            logger.error(f"Error fetching from FMP: {e}")
            return None

    def fetch_income_statement(self, ticker: str, quarterly: bool = True, limit: int = 8) -> List[Dict]:
        """Fetch income statement data.

        Args:
            ticker: Stock ticker
            quarterly: True for quarterly, False for annual
            limit: Number of periods to fetch

        Returns:
            List of income statement periods
        """
        # Fetch from API
        period = "quarter" if quarterly else "annual"
        endpoint = "income-statement"
        params = {'symbol': ticker, 'period': period, 'limit': limit}

        data = self._fetch(endpoint, params)
        return data or []

    def fetch_balance_sheet(self, ticker: str, quarterly: bool = True, limit: int = 8) -> List[Dict]:
        """Fetch balance sheet data.

        Args:
            ticker: Stock ticker
            quarterly: True for quarterly, False for annual
            limit: Number of periods to fetch

        Returns:
            List of balance sheet periods
        """
        # Fetch from API
        period = "quarter" if quarterly else "annual"
        endpoint = "balance-sheet-statement"
        params = {'symbol': ticker, 'period': period, 'limit': limit}

        data = self._fetch(endpoint, params)
        return data or []

    def fetch_cash_flow(self, ticker: str, quarterly: bool = True, limit: int = 8) -> List[Dict]:
        """Fetch cash flow statement data.

        Args:
            ticker: Stock ticker
            quarterly: True for quarterly, False for annual
            limit: Number of periods to fetch

        Returns:
            List of cash flow periods
        """
        # Fetch from API
        period = "quarter" if quarterly else "annual"
        endpoint = "cash-flow-statement"
        params = {'symbol': ticker, 'period': period, 'limit': limit}

        data = self._fetch(endpoint, params)
        return data or []

    def fetch_key_metrics(self, ticker: str, quarterly: bool = True, limit: int = 8) -> List[Dict]:
        """Fetch key financial metrics and ratios.

        Args:
            ticker: Stock ticker
            quarterly: True for quarterly, False for annual
            limit: Number of periods to fetch

        Returns:
            List of metric periods
        """
        # Fetch from API
        period = "quarter" if quarterly else "annual"
        endpoint = "key-metrics"
        params = {'symbol': ticker, 'period': period, 'limit': limit}

        data = self._fetch(endpoint, params)
        return data or []

    def fetch_dcf(self, ticker: str) -> Optional[float]:
        """Fetch Discounted Cash Flow (intrinsic value)."""
        data = self._fetch("discounted-cash-flow", {'symbol': ticker})
        
        if data and isinstance(data, list) and len(data) > 0:
            return data[0].get('dcf')
        return None

    def fetch_insider_trading(self, ticker: str, page: int = 0) -> List[Dict]:
        """Fetch insider trading data."""
        cache_key = "insider"
        cache_path = self._get_cache_path(ticker, cache_key)

        if self._is_cache_valid(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Insider trading is v4 endpoint
        url = "https://financialmodelingprep.com/api/v4/insider-trading"
        params = {'symbol': ticker, 'page': page}
        data = self._request_json('api_v4/insider-trading', params=params, timeout=10, full_url=url)
        if not data:
            return []

        # Cache result
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return data

    def fetch_economic_data(self) -> Dict[str, any]:
        """Fetch key economic indicators (CPI, GDP, Unemployment, Interest Rate)."""
        indicators = {
            'CPI': 'CPI',
            'GDP': 'GDP',
            'Unemployment': 'unemploymentRate',
            'FedFunds': 'federalFunds'
        }
        
        results = {}
        for name, endpoint_key in indicators.items():
            cache_key = f"econ_{name}"
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            
            # Use 24h cache for economic data (doesn't change intraday)
            if self._is_cache_valid(cache_path, hours=24):
                 with open(cache_path, 'rb') as f:
                    results[name] = pickle.load(f)
                    continue

            # Fetch
            endpoint = "economic-indicators"
            params = {'name': endpoint_key}
            data = self._fetch(endpoint, params)
            if data:
                # Store just the latest value and the previous one for trend
                latest = data[0] if len(data) > 0 else {}
                prev = data[1] if len(data) > 1 else {}
                
                info = {
                    'current': latest.get('value'),
                    'date': latest.get('date'),
                    'previous': prev.get('value'),
                    'trend': 'Up' if (latest.get('value',0) > prev.get('value',0)) else 'Down'
                }
                results[name] = info
                
                with open(cache_path, 'wb') as f:
                    pickle.dump(info, f)
                    
        return results

    def fetch_market_news(self, limit: int = 10) -> List[Dict]:
        """Fetch general market news."""
        # Cache for 1 hour only as news is standard
        cache_path = self.cache_dir / "market_news.pkl"
        
        if self._is_cache_valid(cache_path, hours=1):
             with open(cache_path, 'rb') as f:
                return pickle.load(f)
                
        # https://financialmodelingprep.com/api/v3/stock_news?limit=10 (for general stock news) 
        # or fmp/general_news if available, but stock_news without ticker is broad
        
        data = self._fetch("stock_news", {'limit': limit})
        if data:
             with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        
        return data or []

    def fetch_sector_performance(self) -> List[Dict]:
        """Fetch real-time sector performance."""
        cache_path = self.cache_dir / "sector_performance.pkl"
        if self._is_cache_valid(cache_path, hours=1):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # v3 endpoint for sector performance
        data = self._fetch("sector-performance", {})
        if data:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        return data or []

    def fetch_economic_calendar(self, days_forward: int = 7) -> List[Dict]:
        """Fetch upcoming economic events."""
        cache_path = self.cache_dir / "econ_calendar.pkl"
        if self._is_cache_valid(cache_path, hours=4):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        from datetime import date
        today = date.today()
        future = today + timedelta(days=days_forward)
        
        # v3 endpoint
        params = {
            'from': today.strftime('%Y-%m-%d'),
            'to': future.strftime('%Y-%m-%d')
        }
        data = self._fetch("economic_calendar", params)
        if data:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        return data or []

    def fetch_stock_news(self, tickers: List[str], limit: int = 5) -> List[Dict]:
        """Fetch news for specific stock tickers.
        
        Args:
            tickers: List of stock symbols
            limit: Number of news items per ticker
            
        Returns:
            List of news items
        """
        if not tickers:
            return []
            
        # No caching for specific ticker news to keep it fresh
        params = {
            "tickers": ",".join(tickers),
            "limit": limit
        }
        return self._fetch("stock_news", params) or []

    def fetch_comprehensive_fundamentals(self, ticker: str, include_advanced: bool = True) -> Dict:
        """Fetch comprehensive quarterly fundamentals + optional advanced metrics.
        
        Args:
            ticker: Stock ticker
            include_advanced: If True, fetches DCF and Insider data (extra API calls)
        """
        logger.info(f"Fetching comprehensive fundamentals for {ticker}")
        
        data = {
            'ticker': ticker,
            'income_statement': self.fetch_income_statement(ticker, quarterly=True, limit=8),
            'balance_sheet': self.fetch_balance_sheet(ticker, quarterly=True, limit=8),
            'cash_flow': self.fetch_cash_flow(ticker, quarterly=True, limit=8),
            'key_metrics': self.fetch_key_metrics(ticker, quarterly=True, limit=8),
            'fetch_date': datetime.now().isoformat()
        }
        
        if include_advanced:
            data['dcf'] = self.fetch_dcf(ticker)
            data['insider_trading'] = self.fetch_insider_trading(ticker)
            
        return data

    def create_enhanced_snapshot(self, ticker: str, data: Dict = None) -> str:
        """Create enhanced fundamental snapshot with net margins, inventory, etc.

        Args:
            ticker: Stock ticker
            data: Pre-fetched data, or will fetch if None

        Returns:
            Formatted snapshot string
        """
        if data is None:
            data = self.fetch_comprehensive_fundamentals(ticker)

        if not data or not data.get('income_statement'):
            return f"ENHANCED FUNDAMENTAL SNAPSHOT - {ticker}\nNo data available"

        snapshot = [
            "",
            "="*60,
            f"ENHANCED FUNDAMENTAL SNAPSHOT - {ticker}",
            "="*60,
            ""
        ]

        # Latest quarter data
        income = data['income_statement'][0] if data.get('income_statement') else {}
        balance = data['balance_sheet'][0] if data.get('balance_sheet') else {}
        prev_income = data['income_statement'][1] if len(data.get('income_statement', [])) > 1 else {}
        prev_balance = data['balance_sheet'][1] if len(data.get('balance_sheet', [])) > 1 else {}

        # Revenue analysis
        revenue = income.get('revenue', 0)
        prev_revenue = prev_income.get('revenue', 0)

        if revenue and prev_revenue:
            rev_change = ((revenue - prev_revenue) / prev_revenue * 100)
            if rev_change > 20:
                snapshot.append(f"✓ Revenue: ACCELERATING (${revenue/1e9:.2f}B, +{rev_change:.1f}% QoQ)")
            elif rev_change > 5:
                snapshot.append(f"✓ Revenue: Growing (${revenue/1e9:.2f}B, +{rev_change:.1f}% QoQ)")
            elif rev_change > 0:
                snapshot.append(f"• Revenue: Modest growth (${revenue/1e9:.2f}B, +{rev_change:.1f}% QoQ)")
            else:
                snapshot.append(f"✗ Revenue: DECLINING (${revenue/1e9:.2f}B, {rev_change:.1f}% QoQ)")

        # EPS analysis
        eps = income.get('eps', 0)
        prev_eps = prev_income.get('eps', 0)

        if eps and prev_eps:
            eps_change = ((eps - prev_eps) / abs(prev_eps) * 100) if prev_eps != 0 else 0
            if eps_change > 25:
                snapshot.append(f"✓ EPS: STRONG growth (${eps:.2f}, +{eps_change:.1f}% QoQ)")
            elif eps_change > 10:
                snapshot.append(f"✓ EPS: Growing (${eps:.2f}, +{eps_change:.1f}% QoQ)")
            elif eps_change > 0:
                snapshot.append(f"• EPS: Slight growth (${eps:.2f}, +{eps_change:.1f}% QoQ)")
            else:
                snapshot.append(f"✗ EPS: DECLINING (${eps:.2f}, {eps_change:.1f}% QoQ)")

        # Margin analysis - NET MARGINS!
        net_margin = (income.get('netIncomeRatio') or 0) * 100  # As percentage
        gross_margin = (income.get('grossProfitRatio') or 0) * 100
        operating_margin = (income.get('operatingIncomeRatio') or 0) * 100

        prev_net_margin = (prev_income.get('netIncomeRatio') or 0) * 100
        margin_change = net_margin - prev_net_margin

        snapshot.append("")
        snapshot.append("Margins:")
        snapshot.append(f"  Gross Margin:     {gross_margin:.1f}%")
        snapshot.append(f"  Operating Margin: {operating_margin:.1f}%")

        if margin_change > 1:
            snapshot.append(f"  Net Margin:       {net_margin:.1f}% ✓ EXPANDING (+{margin_change:.1f}pp)")
        elif margin_change > 0:
            snapshot.append(f"  Net Margin:       {net_margin:.1f}% • Stable (+{margin_change:.1f}pp)")
        elif margin_change > -1:
            snapshot.append(f"  Net Margin:       {net_margin:.1f}% • Flat ({margin_change:.1f}pp)")
        else:
            snapshot.append(f"  Net Margin:       {net_margin:.1f}% ✗ CONTRACTING ({margin_change:.1f}pp)")

        # Inventory analysis
        inventory = balance.get('inventory') or 0
        prev_inventory = prev_balance.get('inventory') or 0

        if inventory and prev_inventory:
            inv_change = ((inventory - prev_inventory) / prev_inventory * 100)
            inv_to_revenue = (inventory / revenue * 100) if revenue else 0

            snapshot.append("")
            snapshot.append("Inventory:")
            snapshot.append(f"  Total: ${inventory/1e9:.2f}B ({inv_to_revenue:.1f}% of revenue)")

            if inv_change > 15:
                snapshot.append(f"  ⚠ BUILDING rapidly (+{inv_change:.1f}% QoQ)")
                snapshot.append("  → Potential demand weakness")
            elif inv_change > 5:
                snapshot.append(f"  • Moderate build (+{inv_change:.1f}% QoQ)")
            elif inv_change > 0:
                snapshot.append(f"  • Slight increase (+{inv_change:.1f}% QoQ)")
            else:
                snapshot.append(f"  ✓ Drawing down ({inv_change:.1f}% QoQ)")
                snapshot.append("  → Strong demand signal")

        # Balance Sheet Health
        current_assets = balance.get('totalCurrentAssets') or 0
        current_liabilities = balance.get('totalCurrentLiabilities') or 0
        total_liabilities = balance.get('totalLiabilities') or 0
        total_equity = (balance.get('totalStockholdersEquity') or 
                        balance.get('totalEquity') or 
                        (balance.get('totalAssets') or 0) - (balance.get('totalLiabilities') or 0))
        net_income = income.get('netIncome') or 0

        current_ratio = current_assets / current_liabilities if current_liabilities else 0
        debt_to_equity = total_liabilities / total_equity if total_equity else 0
        roe = (net_income / total_equity * 100) if total_equity else 0

        snapshot.append("")
        snapshot.append("Balance Sheet & Efficiency:")
        snapshot.append(f"  Current Ratio:     {current_ratio:.2f} {'✓' if current_ratio > 1.5 else '⚠' if current_ratio < 1 else '•'}")
        snapshot.append(f"  Debt-to-Equity:    {debt_to_equity:.2f} {'✓' if debt_to_equity < 1 else '⚠' if debt_to_equity > 2 else '•'}")
        snapshot.append(f"  Return on Equity:  {roe:.1f}% {'✓' if roe > 15 else '•'}")

        # Overall assessment
        snapshot.append("")
        snapshot.append("Overall Assessment:")

        concerns = []
        if revenue and prev_revenue and ((revenue - prev_revenue) / prev_revenue) < 0:
            concerns.append('revenue declining')
        if eps and prev_eps and ((eps - prev_eps) / abs(prev_eps)) < 0:
            concerns.append('EPS declining')
        if margin_change < -2:
            concerns.append('margins contracting')
        if inventory and prev_inventory and ((inventory - prev_inventory) / prev_inventory) > 15:
            concerns.append('inventory building')
        if current_ratio < 1:
            concerns.append('low liquidity')
        if debt_to_equity > 3:
            concerns.append('high leverage')

        if len(concerns) == 0:
            snapshot.append("✓ Fundamentals SUPPORT technical breakout")
        else:
            snapshot.append(f"⚠ Concerns: {', '.join(concerns)}")
            if len(concerns) >= 2 or 'revenue declining' in concerns or 'EPS declining' in concerns:
                snapshot.append("✗ Fundamentals CONTRADICT technical breakout")

        snapshot.append("="*60)
        return "\n".join(snapshot)

    def fetch_stock_list(self, exchange: str = None) -> List[Dict]:
        """Fetch list of all listed stocks, optionally filtered by exchange.
        
        Args:
            exchange: Optional exchange filter ('NASDAQ', 'NYSE', 'AMEX')
            
        Returns:
            List of dicts with 'symbol', 'name', 'price', 'exchangeShortName'
        """
        cache_key = f"stock_list_{exchange or 'all'}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # 24h cache for stock lists
        if cache_path.exists():
            try:
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=24):
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"FMP stock list loaded from cache: {len(data)} symbols")
                    return data
            except Exception:
                pass
        
        # Use v3 stock list endpoint
        params = {}
        if exchange:
            params['exchange'] = exchange
        
        try:
            url = "https://financialmodelingprep.com/api/v3/stock/list"
            data = self._request_json('api_v3/stock/list', params=params, timeout=30, full_url=url)
            if not data:
                return []
            if not isinstance(data, list):
                logger.warning("FMP stock list returned unexpected format")
                return []
            
            # Filter to US exchanges only
            us_exchanges = {'NASDAQ', 'NYSE', 'AMEX', 'New York Stock Exchange', 'Nasdaq Global Select'}
            filtered = [
                {
                    'symbol': s.get('symbol', ''),
                    'name': s.get('name', ''),
                    'price': s.get('price', 0),
                    'exchange': s.get('exchangeShortName', s.get('exchange', '')),
                    'type': s.get('type', '')
                }
                for s in data
                if (s.get('exchangeShortName', '') in us_exchanges or 
                    s.get('exchange', '') in us_exchanges)
                and s.get('type', '') == 'stock'
            ]
            
            # Cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(filtered, f)
            except Exception as e:
                logger.warning(f"Failed to cache stock list: {e}")
            
            logger.info(f"FMP stock list fetched: {len(filtered)} US stocks")
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to fetch FMP stock list: {e}")
            return []

    def get_bandwidth_stats(self) -> Dict[str, any]:
        """Get bandwidth usage statistics.

        Returns:
            Dict with bandwidth usage info
        """
        used_mb = self.bandwidth_used / 1024 / 1024
        limit_gb = self.bandwidth_limit / 1024 / 1024 / 1024
        pct_used = (self.bandwidth_used / self.bandwidth_limit * 100) if self.bandwidth_limit > 0 else 0

        return {
            'bandwidth_used_mb': round(used_mb, 2),
            'bandwidth_limit_gb': round(limit_gb, 2),
            'bandwidth_pct_used': round(pct_used, 2),
            'is_earnings_season': self._is_earnings_season(),
            'cache_hours': 6 if self._is_earnings_season() else 168
        }

    def get_usage_stats(self) -> Dict[str, int]:
        """Get persisted API usage counters."""
        return {
            'attempted_calls': int(self.usage_state.get('attempted_calls', 0)),
            'successful_calls': int(self.usage_state.get('successful_calls', 0)),
            'throttled_calls': int(self.usage_state.get('throttled_calls', 0)),
            'cache_hits': int(self.usage_state.get('cache_hits', 0)),
            'cache_misses': int(self.usage_state.get('cache_misses', 0)),
        }
