"""FRED (Federal Reserve Economic Data) API fetcher.

This module provides an interface to the FRED API for retrieving economic data releases,
release tables, and series data.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FredFetcher:
    """Fetch economic data from FRED."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "./data/cache",
        cache_ttl_hours: int = 24,
    ):
        """Initialize FredFetcher.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env variable)
            cache_dir: Directory for caching responses
            cache_ttl_hours: Default cache TTL for FRED responses
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')

        if not self.api_key:
            logger.warning(
                "No FRED API key found! Set FRED_API_KEY environment variable.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.base_url = "https://api.stlouisfed.org/fred"
        self.cache_dir = Path(cache_dir) / "fred"
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FredFetcher initialized")

    def _fetch(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        cache_ttl_hours: Optional[int] = None,
    ) -> Optional[Dict]:
        """Fetch from FRED API with disk caching.

        Args:
            endpoint: API endpoint (e.g., 'releases')
            params: Query parameters

        Returns:
            JSON response or None
        """
        if not self.api_key:
            logger.error("Cannot fetch from FRED without API key")
            return None

        ttl_hours = cache_ttl_hours if cache_ttl_hours is not None else self.cache_ttl_hours

        # Prepare parameters
        params = params or {}
        params['api_key'] = self.api_key
        params['file_type'] = 'json'

        # Generate cache key based on endpoint and params (excluding api_key)
        cache_params = params.copy()
        if 'api_key' in cache_params:
            del cache_params['api_key']
        
        param_str = json.dumps(cache_params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:10]
        cache_filename = f"{endpoint.replace('/', '_')}_{param_hash}.json"
        cache_path = self.cache_dir / cache_filename

        # Check cache with explicit TTL
        if cache_path.exists():
            try:
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=ttl_hours):
                    with open(cache_path, 'r') as f:
                        logger.info(f"FRED CACHE HIT: {endpoint} (ttl={ttl_hours}h)")
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read FRED cache for {endpoint}: {e}")

        logger.info(f"FRED CACHE MISS: Fetching {endpoint} from API...")

        try:
            url = f"{self.base_url}/{endpoint}"
            # Respectful rate limiting: FRED limit is 120 requests/minute
            # We'll add a small delay if needed, but the cache should handle most cases
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Save to cache
            try:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save FRED cache for {endpoint}: {e}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from FRED ({endpoint}): {e}")
            return None

    def fetch_releases(
        self, 
        realtime_start: Optional[str] = None,
        realtime_end: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: str = 'release_id',
        sort_order: str = 'asc'
    ) -> List[Dict]:
        """Get all releases of economic data.

        Args:
            realtime_start: Start of real-time period (YYYY-MM-DD)
            realtime_end: End of real-time period (YYYY-MM-DD)
            limit: Maximum results (1-1000)
            offset: Result offset
            order_by: Attribute to order by
            sort_order: 'asc' or 'desc'

        Returns:
            List of release dictionaries
        """
        params = {
            'limit': limit,
            'offset': offset,
            'order_by': order_by,
            'sort_order': sort_order
        }
        if realtime_start: params['realtime_start'] = realtime_start
        if realtime_end: params['realtime_end'] = realtime_end

        data = self._fetch("releases", params)
        return data.get('releases', []) if data else []

    def fetch_release(
        self, 
        release_id: int,
        realtime_start: Optional[str] = None,
        realtime_end: Optional[str] = None
    ) -> Optional[Dict]:
        """Get a specific release of economic data.

        Args:
            release_id: The ID for the release
            realtime_start: Start of real-time period (YYYY-MM-DD)
            realtime_end: End of real-time period (YYYY-MM-DD)

        Returns:
            Release dictionary or None
        """
        params = {'release_id': release_id}
        if realtime_start: params['realtime_start'] = realtime_start
        if realtime_end: params['realtime_end'] = realtime_end

        data = self._fetch("release", params)
        if data and 'releases' in data and len(data['releases']) > 0:
            return data['releases'][0]
        return None

    def fetch_release_tables(
        self, 
        release_id: int, 
        element_id: Optional[int] = None,
        include_observation_values: bool = False,
        observation_date: Optional[str] = None
    ) -> Dict:
        """Get release table trees for a given release.

        Args:
            release_id: The ID for the release
            element_id: Release table element ID to retrieve
            include_observation_values: Whether to include observation values
            observation_date: Specific observation date (YYYY-MM-DD)

        Returns:
            Dictionary containing table tree structure
        """
        params = {'release_id': release_id}
        if element_id:
            params['element_id'] = element_id
        if include_observation_values:
            params['include_observation_values'] = 'true'
        if observation_date:
            params['observation_date'] = observation_date

        return self._fetch("release/tables", params) or {}

    def fetch_series_observations(
        self, 
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        units: str = 'lin',
        frequency: Optional[str] = None,
        aggregation_method: str = 'avg',
        limit: int = 1000,
        offset: int = 0,
        cache_ttl_hours: Optional[int] = None,
    ) -> List[Dict]:
        """Fetch observations for a specific economic series.
        
        Common Series IDs:
        - GDP: 'GDP'
        - CPI: 'CPIAUCSL'
        - Unemployment: 'UNRATE'
        - Fed Funds Rate: 'FEDFUNDS'

        Args:
            series_id: FRED series ID
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
            units: Data units ('lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log')
            frequency: Data frequency ('d', 'w', 'bw', 'm', 'q', 'sa', 'a')
            aggregation_method: 'avg', 'sum', 'eop'
            limit: Maximum results (1-100000)
            offset: Result offset
            cache_ttl_hours: Optional cache TTL override

        Returns:
            List of observation dictionaries
        """
        params = {
            'series_id': series_id,
            'units': units,
            'aggregation_method': aggregation_method,
            'limit': limit,
            'offset': offset
        }
        if observation_start: params['observation_start'] = observation_start
        if observation_end: params['observation_end'] = observation_end
        if frequency: params['frequency'] = frequency

        data = self._fetch("series/observations", params, cache_ttl_hours=cache_ttl_hours)
        return data.get('observations', []) if data else []

    @staticmethod
    def _latest_numeric(observations: List[Dict]) -> Optional[float]:
        """Extract latest numeric value from FRED observations."""
        for observation in reversed(observations):
            value = observation.get("value")
            if value in (None, "."):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _infer_trend(values: List[float]) -> str:
        """Infer directional trend from the first/last value."""
        if len(values) < 2:
            return "Stable"
        if values[-1] > values[0]:
            return "Up"
        if values[-1] < values[0]:
            return "Down"
        return "Stable"

    def get_policy_rate_trend(self) -> Dict[str, str]:
        """Return a policy rate trend proxy from FEDFUNDS."""
        if not self.api_key:
            return {
                "name": "Policy Rate Trend",
                "status": "unavailable",
                "summary": "FRED API key missing; policy-rate trend proxy unavailable.",
            }

        observations = self.fetch_series_observations(
            "FEDFUNDS",
            limit=6,
            cache_ttl_hours=self.cache_ttl_hours,
        )
        numeric_values = [
            float(obs["value"]) for obs in observations if obs.get("value") not in (None, ".")
        ]
        if not numeric_values:
            return {
                "name": "Policy Rate Trend",
                "status": "stale",
                "summary": "No valid Fed Funds observations available.",
            }

        latest = numeric_values[-1]
        trend = self._infer_trend(numeric_values[-3:])
        return {
            "name": "Policy Rate Trend",
            "status": trend.lower(),
            "summary": f"Fed Funds at {latest:.2f}% with a {trend.lower()} short-term policy trend.",
        }

    def get_yield_spread_regime(self) -> Dict[str, str]:
        """Return 2Y/10Y regime using DGS2 and DGS10 spread."""
        if not self.api_key:
            return {
                "name": "2Y/10Y Spread Regime",
                "status": "unavailable",
                "summary": "FRED API key missing; 2Y/10Y regime unavailable.",
            }

        ten_year = self._latest_numeric(self.fetch_series_observations("DGS10", limit=10, cache_ttl_hours=6))
        two_year = self._latest_numeric(self.fetch_series_observations("DGS2", limit=10, cache_ttl_hours=6))
        if ten_year is None or two_year is None:
            return {
                "name": "2Y/10Y Spread Regime",
                "status": "stale",
                "summary": "Insufficient Treasury observations for spread regime.",
            }

        spread = ten_year - two_year
        if spread < 0:
            regime = "Inverted"
            context = "risk-off growth expectations remain elevated"
        elif spread < 0.5:
            regime = "Flat"
            context = "late-cycle uncertainty"
        else:
            regime = "Steep"
            context = "risk-on macro backdrop"

        return {
            "name": "2Y/10Y Spread Regime",
            "status": regime.lower(),
            "summary": f"2s10s spread at {spread:.2f} pp ({regime}); {context}.",
        }

    def get_inflation_momentum_proxy(self) -> Dict[str, str]:
        """Return inflation momentum proxy from YoY CPI."""
        if not self.api_key:
            return {
                "name": "Inflation Momentum Proxy",
                "status": "unavailable",
                "summary": "FRED API key missing; inflation momentum proxy unavailable.",
            }

        observations = self.fetch_series_observations(
            "CPIAUCSL",
            units="pc1",
            limit=6,
            cache_ttl_hours=24,
        )
        numeric_values = [
            float(obs["value"]) for obs in observations if obs.get("value") not in (None, ".")
        ]
        if len(numeric_values) < 2:
            return {
                "name": "Inflation Momentum Proxy",
                "status": "stale",
                "summary": "Insufficient CPI observations for momentum.",
            }

        latest = numeric_values[-1]
        prior = numeric_values[-2]
        direction = "accelerating" if latest > prior else "cooling" if latest < prior else "stable"
        return {
            "name": "Inflation Momentum Proxy",
            "status": direction,
            "summary": f"YoY CPI momentum is {direction} ({latest:.2f}% vs {prior:.2f}% prior).",
        }

    def get_labor_stress_proxy(self) -> Dict[str, str]:
        """Return labor stress proxy combining unemployment and claims direction."""
        if not self.api_key:
            return {
                "name": "Labor Stress Proxy",
                "status": "unavailable",
                "summary": "FRED API key missing; labor stress proxy unavailable.",
            }

        unemployment_obs = self.fetch_series_observations("UNRATE", limit=4, cache_ttl_hours=24)
        claims_obs = self.fetch_series_observations("ICSA", limit=4, cache_ttl_hours=24)
        unemployment_values = [
            float(obs["value"]) for obs in unemployment_obs if obs.get("value") not in (None, ".")
        ]
        claims_values = [float(obs["value"]) for obs in claims_obs if obs.get("value") not in (None, ".")]

        if len(unemployment_values) < 2 or len(claims_values) < 2:
            return {
                "name": "Labor Stress Proxy",
                "status": "stale",
                "summary": "Insufficient labor observations for stress proxy.",
            }

        unrate_change = unemployment_values[-1] - unemployment_values[-2]
        claims_change_pct = (claims_values[-1] / claims_values[-2] - 1) * 100 if claims_values[-2] else 0

        if unrate_change > 0.1 and claims_change_pct > 2:
            regime = "elevated"
            context = "Labor stress is building and supports a defensive risk posture"
        elif unrate_change < 0 and claims_change_pct < 0:
            regime = "improving"
            context = "Labor conditions are improving and support cyclical participation"
        else:
            regime = "stable"
            context = "Labor backdrop is mixed without a clear stress impulse"

        return {
            "name": "Labor Stress Proxy",
            "status": regime,
            "summary": (
                f"Unemployment change {unrate_change:+.2f}pp; claims {claims_change_pct:+.1f}% "
                f"({context})."
            ),
        }

    def get_fixed_macro_panel(self) -> Dict[str, Dict[str, str]]:
        """Build a fixed macro panel used by newsletter generation."""
        return {
            "policy_rate_trend": self.get_policy_rate_trend(),
            "yield_spread_regime": self.get_yield_spread_regime(),
            "inflation_momentum_proxy": self.get_inflation_momentum_proxy(),
            "labor_stress_proxy": self.get_labor_stress_proxy(),
        }
