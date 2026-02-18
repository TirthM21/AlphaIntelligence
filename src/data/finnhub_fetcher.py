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
from typing import Any, Dict, List, Optional
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

    def _safe_get(self, endpoint: str, params: Dict, timeout: int = 10) -> Dict:
        """Perform a Finnhub GET and return parsed JSON dict/list wrapper."""
        if not self.api_key:
            return {}
        try:
            query = {**params, "token": self.api_key}
            response = self.session.get(f"{self.base_url}/{endpoint}", params=query, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list):
                return {"items": payload}
        except Exception as exc:
            logger.error(f"Finnhub request failed for {endpoint}: {exc}")
        return {}

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


    def fetch_us_stock_symbols(self) -> List[Dict]:
        """Fetch US-listed equity symbols for universe fallback usage."""
        cache_path = self._get_cache_path("us_stock_symbols")
        if self._is_cache_valid(cache_path, hours=24):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if not self.api_key:
            return []

        exchanges = ["US"]
        symbols: List[Dict] = []
        seen = set()

        for exch in exchanges:
            try:
                response = self.session.get(
                    f"{self.base_url}/stock/symbol",
                    params={"exchange": exch, "token": self.api_key},
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json() or []
                if not isinstance(payload, list):
                    continue

                for item in payload:
                    symbol = (item.get('symbol') or '').strip().upper()
                    name = (item.get('description') or symbol).strip()
                    instrument_type = (item.get('type') or '').upper()
                    if not symbol or symbol in seen:
                        continue
                    if instrument_type and instrument_type not in {'COMMON STOCK', 'EQS', 'ETP'}:
                        continue
                    seen.add(symbol)
                    symbols.append({'symbol': symbol, 'name': name})
            except Exception as exc:
                logger.warning(f"Failed to fetch Finnhub US symbols for exchange {exch}: {exc}")

        if symbols:
            with open(cache_path, 'wb') as f:
                pickle.dump(symbols, f)

        return symbols

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

    def fetch_top_market_news(self, limit: int = 10, category: str = "general") -> List[Dict]:
        """Fetch and normalize top market headlines ranked by recency and content quality."""
        news = self.fetch_market_news(category=category)
        if not news:
            return []

        def score(item: Dict) -> float:
            now_ts = datetime.now().timestamp()
            ts = float(item.get("datetime") or 0)
            age_hours = max(0.0, (now_ts - ts) / 3600.0) if ts else 72.0
            recency_score = max(0.0, 120.0 - min(age_hours, 120.0))
            headline = (item.get("headline") or "").strip()
            summary = (item.get("summary") or "").strip()
            quality_bonus = min(30.0, len(headline) * 0.35) + min(20.0, len(summary) * 0.04)
            return recency_score + quality_bonus

        ranked = sorted(news, key=score, reverse=True)
        normalized = []
        for item in ranked:
            url = (item.get("url") or "").strip()
            title = (item.get("headline") or "").strip()
            if not url or not title:
                continue
            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "site": (item.get("source") or "Finnhub").strip(),
                    "summary": (item.get("summary") or "").strip(),
                    "datetime": item.get("datetime"),
                    "category": item.get("category"),
                    "relevance_score": round(score(item), 2),
                }
            )
            if len(normalized) >= limit:
                break
        return normalized

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

    def fetch_financial_reports(self, ticker: str, freq: str = "annual") -> List[Dict[str, Any]]:
        """Fetch GAAP financial reports from Finnhub (income/balance/cash sections)."""
        if not self.api_key:
            return []

        payload = self._safe_get(
            "stock/financials-reported",
            {"symbol": ticker, "freq": freq},
        )
        return payload.get("data", []) if isinstance(payload, dict) else []

    def _extract_concept_value(self, concepts: List[Dict[str, Any]], candidates: List[str]) -> Optional[float]:
        """Find first concept value matching candidate keys."""
        if not concepts:
            return None

        normalized_candidates = {c.lower() for c in candidates}
        for item in concepts:
            concept = str(item.get("concept", "")).lower()
            if concept in normalized_candidates or concept.split("_")[-1] in normalized_candidates:
                value = item.get("value")
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    continue
        return None

    def fetch_income_statement(self, ticker: str, freq: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch and normalize Finnhub income statement fields expected by downstream code."""
        reports = self.fetch_financial_reports(ticker, freq=freq)
        normalized: List[Dict[str, Any]] = []

        for report in reports[:limit]:
            ic = report.get("report", {}).get("ic", [])
            normalized.append(
                {
                    "date": report.get("endDate"),
                    "revenue": self._extract_concept_value(ic, ["revenues", "salesrevenuegoodsnet"]),
                    "netIncome": self._extract_concept_value(ic, ["netincomeloss", "profitloss"]),
                    "grossProfitRatio": self._extract_concept_value(ic, ["grossprofitratio", "grossprofit"]),
                    "interestExpense": self._extract_concept_value(ic, ["interestexpense"]),
                    "incomeTaxExpense": self._extract_concept_value(ic, ["incometaxexpensebenefit"]),
                    "depreciationAndAmortization": self._extract_concept_value(
                        ic, ["depreciationdepletionandamortization", "depreciationamortizationandaccretionnet"]
                    ),
                }
            )

        return normalized

    def fetch_balance_sheet(self, ticker: str, freq: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch and normalize Finnhub balance sheet fields expected by downstream code."""
        reports = self.fetch_financial_reports(ticker, freq=freq)
        normalized: List[Dict[str, Any]] = []

        for report in reports[:limit]:
            bs = report.get("report", {}).get("bs", [])
            normalized.append(
                {
                    "date": report.get("endDate"),
                    "totalAssets": self._extract_concept_value(bs, ["assets"]),
                    "totalDebt": self._extract_concept_value(
                        bs,
                        [
                            "debtandfinanceleaseobligations",
                            "longtermdebtandcapitalleaseobligations",
                            "longtermdebt",
                        ],
                    ),
                    "shortTermDebt": self._extract_concept_value(bs, ["shorttermborrowings", "commercialpaper"]),
                    "longTermDebt": self._extract_concept_value(
                        bs,
                        ["longtermdebtnoncurrent", "longtermdebtandcapitalleaseobligations"],
                    ),
                }
            )

        return normalized

    def fetch_cash_flow(self, ticker: str, freq: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch and normalize Finnhub cash flow fields expected by downstream code."""
        reports = self.fetch_financial_reports(ticker, freq=freq)
        normalized: List[Dict[str, Any]] = []

        for report in reports[:limit]:
            cf = report.get("report", {}).get("cf", [])
            normalized.append(
                {
                    "date": report.get("endDate"),
                    "operatingCashFlow": self._extract_concept_value(
                        cf, ["netcashprovidedbyusedinoperatingactivities", "netcashprovidedbyusedinoperatingactivitiescontinuingoperations"]
                    ),
                    "capitalExpenditure": self._extract_concept_value(
                        cf, ["paymentstoacquirepropertyplantandequipment", "capitalexpenditure"]
                    ),
                    "freeCashFlow": self._extract_concept_value(cf, ["freecashflow"]),
                }
            )

        return normalized

    def fetch_key_metrics(self, ticker: str) -> Dict[str, Any]:
        """Expose Finnhub basic metrics in a compact dict for adapter parity."""
        payload = self.fetch_basic_financials(ticker)
        return payload.get("metric", {}) if isinstance(payload, dict) else {}

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

    def fetch_earnings_calendar_standardized(self, days_forward: int = 7, limit: int = 20) -> List[Dict]:
        """Fetch, normalize, and deterministically sort upcoming earnings events."""
        earnings = self.fetch_earnings_calendar(days_forward=days_forward)
        if not earnings:
            return []

        def hour_rank(event_hour: str) -> int:
            hour = (event_hour or "").lower()
            if "bmo" in hour:
                return 0
            if "amc" in hour:
                return 2
            return 1

        normalized = []
        for event in earnings:
            symbol = (event.get("symbol") or "").strip()
            date = (event.get("date") or "").strip()
            if not symbol or not date:
                continue
            normalized.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "hour": (event.get("hour") or "").strip(),
                    "epsEstimate": event.get("epsEstimate"),
                    "revenueEstimate": event.get("revenueEstimate"),
                }
            )

        ranked = sorted(normalized, key=lambda e: (e.get("date", "9999-99-99"), hour_rank(e.get("hour", "")), e.get("symbol", "")))
        return ranked[:limit]

    def fetch_ipo_calendar(self, days_forward: int = 14, limit: int = 20) -> List[Dict]:
        """Fetch IPO/market calendar if available for the API plan."""
        cache_key = f"ipo_cal_{days_forward}_{limit}"
        cache_path = self._get_cache_path(cache_key)
        if self._is_cache_valid(cache_path, hours=12):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if not self.api_key:
            return []

        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_forward)).strftime("%Y-%m-%d")
        payload = self._safe_get("calendar/ipo", {"from": start_date, "to": end_date})
        events = payload.get("ipoCalendar", []) if isinstance(payload, dict) else []
        if not events:
            return []

        normalized = []
        for event in events:
            symbol = (event.get("symbol") or event.get("name") or "").strip()
            date = (event.get("date") or "").strip()
            if not symbol or not date:
                continue
            normalized.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "exchange": event.get("exchange", ""),
                    "price": event.get("price", ""),
                    "shares": event.get("numberOfShares", ""),
                }
            )

        normalized = sorted(normalized, key=lambda x: (x.get("date", "9999-99-99"), x.get("symbol", "")))[:limit]
        if normalized:
            with open(cache_path, "wb") as f:
                pickle.dump(normalized, f)
        return normalized

    def fetch_ticker_sentiment_snapshot(self, ticker: str) -> Dict:
        """Fetch ticker-level sentiment snapshot from Finnhub news sentiment endpoint."""
        symbol = (ticker or "").upper().strip()
        if not symbol:
            return {}

        cache_key = f"sentiment_{symbol}"
        cache_path = self._get_cache_path(cache_key)
        if self._is_cache_valid(cache_path, hours=2):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        payload = self._safe_get("news-sentiment", {"symbol": symbol})
        if not payload:
            return {}

        sentiment = {
            "ticker": symbol,
            "buzz": float(payload.get("buzz", {}).get("buzz", 0.0) or 0.0),
            "weekly_average_buzz": float(payload.get("buzz", {}).get("weeklyAverage", 0.0) or 0.0),
            "bullish_pct": float(payload.get("sentiment", {}).get("bullishPercent", 0.0) or 0.0),
            "bearish_pct": float(payload.get("sentiment", {}).get("bearishPercent", 0.0) or 0.0),
            "sector_average_bullish_pct": float(payload.get("sectorAverageBullishPercent", 0.0) or 0.0),
            "company_news_score": float(payload.get("companyNewsScore", 0.0) or 0.0),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(sentiment, f)
        return sentiment

    def fetch_top_ticker_sentiment_snapshots(
        self,
        top_buy_tickers: Optional[List[str]] = None,
        top_sell_tickers: Optional[List[str]] = None,
        per_side: int = 3,
    ) -> Dict[str, List[Dict]]:
        """Fetch sentiment snapshots for top buy/sell baskets."""
        result = {"top_buys": [], "top_sells": []}
        for key, tickers in (("top_buys", top_buy_tickers or []), ("top_sells", top_sell_tickers or [])):
            for ticker in tickers[:per_side]:
                snapshot = self.fetch_ticker_sentiment_snapshot(ticker)
                if snapshot:
                    result[key].append(snapshot)
        return result

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
