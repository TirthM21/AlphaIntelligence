"""Price service for authoritative market prices via yfinance.

This module intentionally centralizes *all* price retrieval behind yfinance.
FMP/Finnhub data may still be used for news/fundamentals/events/sentiment, but
must not be treated as authoritative for price values.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceService:
    """Fetches current and historical prices from yfinance only."""

    _ALLOWED_PRICE_SOURCES = {"yfinance", "yf", "yahoo", "yahoo_finance"}
    _BLOCKED_PRICE_SOURCES = {"fmp", "financialmodelingprep", "finnhub"}

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Return latest close/last price for ``ticker`` from yfinance."""
        if not ticker:
            return None

        try:
            stock = yf.Ticker(ticker)
            fast_info = getattr(stock, "fast_info", None)
            if fast_info:
                last_price = fast_info.get("lastPrice") or fast_info.get("last_price")
                if last_price and float(last_price) > 0:
                    return float(last_price)

            hist = stock.history(period="2d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].dropna().iloc[-1])
        except Exception as exc:
            logger.warning("PriceService failed to fetch current price for %s: %s", ticker, exc)

        return None

    def get_batch_current_prices(self, tickers: Iterable[str]) -> Dict[str, float]:
        """Return latest prices for tickers via yfinance batch call with fallback."""
        clean_tickers = [t for t in dict.fromkeys(tickers or []) if t]
        prices: Dict[str, float] = {}
        if not clean_tickers:
            return prices

        try:
            data = yf.download(clean_tickers, period="2d", progress=False, threads=True)
            if data is not None and not data.empty and "Close" in data:
                if len(clean_tickers) == 1:
                    close = data["Close"]
                    if close is not None and not close.empty:
                        prices[clean_tickers[0]] = float(close.dropna().iloc[-1])
                else:
                    for ticker in clean_tickers:
                        if ticker in data["Close"].columns:
                            close = data["Close"][ticker]
                            if close is not None and not close.empty:
                                prices[ticker] = float(close.dropna().iloc[-1])
        except Exception as exc:
            logger.warning("Batch yfinance download failed: %s", exc)

        # Fallback for misses
        for ticker in clean_tickers:
            if ticker not in prices:
                current = self.get_current_price(ticker)
                if current and current > 0:
                    prices[ticker] = current

        return prices

    def get_price_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Return historical OHLCV prices for ``ticker`` from yfinance."""
        if not ticker:
            return pd.DataFrame()
        try:
            return yf.Ticker(ticker).history(period=period, interval=interval)
        except Exception as exc:
            logger.warning("PriceService failed historical fetch for %s: %s", ticker, exc)
            return pd.DataFrame()

    def validate_price_payload_source(self, payload: Optional[dict], *, context: str = "") -> Tuple[bool, Optional[str]]:
        """Validate that a payload's declared price source is yfinance-compatible.

        Returns:
            (is_valid, detected_source)
        """
        if not payload:
            return True, None

        candidate = self._extract_price_source(payload)
        if not candidate:
            return True, None

        normalized = candidate.strip().lower()
        if normalized in self._BLOCKED_PRICE_SOURCES:
            logger.error(
                "Rejected non-yfinance price payload%s: source=%s payload_keys=%s",
                f" ({context})" if context else "",
                normalized,
                sorted(payload.keys()),
            )
            return False, normalized

        if normalized in self._ALLOWED_PRICE_SOURCES:
            return True, normalized

        # Unknown source declaration is allowed; caller may still fetch from yfinance directly.
        return True, normalized

    def _extract_price_source(self, payload: dict) -> Optional[str]:
        for key in ("price_source", "current_price_source", "price_provider", "priceDataSource"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val

        for key in ("data_source", "source", "provider"):
            val = payload.get(key)
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in self._ALLOWED_PRICE_SOURCES or lowered in self._BLOCKED_PRICE_SOURCES:
                    return lowered

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return self._extract_price_source(metadata)

        return None
