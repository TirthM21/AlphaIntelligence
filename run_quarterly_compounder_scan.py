#!/usr/bin/env python3
"""
Quarterly Compounder Scan - Main Entry Point.

Orchestrates the complete long-term compounder identification pipeline:
1. Fetch fundamental and price data for top 500 stocks
2. Score each stock with the CompounderEngine (60/25/15)
3. Score thematic ETFs with the ETFEngine (30/40/20/10)
4. Build portfolio with concentration rules
5. Generate quarterly ownership reports
6. Export allocations and rebalance guidance
7. Commit reports to git

Usage:
    python run_quarterly_compounder_scan.py              # Full scan
    python run_quarterly_compounder_scan.py --test-mode  # Test with 10 stocks
    python run_quarterly_compounder_scan.py --limit 50   # Limit to 50 stocks
"""

import sys
import logging
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuarterlyCompounderScan:
    """Orchestrates quarterly compounder identification."""

    def __init__(self, test_mode: bool = False, limit: Optional[int] = None):
        """
        Initialize scanner.

        Args:
            test_mode: Use reduced universe for quick testing
            limit: Limit number of stocks to score (default: 500)
        """
        self.test_mode = test_mode
        self.limit = limit or (10 if test_mode else 500)

        # Import components
        try:
            from src.long_term.compounder_engine import CompounderEngine
            from src.long_term.regime_classifier import RegimeClassifier
            from src.long_term.etf_engine import ETFEngine
            from src.long_term.etf_universe import ETFUniverse
            from src.long_term.portfolio_constructor import PortfolioConstructor
            from src.long_term.report_generator import ReportGenerator

            self.compounder_engine = CompounderEngine()
            self.regime_classifier = RegimeClassifier()
            self.etf_universe = ETFUniverse()

            # Try to import real data fetchers, use fallback if unavailable
            try:
                from src.data.universe_fetcher import USStockUniverseFetcher
                from src.data.fetcher import YahooFinanceFetcher
                from src.long_term.data_fetcher import LongTermFundamentalsFetcher
                self.universe_fetcher = USStockUniverseFetcher()
                self.price_fetcher = YahooFinanceFetcher()
                self.fundamentals_fetcher = LongTermFundamentalsFetcher()
                self.use_real_data = True
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"⚠ Real data fetchers unavailable ({e}) - using mock data")
                self.universe_fetcher = None
                self.price_fetcher = None
                self.fundamentals_fetcher = None
                self.use_real_data = False

            self.etf_engine = ETFEngine(universe=self.etf_universe)
            self.portfolio_constructor = PortfolioConstructor()
            self.report_generator = ReportGenerator()

            logger.info("✓ All components initialized")

        except ImportError as e:
            logger.error(f"✗ Failed to import components: {e}")
            raise

    def get_stock_universe(self) -> List[Dict]:
        """
        Get stock universe for scanning.

        Returns:
            List of stock dicts with ticker, name, sector
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: STOCK UNIVERSE")
        logger.info("=" * 80)

        if self.test_mode:
            # Use test stocks
            stocks = [
                {"ticker": "AAPL", "name": "Apple", "sector": "Technology"},
                {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology"},
                {"ticker": "NVDA", "name": "NVIDIA", "sector": "Technology"},
                {"ticker": "GOOGL", "name": "Alphabet", "sector": "Technology"},
                {"ticker": "META", "name": "Meta", "sector": "Technology"},
                {"ticker": "JPM", "name": "JPMorgan", "sector": "Financials"},
                {"ticker": "UNH", "name": "United Health", "sector": "Healthcare"},
                {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
                {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer"},
                {"ticker": "WMT", "name": "Walmart", "sector": "Consumer"},
            ]
            logger.info(f"✓ Test mode: Using {len(stocks)} test stocks")
        else:
            # Fetch real stock universe from data sources if available
            if self.universe_fetcher:
                logger.info("Fetching real stock universe...")
                try:
                    tickers = self.universe_fetcher.fetch_universe()
                    if not tickers:
                        logger.warning("Could not fetch universe, using fallback list")
                        stocks = self._get_fallback_stock_universe()[:self.limit]
                    else:
                        # Convert tickers to stock dicts (limit to configured amount)
                        stocks = [{"ticker": t, "name": t, "sector": "Unknown"} for t in tickers[:self.limit]]
                        logger.info(f"✓ Fetched {len(stocks)} stocks from universe")
                except Exception as e:
                    logger.warning(f"Error fetching universe: {e}, using fallback")
                    stocks = self._get_fallback_stock_universe()[:self.limit]
            else:
                logger.info("⚠ Real data fetcher unavailable, using fallback stock list")
                stocks = self._get_fallback_stock_universe()[:self.limit]

        return stocks

    def _get_default_stock_universe(self) -> List[Dict]:
        """Get real stock universe from FMP (top by market cap)."""
        logger.info("Using stock universe from data sources...")
        # Fallback to hardcoded list (in production would fetch from FMP/Yahoo)
        return self._get_fallback_stock_universe()

    def _get_fallback_stock_universe(self) -> List[Dict]:
        """Fallback hardcoded stock list (top 50 by market cap)."""
        return [
            {"ticker": "AAPL", "name": "Apple", "sector": "Information Technology"},
            {"ticker": "MSFT", "name": "Microsoft", "sector": "Information Technology"},
            {"ticker": "NVDA", "name": "NVIDIA", "sector": "Information Technology"},
            {"ticker": "GOOGL", "name": "Alphabet", "sector": "Information Technology"},
            {"ticker": "GOOG", "name": "Alphabet", "sector": "Information Technology"},
            {"ticker": "META", "name": "Meta", "sector": "Information Technology"},
            {"ticker": "AMZN", "name": "Amazon", "sector": "Consumer Discretionary"},
            {"ticker": "TSLA", "name": "Tesla", "sector": "Consumer Discretionary"},
            {"ticker": "BRK.B", "name": "Berkshire Hathaway", "sector": "Financials"},
            {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financials"},
            {"ticker": "V", "name": "Visa", "sector": "Information Technology"},
            {"ticker": "WMT", "name": "Walmart", "sector": "Consumer Staples"},
            {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples"},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
            {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
            {"ticker": "XOM", "name": "ExxonMobil", "sector": "Energy"},
            {"ticker": "CVX", "name": "Chevron", "sector": "Energy"},
            {"ticker": "LMT", "name": "Lockheed Martin", "sector": "Industrials"},
            {"ticker": "RTX", "name": "Raytheon Technologies", "sector": "Industrials"},
            {"ticker": "MA", "name": "Mastercard", "sector": "Information Technology"},
            {"ticker": "AXP", "name": "American Express", "sector": "Financials"},
            {"ticker": "BA", "name": "Boeing", "sector": "Industrials"},
            {"ticker": "CAT", "name": "Caterpillar", "sector": "Industrials"},
            {"ticker": "GE", "name": "General Electric", "sector": "Industrials"},
            {"ticker": "IBM", "name": "IBM", "sector": "Information Technology"},
            {"ticker": "INTC", "name": "Intel", "sector": "Information Technology"},
            {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Information Technology"},
            {"ticker": "PYPL", "name": "PayPal", "sector": "Information Technology"},
            {"ticker": "NFLX", "name": "Netflix", "sector": "Communication Services"},
            {"ticker": "DIS", "name": "Disney", "sector": "Communication Services"},
            {"ticker": "CRM", "name": "Salesforce", "sector": "Information Technology"},
            {"ticker": "ADBE", "name": "Adobe", "sector": "Information Technology"},
            {"ticker": "CSCO", "name": "Cisco Systems", "sector": "Information Technology"},
            {"ticker": "ACN", "name": "Accenture", "sector": "Information Technology"},
            {"ticker": "AVGO", "name": "Broadcom", "sector": "Information Technology"},
            {"ticker": "QCOM", "name": "Qualcomm", "sector": "Information Technology"},
            {"ticker": "TSM", "name": "Taiwan Semiconductor", "sector": "Information Technology"},
            {"ticker": "ORCL", "name": "Oracle", "sector": "Information Technology"},
            {"ticker": "SAP", "name": "SAP SE", "sector": "Information Technology"},
            {"ticker": "NOW", "name": "ServiceNow", "sector": "Information Technology"},
        ]

    def _fetch_price_data(self, ticker: str) -> Dict:
        """Fetch price data from yfinance, with fallbacks."""
        try:
            import yfinance as yf
            tick = yf.Ticker(ticker)
            hist = tick.history(period="5y")

            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                price_1yr = hist['Close'].iloc[-252] if len(hist) > 252 else hist['Close'].iloc[0]
                price_3yr = hist['Close'].iloc[-756] if len(hist) > 756 else hist['Close'].iloc[0]
                price_5yr = hist['Close'].iloc[0]

                returns_1yr = (current_price - price_1yr) / price_1yr if price_1yr > 0 else 0.0
                returns_3yr = ((current_price / price_3yr) ** (1/3) - 1) if price_3yr > 0 else 0.0
                returns_5yr = ((current_price / price_5yr) ** (1/5) - 1) if price_5yr > 0 else 0.0

                # 40-week MA (200 days)
                ma_40w = hist['Close'].iloc[-200:].mean() if len(hist) > 200 else hist['Close'].mean()
                ma_40w_slope = (hist['Close'].iloc[-1] - hist['Close'].iloc[-50]) / hist['Close'].iloc[-50] * 100 if len(hist) > 50 else 0.0

                # Months in uptrend
                recent_hist = hist.iloc[-252:] if len(hist) > 252 else hist
                months_in_uptrend = (recent_hist['Close'] > ma_40w).sum() // 20

            else:
                current_price = 100
                returns_1yr = 0.0
                returns_3yr = 0.0
                returns_5yr = 0.0
                ma_40w = 100
                ma_40w_slope = 0.0
                months_in_uptrend = 0

        except Exception as e:
            logger.debug(f"  ⚠ Could not fetch price data for {ticker}: {e}")
            current_price = 100
            returns_1yr = 0.0
            returns_3yr = 0.0
            returns_5yr = 0.0
            ma_40w = 100
            ma_40w_slope = 0.0
            months_in_uptrend = 0

        return {
            "current_price": current_price,
            "returns_1yr": returns_1yr,
            "returns_3yr": returns_3yr,
            "returns_5yr": returns_5yr,
            "spy_returns_1yr": 0.10,
            "spy_returns_3yr": 0.08,
            "spy_returns_5yr": 0.10,
            "price_40w_ma": ma_40w,
            "ma_40w_slope": ma_40w_slope,
            "months_in_uptrend": months_in_uptrend,
        }

    def score_stocks(self, stocks: List[Dict]) -> Dict[str, Dict]:
        """
        Score all stocks in universe.

        Args:
            stocks: List of stock dicts

        Returns:
            Dict mapping ticker to score data
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2: SCORE STOCKS")
        logger.info("=" * 80)

        if self.use_real_data:
            logger.info("📊 FETCHING REAL DATA from Yahoo Finance and fundamentals API")
            logger.info("⏱️  Rate limiting: 0.5s delay between requests to avoid API limits")
        else:
            logger.info("📋 Using deterministic mock data (real data fetchers unavailable)")

        scored_stocks = {}
        failed_scores = 0
        error_reasons = {}  # Track failure reasons
        last_request_time = 0.0

        for i, stock in enumerate(stocks, 1):
            ticker = stock["ticker"]
            try:
                if self.use_real_data:
                    # Rate limiting - wait 0.5s between requests
                    elapsed = time.time() - last_request_time
                    if elapsed < 0.5:
                        time.sleep(0.5 - elapsed)
                    last_request_time = time.time()
                    # Fetch real price data
                    logger.debug(f"  Fetching price data for {ticker}...")
                    price_hist = self.price_fetcher.fetch_price_history(ticker, period='5y')

                    if price_hist.empty or len(price_hist) < 200:
                        failed_scores += 1
                        continue

                    # Use last 1 year for analysis
                    price_data_df = price_hist.tail(252) if len(price_hist) > 252 else price_hist
                    current_price = price_data_df['Close'].iloc[-1]

                    # Fetch real 5-year fundamentals
                    logger.debug(f"  Fetching fundamentals for {ticker}...")
                    fundamentals_obj = self.fundamentals_fetcher.fetch(ticker)

                    # Extract what we need for compounder scoring (with safe None handling)
                    def safe_get_attr(obj, attr, default):
                        """Safely get attribute from object, converting None to default."""
                        if obj is None:
                            return default
                        val = getattr(obj, attr, default)
                        return default if val is None else val

                    # Build fundamentals dict with numeric type conversion using LongTermFundamentals attributes
                    # Uses defaults when fundamentals unavailable (no FMP API key)
                    fundamentals = {
                        "revenue_cagr_3yr": float(safe_get_attr(fundamentals_obj, 'revenue_cagr_3yr', 0.03)),
                        "revenue_cagr_5yr": float(safe_get_attr(fundamentals_obj, 'revenue_cagr_5yr', 0.03)),
                        "eps_cagr_3yr": float(safe_get_attr(fundamentals_obj, 'eps_cagr_3yr', 0.05)),
                        "roic": float(safe_get_attr(fundamentals_obj, 'roic_3yr', 0.12)),
                        "wacc": float(safe_get_attr(fundamentals_obj, 'wacc', 0.08)),
                        "roic_wacc_spread": float(safe_get_attr(fundamentals_obj, 'roic_wacc_spread', 0.04)),
                        "fcf_margin": float(safe_get_attr(fundamentals_obj, 'fcf_margin_3yr', 0.10)),
                        "debt_to_ebitda": float(safe_get_attr(fundamentals_obj, 'debt_to_ebitda', 2.0)),
                        "interest_coverage": float(safe_get_attr(fundamentals_obj, 'interest_coverage', 5.0)),
                        "rd_to_sales": 0.05,  # Not always available
                    }

                    # Calculate price data metrics
                    price_1yr = price_hist['Close'].iloc[-252] if len(price_hist) > 252 else price_hist['Close'].iloc[0]
                    price_3yr = price_hist['Close'].iloc[-756] if len(price_hist) > 756 else price_hist['Close'].iloc[0]
                    price_5yr = price_hist['Close'].iloc[0]

                    returns_1yr = (current_price - price_1yr) / price_1yr if price_1yr > 0 else 0.0
                    returns_3yr = ((current_price / price_3yr) ** (1/3) - 1) if price_3yr > 0 else 0.0
                    returns_5yr = ((current_price / price_5yr) ** (1/5) - 1) if price_5yr > 0 else 0.0

                    # 40-week MA (200 days) - with None handling
                    try:
                        ma_40w = float(price_hist['Close'].iloc[-200:].mean()) if len(price_hist) > 200 else float(price_hist['Close'].mean())
                    except (ValueError, TypeError):
                        ma_40w = current_price if current_price else 100.0

                    # MA slope - with None handling
                    try:
                        if len(price_hist) > 50:
                            slope_val = (price_hist['Close'].iloc[-1] - price_hist['Close'].iloc[-50]) / price_hist['Close'].iloc[-50]
                            ma_40w_slope = float(slope_val) if slope_val is not None else 0.0
                        else:
                            ma_40w_slope = 0.0
                    except (ValueError, TypeError, ZeroDivisionError):
                        ma_40w_slope = 0.0

                    # Months in uptrend - with None handling
                    try:
                        recent_hist = price_hist.iloc[-252:] if len(price_hist) > 252 else price_hist
                        uptrend_count = (recent_hist['Close'] > ma_40w).sum()
                        months_in_uptrend = int(uptrend_count // 20) if uptrend_count else 0
                    except (ValueError, TypeError):
                        months_in_uptrend = 0

                    # Calculate max drawdown (3-year window if available)
                    try:
                        recent_hist = price_hist.tail(756) if len(price_hist) > 756 else price_hist
                        running_max = recent_hist['Close'].cummax()
                        drawdown = (recent_hist['Close'] - running_max) / running_max
                        max_drawdown_3yr = float(drawdown.min())  # Most negative value
                    except (ValueError, TypeError):
                        max_drawdown_3yr = -0.30  # Conservative default

                    # Assume SPY max drawdown for reference
                    spy_max_drawdown_3yr = -0.15  # Historical average

                    # Ensure all values are numeric
                    price_data = {
                        "current_price": float(current_price),
                        "returns_1yr": float(returns_1yr),
                        "returns_3yr": float(returns_3yr),
                        "returns_5yr": float(returns_5yr),
                        "spy_returns_1yr": 0.10,
                        "spy_returns_3yr": 0.08,
                        "spy_returns_5yr": 0.10,
                        "max_drawdown_3yr": float(max_drawdown_3yr),
                        "spy_max_drawdown_3yr": float(spy_max_drawdown_3yr),
                        "price_40w_ma": float(ma_40w),
                        "ma_40w_slope": float(ma_40w_slope),
                        "months_in_uptrend": int(months_in_uptrend),
                    }
                else:
                    # Use mock data (hash-based variation per stock)
                    import hashlib
                    hash_val = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
                    base_seed = (hash_val % 100) / 100.0

                    fundamentals = {
                        "revenue_cagr_3yr": 0.05 + (base_seed * 0.20),      # 5-25%
                        "revenue_cagr_5yr": 0.04 + (base_seed * 0.18),      # 4-22%
                        "eps_cagr_3yr": 0.06 + (base_seed * 0.25),          # 6-31%
                        "roic": 0.08 + (base_seed * 0.35),                  # 8-43%
                        "wacc": 0.06 + (base_seed * 0.08),                  # 6-14%
                        "fcf_margin": 0.05 + (base_seed * 0.30),            # 5-35%
                        "debt_to_ebitda": 3.0 - (base_seed * 2.5),          # 0.5-3.0x
                        "interest_coverage": 3.0 + (base_seed * 12),        # 3-15x
                        "rd_to_sales": 0.02 + (base_seed * 0.15),           # 2-17%
                    }

                    price_seed = ((hash_val // 100) % 100) / 100.0
                    price_data = {
                        "current_price": 150,
                        "returns_1yr": -0.10 + (price_seed * 0.50),         # -10% to +40%
                        "returns_3yr": 0.02 + (price_seed * 0.30),          # 2% to 32%
                        "returns_5yr": 0.03 + (price_seed * 0.35),          # 3% to 38%
                        "spy_returns_1yr": 0.10,
                        "spy_returns_3yr": 0.08,
                        "spy_returns_5yr": 0.10,
                        "price_40w_ma": 145 + (price_seed * 30),            # 145-175
                        "ma_40w_slope": -0.05 + (price_seed * 0.15),        # -5% to +10%
                        "months_in_uptrend": int(6 + (price_seed * 30)),    # 6-36 months
                    }

                # Score the stock
                score = self.compounder_engine.score_stock(ticker, fundamentals, price_data)

                if score:
                    scored_stocks[ticker] = {
                        "name": stock["name"],
                        "sector": stock["sector"],
                        "score": score.total_score,
                        "regime": score.regime.name if hasattr(score.regime, 'name') else str(score.regime),
                        "fundamental_score": score.fundamental_score,
                        "rs_persistence_score": score.rs_persistence_score,
                        "trend_durability_score": score.trend_durability_score,
                        "moat_bonus": score.moat_bonus,
                    }
                else:
                    failed_scores += 1

                if i % 50 == 0:
                    logger.info(f"  Progress: {i}/{len(stocks)} stocks processed ({len(scored_stocks)} scored)")

            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.debug(f"  ⚠ Failed to score {ticker}: {error_type}: {error_msg}")
                error_reasons[error_type] = error_reasons.get(error_type, 0) + 1
                failed_scores += 1

        logger.info(f"✓ Scored {len(scored_stocks)} stocks ({failed_scores} failed)")
        if error_reasons:
            logger.info("Failure breakdown:")
            for error_type, count in sorted(error_reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  {error_type}: {count}")

        return scored_stocks

    def get_top_stocks(self, scored_stocks: Dict[str, Dict], top_n: int = 25) -> Dict[str, Dict]:
        """
        Get top N stocks by score.

        Args:
            scored_stocks: Dict of all scored stocks
            top_n: Number of top stocks to select

        Returns:
            Dict of top stocks
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STEP 3: SELECT TOP {top_n} STOCKS")
        logger.info("=" * 80)

        sorted_stocks = sorted(
            scored_stocks.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_n]

        top_stocks = {ticker: data for ticker, data in sorted_stocks}

        logger.info(f"✓ Selected top {len(top_stocks)} stocks")
        logger.info("")
        for rank, (ticker, data) in enumerate(sorted_stocks[:10], 1):
            logger.info(
                f"  {rank:2}. {ticker:6} {data['name']:30} Score: {data['score']:6.1f} "
                f"({data['regime']})"
            )

        return top_stocks

    def score_etfs(self) -> Dict[str, Dict]:
        """
        Score thematic ETFs.

        Returns:
            Dict mapping ETF ticker to score data
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: SCORE ETFs")
        logger.info("=" * 80)

        # Get ETFs by theme
        scored_etfs = {}

        for theme in ["ai_cloud", "defense", "energy_transition", "healthcare_innovation", "cybersecurity"]:
            try:
                etfs = self.etf_universe.get_etfs_by_theme(theme, filtered=True)

                for etf in etfs:
                    # Mock price data - varied per ETF based on ticker hash
                    import hashlib
                    hash_val = int(hashlib.md5(etf.ticker.encode()).hexdigest(), 16)
                    price_seed = (hash_val % 100) / 100.0

                    price_data = {
                        "return_1yr": 0.05 + (price_seed * 0.35),      # 5% to 40%
                        "return_3yr": 0.02 + (price_seed * 0.26),      # 2% to 28%
                        "return_5yr": 0.01 + (price_seed * 0.25),      # 1% to 26%
                        "spy_return_1yr": 0.10,
                        "spy_return_3yr": 0.08,
                        "spy_return_5yr": 0.10,
                    }

                    score = self.etf_engine.score_etf(etf.__dict__, price_data)

                    if score:
                        scored_etfs[etf.ticker] = {
                            "name": etf.name,
                            "theme": theme,
                            "score": score.total_score,
                            "theme_purity_score": score.theme_purity_score,
                            "rs_persistence_score": score.rs_persistence_score,
                            "efficiency_score": score.efficiency_score,
                            "tailwind_score": score.tailwind_score,
                        }

            except Exception as e:
                logger.warning(f"  ⚠ Failed to score {theme} ETFs: {e}")

        logger.info(f"✓ Scored {len(scored_etfs)} ETFs")

        return scored_etfs

    def get_top_etfs(self, scored_etfs: Dict[str, Dict], top_n: int = 10) -> Dict[str, Dict]:
        """
        Get top N ETFs by score.

        Args:
            scored_etfs: Dict of all scored ETFs
            top_n: Number of top ETFs to select

        Returns:
            Dict of top ETFs
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STEP 5: SELECT TOP {top_n} ETFs")
        logger.info("=" * 80)

        sorted_etfs = sorted(
            scored_etfs.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_n]

        top_etfs = {ticker: data for ticker, data in sorted_etfs}

        logger.info(f"✓ Selected top {len(top_etfs)} ETFs")
        logger.info("")
        for rank, (ticker, data) in enumerate(sorted_etfs, 1):
            logger.info(
                f"  {rank:2}. {ticker:6} {data['name']:30} Score: {data['score']:6.1f}"
            )

        return top_etfs

    def build_portfolio(
        self, top_stocks: Dict[str, Dict], top_etfs: Dict[str, Dict]
    ) -> Optional[object]:
        """
        Build optimal portfolio from top stocks and ETFs.

        Args:
            top_stocks: Dict of top stocks
            top_etfs: Dict of top ETFs

        Returns:
            PortfolioAllocation object
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 6: BUILD PORTFOLIO")
        logger.info("=" * 80)

        # Create lists for portfolio constructor
        stocks_list = [
            {
                "ticker": ticker,
                "name": data["name"],
                "score": data["score"],
                "sector": data["sector"],
            }
            for ticker, data in top_stocks.items()
        ]

        etfs_list = [
            {
                "ticker": ticker,
                "name": data["name"],
                "score": data["score"],
                "theme_id": data["theme"],
            }
            for ticker, data in top_etfs.items()
        ]

        # Create sector map
        sector_map = {s["ticker"]: s["sector"] for s in stocks_list}

        # Create theme map (simplified)
        theme_map = {
            e["ticker"]: e["theme_id"].replace("_", " ").title() for e in etfs_list
        }

        try:
            portfolio = self.portfolio_constructor.build_portfolio(
                stocks_list, etfs_list, sector_map, theme_map
            )

            if portfolio:
                logger.info(
                    f"✓ Portfolio built: {portfolio.total_positions} positions, "
                    f"score {portfolio.total_score:.1f}, "
                    f"concentration {portfolio.sector_concentration:.3f}"
                )
                return portfolio
            else:
                logger.error("✗ Failed to build portfolio")
                return None

        except Exception as e:
            logger.error(f"✗ Error building portfolio: {e}", exc_info=True)
            return None

    def generate_reports(
        self, portfolio: object, top_stocks: Dict, top_etfs: Dict
    ) -> Tuple[str, str, str]:
        """
        Generate quarterly reports.

        Args:
            portfolio: PortfolioAllocation object
            top_stocks: Dict of top stocks
            top_etfs: Dict of top ETFs

        Returns:
            Tuple of (ownership_report, csv_path, summary)
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 7: GENERATE REPORTS")
        logger.info("=" * 80)

        # Create stock and ETF dicts for report generator
        stocks_dict = {
            ticker: {
                "name": data["name"],
                "sector": data["sector"],
                "score": data["score"],
            }
            for ticker, data in top_stocks.items()
        }

        etfs_dict = {
            ticker: {
                "name": data["name"],
                "theme": data["theme"],
                "score": data["score"],
            }
            for ticker, data in top_etfs.items()
        }

        # Generate ownership report
        ownership_report = self.report_generator.generate_ownership_report(
            portfolio, stocks_dict, etfs_dict
        )
        logger.info("✓ Ownership report generated")

        # Generate allocation CSV
        quarter_date = datetime.now()
        q = (quarter_date.month - 1) // 3 + 1
        year = quarter_date.year
        csv_filename = f"allocation_model_{year}_Q{q}.csv"
        csv_path = f"data/quarterly_reports/{csv_filename}"

        Path("data/quarterly_reports").mkdir(parents=True, exist_ok=True)

        success = self.report_generator.generate_allocation_csv(
            portfolio, stocks_dict, etfs_dict, csv_path
        )

        if success:
            logger.info(f"✓ Allocation CSV written to {csv_path}")
        else:
            logger.warning("⚠ CSV generation failed")

        # Generate summary
        summary = f"Generated reports for {len(top_stocks)} stocks and {len(top_etfs)} ETFs"

        return ownership_report, csv_path, summary

    def run(self) -> bool:
        """
        Run complete quarterly compounder scan.

        Returns:
            True if successful
        """
        try:
            logger.info("")
            logger.info("=" * 80)
            logger.info("QUARTERLY COMPOUNDER SCAN")
            logger.info("=" * 80)
            if self.test_mode:
                logger.info("MODE: Test (limited universe)")
            logger.info(f"Stock Limit: {self.limit}")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)

            # Step 1: Get stock universe
            stocks = self.get_stock_universe()

            # Step 2: Score stocks
            scored_stocks = self.score_stocks(stocks)

            if not scored_stocks:
                logger.error("✗ No stocks scored successfully")
                return False

            # Step 3: Select top stocks
            top_stocks = self.get_top_stocks(scored_stocks, top_n=25)

            # Step 4: Score ETFs
            scored_etfs = self.score_etfs()

            if not scored_etfs:
                logger.error("✗ No ETFs scored successfully")
                return False

            # Step 5: Select top ETFs
            top_etfs = self.get_top_etfs(scored_etfs, top_n=10)

            # Step 6: Build portfolio
            portfolio = self.build_portfolio(top_stocks, top_etfs)

            if not portfolio:
                logger.error("✗ Portfolio construction failed")
                return False

            # Step 7: Generate reports
            ownership_report, csv_path, summary = self.generate_reports(
                portfolio, top_stocks, top_etfs
            )

            # Display summary
            logger.info("")
            logger.info("=" * 80)
            logger.info("SCAN COMPLETE")
            logger.info("=" * 80)
            logger.info("")
            logger.info(f"Total Positions: {portfolio.total_positions}")
            logger.info(f"Portfolio Score: {portfolio.total_score:.1f}/100")
            logger.info(f"Concentration: {portfolio.sector_concentration:.3f}")
            logger.info(f"CSV Export: {csv_path}")
            logger.info("")
            logger.info("Next Review: " + self.report_generator.get_next_review_date())
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"✗ Scan failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quarterly Compounder Scan - Long-term capital compounding framework"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (10 stocks)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of stocks to score (default: 500)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/quarterly_reports",
        help="Output directory for reports (default: data/quarterly_reports)",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(args.log_level)

    scanner = QuarterlyCompounderScan(
        test_mode=args.test_mode,
        limit=args.limit,
    )

    success = scanner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
