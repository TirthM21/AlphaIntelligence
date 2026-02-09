"""Enhanced fundamentals wrapper that intelligently uses FMP + yfinance.

This module provides a unified interface for fetching quarterly fundamentals:
- Tries FMP first (if API key available) for net margins, operating margins, detailed inventory
- Falls back to yfinance if FMP unavailable or rate limited
- Caches results to minimize API calls

Strategy:
- Use FMP for top buy candidates (detailed analysis)
- Use yfinance for initial screening (fast, no rate limits)
"""

import logging
import os
from typing import Dict, Optional

from .sec_fetcher import SECFetcher
from .fmp_fetcher import FMPFetcher
from .fundamentals_fetcher import (
    create_fundamental_snapshot, 
    analyze_fundamentals_for_signal,
    fetch_quarterly_financials
)

logger = logging.getLogger(__name__)

class EnhancedFundamentalsFetcher:
    """Unified fundamentals fetcher using FMP + yfinance + SEC Edgar."""

    def __init__(self):
        """Initialize fetcher with FMP if API key available."""
        self.fmp_available = False
        self.fmp_fetcher = None
        self.sec_fetcher = SECFetcher()

        # Check if FMP API key is available
        fmp_api_key = os.getenv('FMP_API_KEY')
        if fmp_api_key:
            try:
                self.fmp_fetcher = FMPFetcher(api_key=fmp_api_key)
                self.fmp_available = True
                logger.info("FMP available - will use for enhanced fundamentals")
            except Exception as e:
                logger.warning(f"FMP initialization failed: {e}. Using yfinance only.")
        else:
            logger.info("FMP_API_KEY not set - using yfinance only")

        self.fmp_call_count = 0
        self.fmp_daily_limit = 250

    def fetch_quarterly_data(
        self,
        ticker: str,
        use_fmp: bool = True
    ) -> Dict[str, any]:
        """Fetch quarterly financial data from FMP (primary) or yfinance (fallback)."""
        # 1. Try FMP First (if available and not at limit)
        if self.fmp_available and self.fmp_fetcher:
            if self.fmp_call_count < self.fmp_daily_limit:
                try:
                    logger.info(f"FMP [{ticker}]: Attempting fetch...")
                    data = self.fmp_fetcher.fetch_comprehensive_fundamentals(ticker, include_advanced=True)
                    self.fmp_call_count += 6
                    
                    if data and data.get('income_statement'):
                        logger.info(f"✅ FMP [{ticker}]: Data received (Net Margin: {data['income_statement'][0].get('netIncomeRatio', 0)*100:.1f}%)")
                        return self._convert_fmp_to_standard(data)
                    else:
                        logger.warning(f"⚠️ FMP [{ticker}]: Empty response, falling back to yfinance")
                except Exception as e:
                    logger.warning(f"❌ FMP [{ticker}]: Error {e}, falling back to yfinance")
            else:
                logger.warning(f"⏹️ FMP limit reached ({self.fmp_call_count}/{self.fmp_daily_limit})")

        # 2. Fall back to yfinance
        logger.info(f"🔄 YFINANCE [{ticker}]: Fetching standard fundamentals...")
        return fetch_quarterly_financials(ticker)

    def download_sec_filing(self, ticker: str, filing_type: str = '10-Q') -> str:
        """Download latest SEC filing."""
        if filing_type == '10-Q':
            return self.sec_fetcher.download_latest_10q(ticker)
        elif filing_type == '10-K':
            return self.sec_fetcher.download_latest_10k(ticker)
        return "Invalid filing type"

    def _convert_fmp_to_standard(self, fmp_data: Dict) -> Dict[str, any]:
        """Convert FMP data format to standard format used by signal engine."""
        if not fmp_data or not fmp_data.get('income_statement'):
            return {}

        income = fmp_data.get('income_statement', [])
        balance = fmp_data.get('balance_sheet', [])
        
        # Convert FMP data to standard format
        result = self._standard_conversion_logic(fmp_data)
        
        # Attach advanced data if available
        if 'dcf' in fmp_data:
            result['dcf'] = fmp_data['dcf']
        if 'insider_trading' in fmp_data:
            result['insider_trading'] = fmp_data['insider_trading']
            
        return result

    def _standard_conversion_logic(self, fmp_data: Dict) -> Dict[str, any]:
        """Internal helper to reuse the original conversion logic since I can't easily call super() or original method in this overwrite."""
        income = fmp_data.get('income_statement', [])
        balance = fmp_data.get('balance_sheet', [])

        if len(income) == 0: return {}

        result = {
            'ticker': fmp_data['ticker'],
            'fetch_date': fmp_data['fetch_date'],
            'data_source': 'fmp'
        }

        # Latest quarter
        latest_income = income[0]
        prev_income = income[1] if len(income) > 1 else {}
        latest_balance = balance[0] if len(balance) > 0 else {}
        prev_balance = balance[1] if len(balance) > 1 else {}

        # Revenue
        revenue = latest_income.get('revenue', 0)
        prev_revenue = prev_income.get('revenue', 0)

        if revenue:
            result['latest_revenue'] = revenue
            if prev_revenue:
                result['revenue_qoq_change'] = ((revenue - prev_revenue) / prev_revenue * 100)

        # YoY revenue (4 quarters ago)
        if len(income) >= 5:
            yoy_revenue = income[4].get('revenue', 0)
            if yoy_revenue:
                result['revenue_yoy_change'] = ((revenue - yoy_revenue) / yoy_revenue * 100)

        # EPS
        eps = latest_income.get('eps', 0)
        prev_eps = prev_income.get('eps', 0)

        if eps:
            result['latest_eps'] = eps
            if prev_eps and prev_eps != 0:
                result['eps_qoq_change'] = ((eps - prev_eps) / abs(prev_eps) * 100)

        # YoY EPS
        if len(income) >= 5:
            yoy_eps = income[4].get('eps', 0)
            if yoy_eps and yoy_eps != 0:
                result['eps_yoy_change'] = ((eps - yoy_eps) / abs(yoy_eps) * 100)

        # NET MARGIN (not available in yfinance!)
        net_margin = latest_income.get('netIncomeRatio', 0) * 100
        prev_net_margin = prev_income.get('netIncomeRatio', 0) * 100

        result['net_margin'] = round(net_margin, 2)
        result['net_margin_change'] = round(net_margin - prev_net_margin, 2)

        # OPERATING MARGIN (not available in yfinance!)
        operating_margin = latest_income.get('operatingIncomeRatio', 0) * 100
        result['operating_margin'] = round(operating_margin, 2)

        # Gross margin
        gross_margin = latest_income.get('grossProfitRatio', 0) * 100
        prev_gross_margin = prev_income.get('grossProfitRatio', 0) * 100

        result['gross_margin'] = round(gross_margin, 2)
        result['margin_change'] = round(gross_margin - prev_gross_margin, 2)

        # Inventory (detailed in FMP)
        inventory = latest_balance.get('inventory', 0)
        prev_inventory = prev_balance.get('inventory', 0)

        if inventory:
            result['latest_inventory'] = inventory
            if prev_inventory:
                inv_change = ((inventory - prev_inventory) / prev_inventory * 100)
                result['inventory_qoq_change'] = round(inv_change, 2)

            if revenue:
                result['inventory_to_sales_ratio'] = round(inventory / revenue, 3)

        return result

    def create_snapshot(
        self,
        ticker: str,
        quarterly_data: Optional[Dict] = None,
        use_fmp: bool = False
    ) -> str:
        """Create fundamental snapshot."""
        # Fetch data if not provided
        if quarterly_data is None:
            quarterly_data = self.fetch_quarterly_data(ticker, use_fmp=use_fmp)

        # If data came from FMP and has enhanced fields, use FMP snapshot
        if (quarterly_data.get('data_source') == 'fmp' and
            self.fmp_available and
            self.fmp_fetcher):
            # Re-fetch FMP data for enhanced snapshot (gets comprehensive dict)
            fmp_data = self.fmp_fetcher.fetch_comprehensive_fundamentals(ticker, include_advanced=True)
            if fmp_data:
                snapshot = self.fmp_fetcher.create_enhanced_snapshot(ticker, fmp_data)
                
                # Append DCF and Insider data if present
                dcf = fmp_data.get('dcf')
                insider = fmp_data.get('insider_trading', [])
                
                extra_info = []
                if dcf:
                    # Simple undervalued check
                    # We need current price which isn't passed here, but we can assume user checks it
                    extra_info.append(f"\nIntrinsic Value (DCF): ${dcf:.2f}")
                
                if insider:
                    # Check for recent buys
                    recent_buys = [t for t in insider if 'Buy' in t.get('transactionType', '') or 'P - Purchase' in t.get('transactionType', '')]
                    if recent_buys:
                        extra_info.append(f"Insider Activity: {len(recent_buys)} recent buys detected!")
                
                if extra_info:
                    snapshot += "\n" + "\n".join(extra_info)

                return snapshot

        # Fall back to standard snapshot
        return create_fundamental_snapshot(ticker, quarterly_data)

    def analyze_for_signal(
        self,
        ticker: str,
        quarterly_data: Optional[Dict] = None,
        use_fmp: bool = False
    ) -> Dict[str, any]:
        """Analyze fundamentals for signal engine.

        Args:
            ticker: Stock ticker
            quarterly_data: Pre-fetched data, or will fetch if None
            use_fmp: Use FMP for enhanced analysis if available

        Returns:
            Dict with trend analysis and penalty
        """
        if quarterly_data is None:
            quarterly_data = self.fetch_quarterly_data(ticker, use_fmp=use_fmp)

        return analyze_fundamentals_for_signal(quarterly_data)

    def get_api_usage(self) -> Dict[str, int]:
        """Get API usage statistics.

        Returns:
            Dict with FMP call count, limit, and bandwidth
        """
        usage = {
            'fmp_available': self.fmp_available,
            'fmp_calls_used': self.fmp_call_count,
            'fmp_daily_limit': self.fmp_daily_limit,
            'fmp_calls_remaining': max(0, self.fmp_daily_limit - self.fmp_call_count)
        }

        # Add bandwidth stats if FMP is available
        if self.fmp_available and self.fmp_fetcher:
            bandwidth_stats = self.fmp_fetcher.get_bandwidth_stats()
            usage.update(bandwidth_stats)

        return usage

    def reset_usage_counter(self):
        """Reset FMP usage counter (call at start of new day)."""
        self.fmp_call_count = 0
        logger.info("FMP usage counter reset")
