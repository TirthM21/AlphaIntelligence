#!/usr/bin/env python3
"""Local position fetcher using YFinance for prices.

Replaces Robinhood integration. Reads positions from data/positions.json.
This module simulates the interface of the previous RobinhoodPositionFetcher
but uses local data and YFinance for prices.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yfinance as yf

# For compatibility with legacy imports
ROBINHOOD_AVAILABLE = True 

logger = logging.getLogger(__name__)

class YFinancePositionFetcher:
    """Fetch positions from local file and update with YFinance prices."""

    def __init__(self, positions_file: str = "data/positions.json"):
        """Initialize fetcher.
        
        Args:
            positions_file: Path to JSON file containing positions.
                            Format: [{"ticker": "AAPL", "quantity": 10, "average_buy_price": 150.0}, ...]
        """
        self.positions_file = Path(positions_file)
        self.logged_in = True  # Always "logged in" for local file

    def login(self, password: Optional[str] = None, mfa_code: Optional[str] = None) -> bool:
        """Dummy login for compatibility. Always succeeds."""
        logger.info("Using local portfolio (YFinance) - Login successful")
        return True

    def logout(self):
        """Dummy logout for compatibility."""
        pass

    def fetch_positions(self) -> List[Dict]:
        """Read positions from file and fetch current prices via yfinance."""
        if not self.positions_file.exists():
            logger.warning(f"Positions file not found: {self.positions_file}")
            logger.warning("Please create data/positions.json with your portfolio data.")
            return []

        try:
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)
            
            if not positions_data:
                logger.info("Positions file is empty")
                return []

            logger.info("Fetching current prices from YFinance...")
            result = []
            
            for pos in positions_data:
                ticker = pos.get('ticker')
                qty = float(pos.get('quantity', 0))
                avg_buy = float(pos.get('average_buy_price', 0))
                
                if qty <= 0:
                    continue

                try:
                    # Fetch current price and previous close using fast_info
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Use fast_info for speed
                    current_price = ticker_obj.fast_info.last_price
                    prev_close = ticker_obj.fast_info.previous_close
                    
                    # Fallback if fast_info fails or returns None
                    if current_price is None or current_price == 0:
                         hist = ticker_obj.history(period='2d')
                         if not hist.empty:
                             current_price = float(hist['Close'].iloc[-1])
                             if len(hist) > 1:
                                 prev_close = float(hist['Close'].iloc[-2])
                             else:
                                 prev_close = current_price # Fallback
                         else:
                             current_price = 0.0
                             prev_close = 0.0

                    # Calculate Unrealized P/L
                    market_value = current_price * qty
                    cost_basis = avg_buy * qty
                    
                    if cost_basis > 0:
                        unrealized_pl = market_value - cost_basis
                        unrealized_pl_pct = (unrealized_pl / cost_basis) * 100
                    else:
                        unrealized_pl = 0.0
                        unrealized_pl_pct = 0.0

                    # Calculate Day's P/L
                    if prev_close > 0 and current_price > 0:
                        day_change = current_price - prev_close
                        day_change_pct = (day_change / prev_close) * 100
                        day_pl = day_change * qty
                    else:
                        day_change = 0.0
                        day_change_pct = 0.0
                        day_pl = 0.0

                    result.append({
                        'ticker': ticker,
                        'quantity': qty,  
                        'average_buy_price': round(avg_buy, 2),
                        'current_price': round(current_price, 2),
                        'previous_close': round(prev_close, 2),
                        'market_value': round(market_value, 2),
                        'cost_basis': round(cost_basis, 2),
                        'unrealized_pl': round(unrealized_pl, 2),
                        'unrealized_pl_pct': round(unrealized_pl_pct, 2),
                        'day_change': round(day_change, 2),
                        'day_change_pct': round(day_change_pct, 2),
                        'day_pl': round(day_pl, 2)
                    })
                    
                except Exception as e:
                    logger.error(f"Error fetching price for {ticker}: {e}")
                    # Include with 0 price so user sees it exists but arguably failed
                    result.append({
                        'ticker': ticker,
                        'quantity': qty,
                        'average_buy_price': round(avg_buy, 2),
                        'current_price': 0.0,
                        'market_value': 0.0,
                        'unrealized_pl_pct': 0.0,
                        'error': str(e)
                    })

            logger.info(f"✓ Fetched prices for {len(result)} positions")
            return result

        except Exception as e:
            logger.error(f"Error reading positions from {self.positions_file}: {e}")
            return []

    def get_position_tickers(self) -> List[str]:
        """Get just the tickers of current positions."""
        positions = self.fetch_positions()
        return [p['ticker'] for p in positions]

    def format_positions_report(self, positions: List[Dict] = None) -> str:
        """Format positions as a readable text report with ANSI colors.
        
        Args:
            positions: Optional list of positions to format. If None, fetches fresh.
        """
        if positions is None:
            positions = self.fetch_positions()

        if not positions:
            if not self.positions_file.exists():
                return f"No open positions found. (File {self.positions_file} missing)"
            return "No open positions found."

        # ANSI Colors
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        lines = []
        lines.append("="*90)
        lines.append(f"{BOLD}CURRENT PORTFOLIO POSITIONS (Source: Local File + YFinance){RESET}")
        lines.append(f"Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*90)
        lines.append(f"{'TICKER':<8} {'SHARES':<8} {'AVG PRICE':<12} {'PRICE':<10} {'DAY CHG %':<12} {'TOTAL P/L':<15} {'VALUE':<12}")
        lines.append("-" * 90)

        total_value = 0.0
        total_cost = 0.0
        total_day_pl = 0.0

        for pos in positions:
            ticker = pos.get('ticker', 'UNKNOWN')
            qty = pos.get('quantity', 0)
            avg_price = pos.get('average_buy_price', 0)
            curr_price = pos.get('current_price', 0)
            day_chg_pct = pos.get('day_change_pct', 0)
            unrealized_pl = pos.get('unrealized_pl', 0)
            unrealized_pl_pct = pos.get('unrealized_pl_pct', 0)
            market_value = pos.get('market_value', 0)
            day_pl = pos.get('day_pl', 0)

            total_value += market_value
            total_cost += pos.get('cost_basis', 0)
            total_day_pl += day_pl

            # Color coding
            day_color = GREEN if day_chg_pct >= 0 else RED
            pl_color = GREEN if unrealized_pl >= 0 else RED
            
            day_chg_str = f"{day_color}{day_chg_pct:+.2f}%{RESET}"
            pl_str = f"{pl_color}${unrealized_pl:,.2f} ({unrealized_pl_pct:+.2f}%){RESET}"

            lines.append(
                f"{ticker:<8} "
                f"{qty:<8} "
                f"${avg_price:<11.2f} "
                f"${curr_price:<9.2f} "
                f"{day_chg_str:<21} "  # Extra width for ANSI codes
                f"{pl_str:<24} "      # Extra width for ANSI codes
                f"${market_value:,.2f}"
            )

        lines.append("-" * 90)
        
        # Portfolio Summary
        total_pl = total_value - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
        total_color = GREEN if total_pl >= 0 else RED
        day_total_color = GREEN if total_day_pl >= 0 else RED
        
        lines.append(f"{BOLD}PORTFOLIO SUMMARY{RESET}")
        lines.append(f"Total Value:    ${total_value:,.2f}")
        lines.append(f"Total Cost:     ${total_cost:,.2f}")
        lines.append(f"Total P/L:      {total_color}${total_pl:,.2f} ({total_pl_pct:+.2f}%){RESET}")
        lines.append(f"Day's P/L:      {day_total_color}${total_day_pl:,.2f}{RESET}")
        lines.append("="*90)

        return "\n".join(lines)

def main():
    """Example usage."""
    import sys
    
    fetcher = YFinancePositionFetcher()
    print(fetcher.format_positions_report())

if __name__ == '__main__':
    main()
