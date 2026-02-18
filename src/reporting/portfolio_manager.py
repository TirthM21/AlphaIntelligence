"""Portfolio Management and Advanced Reporting Engine.

Features:
- Ownership report (TXT)
- Allocation CSV
- Rebalance actions
- Alpha tracking vs S&P 500
"""

import logging
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from ..database.db_manager import DBManager
from ..data.price_service import PriceService

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages portfolio construction, rebalancing, and performance reporting."""
    
    def __init__(self, 
                 portfolio_path: str = "./data/positions.json",
                 report_dir: str = "./data/reports"):
        self.portfolio_path = Path(portfolio_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.db = DBManager()
        self.price_service = PriceService()

    def generate_reports(self, buy_signals: List[Dict], sell_signals: List[Dict]):
        """Generate all required portfolio and allocation reports."""
        logger.info("Generating advanced portfolio reports...")
        
        # 1. Ownership Report (TXT)
        try:
            self._generate_ownership_report()
        except Exception as e:
            logger.error(f"Failed to generate ownership report: {e}")
        
        # 2. Allocation CSV
        try:
            self._generate_allocation_csv(buy_signals)
        except Exception as e:
            logger.error(f"Failed to generate allocation plan: {e}")
        
        # 3. Rebalance Actions
        try:
            self._generate_rebalance_actions(sell_signals)
        except Exception as e:
            logger.error(f"Failed to generate rebalance actions: {e}")
        
        # 4. Alpha Performance Comparison
        try:
            self._generate_alpha_report()
        except Exception as e:
            logger.error(f"Failed to generate alpha tracker: {e}")

        # 5. Swing Trade Tracker (CSV for Google Sheets)
        try:
            self._generate_trade_tracker_csv(buy_signals)
        except Exception as e:
            logger.error(f"Failed to generate swing trade tracker: {e}")

    def _generate_ownership_report(self):
        """Creates a text file showing current ownership and holdings."""
        if not self.portfolio_path.exists():
            return
            
        with open(self.portfolio_path, 'r') as f:
            portfolio = json.load(f)
            
        report = []
        report.append("="*60)
        report.append(f"CURRENT PORTFOLIO OWNERSHIP REPORT - {datetime.now().strftime('%Y-%m-%d')}")
        report.append("="*60)
        
        total_value = 0
        for p in portfolio:
            cost_basis = p['quantity'] * p['average_buy_price']
            total_value += cost_basis
            report.append(f"Ticker: {p['ticker']:<6} | Qty: {p['quantity']:<5} | Avg Price: ${p['average_buy_price']:<8.2f} | Cost: ${cost_basis:,.2f}")
            
        report.append("-" * 60)
        report.append(f"TOTAL PORTFOLIO COST BASIS: ${total_value:,.2f}")
        report.append("="*60)
        
        output_file = self.report_dir / f"ownership_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        logger.info(f"Ownership report saved to {output_file}")

    def _generate_allocation_csv(self, buy_signals: List[Dict]):
        """Creates a CSV suggesting allocation for new buy signals."""
        allocations = []
        # Target: Spend $10,000 across top 5 signals
        total_budget = 10000
        budget_per_stock = total_budget / 5 if buy_signals else 0
        
        for s in buy_signals[:5]:
            ticker = s['ticker']
            is_valid, source = self.price_service.validate_price_payload_source(
                s,
                context=f"allocation signal {ticker}",
            )
            if not is_valid:
                logger.error("Rejecting allocation signal payload for %s due to blocked price source=%s", ticker, source)
                continue

            price = self.price_service.get_current_price(ticker) or 0
            if price <= 0:
                continue
                
            shares = int(budget_per_stock / price)
            
            allocations.append({
                'Ticker': ticker,
                'Signal_Score': s['score'],
                'Current_Price': price,
                'Recommended_Shares': shares,
                'Target_Weight': '20%',
                'Est_Cost': shares * price
            })
            
        df = pd.DataFrame(allocations)
        output_file = self.report_dir / f"allocation_plan_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Allocation plan saved to {output_file}")

    def _generate_rebalance_actions(self, sell_signals: List[Dict]):
        """Generates a text file with suggested exit/rebalance actions."""
        # Load portfolio to see if we own any of the sell signals
        if not self.portfolio_path.exists():
            return
            
        with open(self.portfolio_path, 'r') as f:
            portfolio = json.load(f)
            
        owned_tickers = {p['ticker']: p for p in portfolio}
        actions = []
        actions.append("="*60)
        actions.append(f"REQUIRED REBALANCE & EXIT ACTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        actions.append("="*60)
        
        found_action = False
        for s in sell_signals:
            if s['ticker'] in owned_tickers:
                found_action = True
                p = owned_tickers[s['ticker']]
                actions.append(f"🔴 EXIT: {s['ticker']}")
                actions.append(f"   Reason: {s['reasons'][0]}")
                actions.append(f"   Current Holding: {p['quantity']} shares")
                actions.append(f"   Action: Sell entire position at mark-open.")
                actions.append("-" * 30)
                
        if not found_action:
            actions.append("No active portfolio positions triggered a sell signal. Sit tight.")
            
        output_file = self.report_dir / f"rebalance_actions_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(output_file, 'w') as f:
            f.write("\n".join(actions))
        logger.info(f"Rebalance actions saved to {output_file}")

    def _generate_alpha_report(self):
        """Compares historical recommendations against S&P 500."""
        perf_data = self.db.get_recommendation_performance()
        if not perf_data:
            logger.info("No historical recommendations found for alpha report.")
            return

        report = []
        report.append("="*60)
        report.append(f"ALPHA TRACKER: RECOMMENDATIONS VS S&P 500")
        report.append("="*60)
        report.append(f"{'Ticker':<8} | {'Entry':<8} | {'Current':<8} | {'ROI %':<8} | {'Alpha vs SPY'}")
        report.append("-" * 60)

        # Get current SPY price
        current_spy = self.price_service.get_current_price("SPY") or 0
        if current_spy <= 0:
            logger.warning("Unable to fetch SPY price for alpha report")
            return

        total_alpha = 0
        count = 0

        for r in perf_data:
            try:
                current_price = self.price_service.get_current_price(r['ticker'])
                if not current_price or current_price <= 0:
                    continue
                
                stock_roi = (current_price - r['entry_price']) / r['entry_price'] * 100
                spy_roi = (current_spy - r['spy_entry']) / r['spy_entry'] * 100
                alpha = stock_roi - spy_roi
                
                total_alpha += alpha
                count += 1
                
                report.append(f"{r['ticker']:<8} | {r['entry_price']:<8.2f} | {current_price:<8.2f} | {stock_roi:>7.1f}% | {alpha:>+7.1f}%")
            except Exception as e:
                logger.debug(f"Failed to track {r['ticker']}: {e}")

        if count > 0:
            avg_alpha = total_alpha / count
            report.append("-" * 60)
            report.append(f"AVERAGE ALPHA ACROSS {count} SIGNALS: {avg_alpha:>+7.2f}%")
        
        output_file = self.report_dir / f"alpha_performance_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        logger.info(f"Alpha report saved to {output_file}")

    def _generate_trade_tracker_csv(self, buy_signals: List[Dict]):
        """Creates a CSV formatted specifically for the 'Simple Trade Tracker' spreadsheet."""
        tracker_rows = []
        date_str = datetime.now().strftime('%m/%d/%Y')
        
        # Risk settings
        risk_per_trade_dollars = 500.0
        
        for s in buy_signals[:10]: # Top 10 signals
            ticker = s['ticker']
            score = s['score']
            is_valid, source = self.price_service.validate_price_payload_source(
                s,
                context=f"trade tracker signal {ticker}",
            )
            if not is_valid:
                logger.error("Rejecting trade tracker signal payload for %s due to blocked price source=%s", ticker, source)
                continue

            entry_price = self.price_service.get_current_price(ticker) or 0
            stop_loss = s.get('stop_loss') or (entry_price * 0.93) # Default 7% stop if missing
            target_price = s.get('reward_target') or (entry_price * 1.20) # Default 20% target if missing
            rr_ratio = s.get('risk_reward_ratio') or 2.5
            
            if entry_price <= 0:
                continue
                
            # Calculation: Shares = $Risk / (Entry - Stop)
            risk_per_share = entry_price - stop_loss
            if risk_per_share > 0:
                shares = int(risk_per_trade_dollars / risk_per_share)
            else:
                shares = 0
                
            total_risk = shares * risk_per_share
            
            # AI Thesis (truncate for CSV)
            notes = s.get('reason', 'Strong buy signal detected')
            if 'fundamental_snapshot' in s:
                 notes = s['fundamental_snapshot'].replace('\n', ' ')[:200]
            
            tracker_rows.append({
                'Date': date_str,
                'Ticker': ticker,
                'Score': f"{score:.1f}",
                'Entry $': f"{entry_price:.2f}",
                'Stop $': f"{stop_loss:.2f}",
                'Target $': f"{target_price:.2f}",
                'Shares': shares,
                'Risk $': f"{total_risk:.2f}",
                'R/R': f"{rr_ratio:.1f}",
                'Exit Date': '',
                'Exit $': '',
                'P/L $': '',
                'Notes': notes
            })
            
        if not tracker_rows:
            return
            
        df = pd.DataFrame(tracker_rows)
        output_file = self.report_dir / f"trade_tracker_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Swing trade tracker CSV saved to {output_file} (Ready for Sheets import)")
