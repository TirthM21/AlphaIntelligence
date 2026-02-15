"""AlphaIntelligence Capital — Performance Tracker.

Tracks hedge fund performance by:
1. Opening positions when BUY signals fire
2. Closing positions when SELL signals fire or stop-loss/SMA violations occur
3. Computing fund-level metrics: P&L, win rate, Sharpe, drawdown, alpha vs SPY
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf

from ..database.db_manager import DBManager

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks and reports fund performance based on daily/quarterly signals."""

    def __init__(self, strategy: str = 'DAILY'):
        """Initialize tracker.
        
        Args:
            strategy: 'DAILY' for short-term momentum, 'QUARTERLY' for compounders
        """
        self.strategy = strategy
        self.db = DBManager()
        self._spy_price = None

    @property
    def spy_price(self) -> float:
        """Get current SPY price (cached per session)."""
        if self._spy_price is None:
            try:
                spy = yf.Ticker("SPY")
                hist = spy.history(period="2d")
                if not hist.empty:
                    self._spy_price = hist['Close'].iloc[-1]
                else:
                    self._spy_price = 0.0
            except Exception as e:
                logger.warning(f"Could not fetch SPY price: {e}")
                self._spy_price = 0.0
        return self._spy_price

    def process_signals(self, buy_signals: List[Dict], sell_signals: List[Dict],
                        spy_price: float = None):
        """Process buy/sell signals from a scan run.
        
        - Opens positions for new BUY signals
        - Closes positions for SELL signals that match open positions
        
        Args:
            buy_signals: List of buy signal dicts from the scanner
            sell_signals: List of sell signal dicts from the scanner
            spy_price: Current SPY price for benchmarking
        """
        if spy_price:
            self._spy_price = spy_price

        opened = 0
        closed = 0

        # Process SELL signals first (close positions)
        sell_tickers = set()
        for signal in sell_signals:
            ticker = signal.get('ticker', '')
            if not ticker:
                continue
            sell_tickers.add(ticker)
            
            current_price = signal.get('current_price', 0)
            if current_price > 0:
                result = self.db.close_position(
                    ticker=ticker,
                    exit_price=current_price,
                    exit_reason='SELL_SIGNAL',
                    strategy=self.strategy,
                    spy_price=self.spy_price
                )
                if result:
                    closed += 1

        # Process BUY signals (open positions)
        for signal in buy_signals:
            ticker = signal.get('ticker', '')
            if not ticker or ticker in sell_tickers:
                continue

            current_price = signal.get('current_price', 0)
            stop_loss = signal.get('stop_loss')
            score = signal.get('score', 0)

            if current_price > 0:
                success = self.db.open_position(
                    ticker=ticker,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    signal_score=score,
                    strategy=self.strategy,
                    spy_price=self.spy_price
                )
                if success:
                    opened += 1

        logger.info(f"📊 Signal processing complete: {opened} opened, {closed} closed ({self.strategy})")

    def check_stop_losses(self) -> List[Dict]:
        """Check all open positions for stop-loss violations.
        
        Fetches current prices and closes positions that hit their stop loss.
        
        Returns:
            List of closed position dicts
        """
        open_positions = self.db.get_open_positions(strategy=self.strategy)
        if not open_positions:
            return []

        closed = []
        tickers = [p['ticker'] for p in open_positions]

        # Batch fetch current prices
        current_prices = self._batch_fetch_prices(tickers)

        for pos in open_positions:
            ticker = pos['ticker']
            current_price = current_prices.get(ticker)
            if not current_price or current_price <= 0:
                continue

            stop_loss = pos.get('stop_loss')

            # Check stop loss
            if stop_loss and current_price <= stop_loss:
                result = self.db.close_position(
                    ticker=ticker,
                    exit_price=current_price,
                    exit_reason='STOP_LOSS',
                    strategy=self.strategy,
                    spy_price=self.spy_price
                )
                if result:
                    closed.append(result)
                    logger.warning(f"🛑 Stop loss hit: {ticker} @ ${current_price:.2f} (stop: ${stop_loss:.2f})")

        if closed:
            logger.info(f"🛑 {len(closed)} position(s) closed via stop loss")
        return closed

    def compute_fund_metrics(self) -> Dict:
        """Compute comprehensive fund-level performance metrics.
        
        Returns:
            Dict with all fund metrics
        """
        open_positions = self.db.get_open_positions(strategy=self.strategy)
        closed_positions = self.db.get_closed_positions(strategy=self.strategy, limit=500)

        # Current prices for open positions
        if open_positions:
            tickers = [p['ticker'] for p in open_positions]
            current_prices = self._batch_fetch_prices(tickers)
        else:
            current_prices = {}

        # ---- Open position unrealized P&L ----
        unrealized_pnl = []
        for pos in open_positions:
            ticker = pos['ticker']
            current = current_prices.get(ticker)
            if current and pos['entry_price'] > 0:
                pnl = ((current - pos['entry_price']) / pos['entry_price']) * 100
                unrealized_pnl.append({
                    'ticker': ticker,
                    'entry': pos['entry_price'],
                    'current': current,
                    'pnl_pct': pnl,
                    'entry_date': pos['entry_date']
                })

        # ---- Closed position realized P&L ----
        realized_pnl = [p['pnl_pct'] for p in closed_positions if p.get('pnl_pct') is not None]
        
        wins = [p for p in realized_pnl if p > 0]
        losses = [p for p in realized_pnl if p <= 0]

        win_rate = (len(wins) / len(realized_pnl) * 100) if realized_pnl else 0
        avg_gain = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        total_realized_pnl = sum(realized_pnl) if realized_pnl else 0
        total_unrealized_pnl = sum(p['pnl_pct'] for p in unrealized_pnl) if unrealized_pnl else 0

        # Best/worst trades
        best_trade = None
        worst_trade = None
        if closed_positions:
            best = max(closed_positions, key=lambda x: x.get('pnl_pct', 0), default=None)
            worst = min(closed_positions, key=lambda x: x.get('pnl_pct', 0), default=None)
            if best:
                best_trade = f"{best['ticker']} ({best['pnl_pct']:+.1f}%)"
            if worst:
                worst_trade = f"{worst['ticker']} ({worst['pnl_pct']:+.1f}%)"

        # Sharpe ratio (simplified — daily returns from closed trades)
        sharpe = self._compute_sharpe(realized_pnl)

        # Max drawdown
        max_dd = self._compute_max_drawdown(realized_pnl)

        # Alpha vs SPY
        alpha = self._compute_alpha(closed_positions)

        # SPY return (over same period as our trades)
        spy_return = self._compute_spy_return(closed_positions)

        metrics = {
            'strategy': self.strategy,
            'total_pnl_pct': round(total_realized_pnl + total_unrealized_pnl, 2),
            'realized_pnl_pct': round(total_realized_pnl, 2),
            'unrealized_pnl_pct': round(total_unrealized_pnl, 2),
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'win_rate': round(win_rate, 1),
            'avg_gain': round(avg_gain, 2),
            'avg_loss': round(avg_loss, 2),
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'sharpe_ratio': round(sharpe, 2) if sharpe else None,
            'max_drawdown': round(max_dd, 2) if max_dd else None,
            'alpha_vs_spy': round(alpha, 2),
            'spy_return': round(spy_return, 2),
            'open_position_details': unrealized_pnl,
            'total_trades': len(realized_pnl)
        }

        # Record to database
        self.db.record_daily_performance(metrics)

        return metrics

    def get_newsletter_section(self) -> str:
        """Generate a markdown section for the newsletter with fund performance.
        
        Returns:
            Markdown string with performance data
        """
        metrics = self.compute_fund_metrics()
        
        lines = []
        strategy_label = "Short-Term Alpha" if self.strategy == 'DAILY' else "Long-Term Compounder"
        lines.append(f"## 📊 Fund Performance — {strategy_label}")
        lines.append("")

        # Summary stats
        lines.append(f"| Metric | Value |")
        lines.append(f"|:---|:---|")
        lines.append(f"| **Total P&L** | {metrics['total_pnl_pct']:+.2f}% |")
        lines.append(f"| **Realized** | {metrics['realized_pnl_pct']:+.2f}% |")
        lines.append(f"| **Unrealized** | {metrics['unrealized_pnl_pct']:+.2f}% |")
        lines.append(f"| **Win Rate** | {metrics['win_rate']:.1f}% ({len([x for x in (self.db.get_closed_positions(self.strategy) or []) if x.get('pnl_pct',0)>0])}/{metrics['closed_positions']}) |")
        lines.append(f"| **Avg Win** | {metrics['avg_gain']:+.2f}% |")
        lines.append(f"| **Avg Loss** | {metrics['avg_loss']:+.2f}% |")
        if metrics.get('sharpe_ratio') is not None:
            lines.append(f"| **Sharpe Ratio** | {metrics['sharpe_ratio']:.2f} |")
        if metrics.get('max_drawdown') is not None:
            lines.append(f"| **Max Drawdown** | {metrics['max_drawdown']:.2f}% |")
        lines.append(f"| **Alpha vs SPY** | {metrics['alpha_vs_spy']:+.2f}% |")
        lines.append(f"| **SPY Return** | {metrics['spy_return']:+.2f}% |")
        if metrics.get('best_trade'):
            lines.append(f"| **Best Trade** | {metrics['best_trade']} |")
        if metrics.get('worst_trade'):
            lines.append(f"| **Worst Trade** | {metrics['worst_trade']} |")
        lines.append("")

        # Open positions table
        open_details = metrics.get('open_position_details', [])
        if open_details:
            lines.append(f"### 📈 Open Positions ({len(open_details)})")
            lines.append("| Ticker | Entry | Current | P&L | Days |")
            lines.append("|:---|:---|:---|:---|:---|")
            for pos in sorted(open_details, key=lambda x: x['pnl_pct'], reverse=True):
                days_held = (datetime.utcnow() - pos['entry_date']).days if pos.get('entry_date') else 0
                emoji = "🟢" if pos['pnl_pct'] > 0 else "🔴"
                lines.append(
                    f"| {emoji} {pos['ticker']} | ${pos['entry']:.2f} | ${pos['current']:.2f} | "
                    f"{pos['pnl_pct']:+.1f}% | {days_held}d |"
                )
            lines.append("")

        # Recent closed trades
        closed = self.db.get_closed_positions(strategy=self.strategy, limit=5)
        if closed:
            lines.append("### 🏁 Recent Closed Trades")
            lines.append("| Ticker | Entry | Exit | P&L | Reason | Days |")
            lines.append("|:---|:---|:---|:---|:---|:---|")
            for trade in closed:
                emoji = "💰" if trade['pnl_pct'] > 0 else "📉"
                lines.append(
                    f"| {emoji} {trade['ticker']} | ${trade['entry_price']:.2f} | "
                    f"${trade['exit_price']:.2f} | {trade['pnl_pct']:+.1f}% | "
                    f"{trade['exit_reason']} | {trade['hold_days']}d |"
                )
            lines.append("")

        return "\n".join(lines)

    # ==================== PRIVATE HELPERS ====================

    def _batch_fetch_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Batch fetch current prices via yfinance."""
        prices = {}
        if not tickers:
            return prices

        try:
            # yfinance batch download
            data = yf.download(tickers, period="2d", progress=False, threads=True)
            if data.empty:
                return prices

            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        close = data['Close']
                    else:
                        close = data['Close'][ticker] if ticker in data['Close'].columns else None
                    
                    if close is not None and not close.empty:
                        prices[ticker] = float(close.dropna().iloc[-1])
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Batch price fetch failed, trying individual: {e}")
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="2d")
                    if not hist.empty:
                        prices[ticker] = float(hist['Close'].iloc[-1])
                except Exception:
                    continue

        return prices

    def _compute_sharpe(self, returns: List[float], risk_free_rate: float = 0.05) -> Optional[float]:
        """Compute simplified Sharpe ratio from trade returns."""
        if len(returns) < 3:
            return None
        
        arr = np.array(returns)
        excess = arr - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess) == 0:
            return None
        
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    def _compute_max_drawdown(self, returns: List[float]) -> Optional[float]:
        """Compute max drawdown from sequential trade returns."""
        if len(returns) < 2:
            return None

        # Build equity curve from trade returns
        equity = [100]  # Start at 100
        for r in returns:
            equity.append(equity[-1] * (1 + r / 100))

        peak = equity[0]
        max_dd = 0
        for val in equity[1:]:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _compute_alpha(self, closed_positions: List[Dict]) -> float:
        """Compute alpha vs SPY from closed positions."""
        if not closed_positions:
            return 0.0

        alphas = []
        for pos in closed_positions:
            if (pos.get('spy_entry_price') and pos.get('spy_exit_price') 
                    and pos.get('pnl_pct') is not None
                    and pos['spy_entry_price'] > 0):
                spy_return = ((pos['spy_exit_price'] - pos['spy_entry_price']) 
                              / pos['spy_entry_price']) * 100
                alpha = pos['pnl_pct'] - spy_return
                alphas.append(alpha)

        return float(np.mean(alphas)) if alphas else 0.0

    def _compute_spy_return(self, closed_positions: List[Dict]) -> float:
        """Compute average SPY return over the same periods as our trades."""
        if not closed_positions:
            return 0.0

        spy_returns = []
        for pos in closed_positions:
            if (pos.get('spy_entry_price') and pos.get('spy_exit_price')
                    and pos['spy_entry_price'] > 0):
                spy_r = ((pos['spy_exit_price'] - pos['spy_entry_price'])
                         / pos['spy_entry_price']) * 100
                spy_returns.append(spy_r)

        return float(np.mean(spy_returns)) if spy_returns else 0.0
