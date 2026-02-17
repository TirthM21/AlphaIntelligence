
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import yfinance as yf

from ..data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from ..data.finnhub_fetcher import FinnhubFetcher
from ..data.marketaux_fetcher import MarketauxFetcher
from ..data.fred_fetcher import FredFetcher
from ..ai.ai_agent import AIAgent
from .visualizer import MarketVisualizer

logger = logging.getLogger(__name__)

class NewsletterGenerator:
    """Generates a high-quality, professional daily market newsletter."""

    def __init__(self, portfolio_path: str = "./data/positions.json"):
        try:
            self.fetcher = EnhancedFundamentalsFetcher()
        except Exception as e:
            logger.warning(f"EnhancedFundamentalsFetcher unavailable during init: {e}")
            self.fetcher = None
        self.finnhub = FinnhubFetcher()
        self.marketaux = MarketauxFetcher()
        self.fred = FredFetcher()
        self.ai_agent = AIAgent()
        self.visualizer = MarketVisualizer()
        self.portfolio_path = Path(portfolio_path)

    def load_portfolio(self) -> List[Dict]:
        """Load user portfolio from positions.json."""
        if not self.portfolio_path.exists():
            return []
        try:
            with open(self.portfolio_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return []

    def generate_newsletter(self, 
                          market_status: Dict = None, 
                          top_buys: List[Dict] = None,
                          top_sells: List[Dict] = None,
                          fund_performance_md: str = "",
                          output_path: Optional[str] = None) -> str:
        """Generate the comprehensive professional daily newsletter."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path("./data/newsletters")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"daily_newsletter_{timestamp}.md")
            
        logger.info(f"Generating professional daily newsletter to {output_path}...")
        
        # 1. Initialize data containers
        econ_data = {}
        market_news = []
        portfolio = self.load_portfolio()
        portfolio_tickers = [p['ticker'] for p in portfolio]
        portfolio_news = []
        econ_calendar = []
        trending_entities = []
        earnings_cal = []

        def _clean_news_item(item: Dict) -> Optional[Dict]:
            title = (item.get('title') or '').strip()
            url = (item.get('url') or '').strip()
            if not title or not url:
                return None
            return {
                'title': title,
                'url': url,
                'site': (item.get('site') or 'News').strip(),
                'summary': (item.get('summary') or '').strip()
            }

        def _dedupe_news(items: List[Dict], limit: int = 10) -> List[Dict]:
            seen = set()
            output = []
            for raw in items:
                cleaned = _clean_news_item(raw)
                if not cleaned:
                    continue
                key = (cleaned['title'].lower(), cleaned['url'])
                if key in seen:
                    continue
                seen.add(key)
                output.append(cleaned)
                if len(output) >= limit:
                    break
            return output
        
        # 2. Try Finnhub & Marketaux NEWS first (Higher quality)
        if self.finnhub.api_key:
            try:
                logger.info("Fetching Finnhub market news & calendars...")
                finnhub_news = self.finnhub.fetch_market_news(category='general')
                for item in finnhub_news[:6]:
                    market_news.append({
                        'title': item.get('headline'),
                        'url': item.get('url'),
                        'site': item.get('source', 'Finnhub'),
                        'summary': item.get('summary', '')
                    })
                
                earnings_cal = self.finnhub.fetch_earnings_calendar(days_forward=5)
                econ_calendar = self.finnhub.fetch_economic_calendar() # Might return [] if not premium
                
                if portfolio_tickers:
                    for t in portfolio_tickers[:3]:
                        p_news = self.finnhub.fetch_company_news(t)
                        for item in p_news[:2]:
                            portfolio_news.append({
                                'title': item.get('headline'),
                                'symbol': t,
                                'url': item.get('url'),
                                'summary': item.get('summary', '')
                            })
            except Exception as e:
                logger.error(f"Finnhub news fetch failed: {e}")

        # Try Marketaux for additional context & trending
        if self.marketaux.api_key:
            try:
                logger.info("Fetching Marketaux trending entities...")
                trending_entities = self.marketaux.fetch_trending_entities()
                
                # If news is still thin, add from Marketaux
                if len(market_news) < 3:
                    ma_news = self.marketaux.fetch_market_news(limit=5)
                    for item in ma_news:
                        market_news.append({
                            'title': item.get('title'),
                            'url': item.get('url'),
                            'site': item.get('source', 'Marketaux'),
                            'summary': item.get('snippet', '')
                        })
            except Exception as e:
                logger.error(f"Marketaux fetch failed: {e}")

        # Try FRED for Macro Data (Highest priority for macro)
        if self.fred.api_key:
            try:
                logger.info("Fetching FRED macro economic indicators...")
                macro_indicators = {
                    'GDP': 'GDP',
                    'CPI': 'CPIAUCSL',
                    'Unemployment': 'UNRATE',
                    'Fed Funds': 'FEDFUNDS'
                }
                for name, series_id in macro_indicators.items():
                    obs = self.fred.fetch_series_observations(series_id, limit=2)
                    if obs:
                        latest = obs[-1]
                        prev = obs[-2] if len(obs) > 1 else {}
                        val = latest.get('value', '0')
                        p_val = prev.get('value', '0')
                        
                        try:
                            val_f = float(val) if val != '.' else 0
                            p_val_f = float(p_val) if p_val != '.' else 0
                            trend = "Up" if val_f > p_val_f else "Down"
                        except (TypeError, ValueError):
                            trend = "Stable"
                            
                        econ_data[name] = {
                            'current': val,
                            'date': latest.get('date'),
                            'previous': p_val,
                            'trend': trend
                        }
            except Exception as e:
                logger.error(f"FRED fetch failed: {e}")

        # Fallback/Supplemental Data from FMP
        if self.fetcher and self.fetcher.fmp_available and self.fetcher.fmp_fetcher:
            try:
                # FMP used for DAILY news as requested
                logger.info("Fetching FMP daily market news...")
                market_news_fmp = self.fetcher.fmp_fetcher.fetch_market_news(limit=10)
                if market_news_fmp:
                    for item in market_news_fmp:
                        market_news.append({
                            'title': item.get('title'),
                            'url': item.get('url'),
                            'site': 'FMP News',
                            'summary': item.get('text', '')
                        })
                
                # FMP for economic calendar if not already populated
                if not econ_calendar:
                    econ_calendar = self.fetcher.fmp_fetcher.fetch_economic_calendar(days_forward=3)
                
                # FMP for portfolio news DAILY 
                if not portfolio_news and portfolio_tickers:
                    portfolio_news = self.fetcher.fmp_fetcher.fetch_stock_news(portfolio_tickers, limit=3)
            except Exception as e:
                logger.error(f"Failed to fetch FMP daily news: {e}")

        # Fallback for news if everything else failed
        if not market_news:
            try:
                logger.info("Falling back to basic yfinance news...")
                for t in ['SPY', 'QQQ', 'DIA']:
                    stock = yf.Ticker(t)
                    for n in (stock.news or [])[:2]:
                        market_news.append({
                            'title': n.get('title'),
                            'url': n.get('link'),
                            'site': 'Yahoo Finance'
                        })
            except Exception as e:
                logger.error(f"News fallback failed: {e}")

        market_news = _dedupe_news(market_news, limit=12)

        if not portfolio_news and portfolio_tickers:
            try:
                for t in portfolio_tickers[:5]:
                    stock = yf.Ticker(t)
                    for n in (stock.news or [])[:1]:
                        portfolio_news.append({
                            'title': n.get('title'),
                            'symbol': t,
                            'url': n.get('link')
                        })
            except Exception as e:
                logger.error(f"Portfolio news fallback failed: {e}")

        # 2.5 Near-Term Catalysts (Earnings & Markets)
        catalysts = []
        if earnings_cal:
            for e in earnings_cal[:3]:
                catalysts.append(f"**{e.get('symbol')}** Earnings: {e.get('date')} ({e.get('hour', '').upper()})")
        if econ_calendar:
             for ev in econ_calendar[:2]:
                 catalysts.append(f"**{ev.get('event')}**: {ev.get('date')}")

        # 3. Dynamic Sector & Cap Analysis
        sector_perf = []
        cap_perf = {}
        index_perf = {}
        if self.fetcher and self.fetcher.fmp_available and self.fetcher.fmp_fetcher:
            try:
                sector_perf = self.fetcher.fmp_fetcher.fetch_sector_performance()
                # Sort sectors by performance
                if sector_perf:
                    sector_perf = sorted(sector_perf, key=lambda x: float(x.get('changesPercentage', '0').replace('%','')), reverse=True)
            except Exception as e:
                logger.error(f"Sector perf fetch failed: {e}")
        
        # Cap Segment Analysis (SPY, MDY, IWM)
        try:
            for symbol, label in [('SPY', 'Large Cap'), ('MDY', 'Mid Cap'), ('IWM', 'Small Cap')]:
                t = yf.Ticker(symbol)
                hist = t.history(period='2d')
                if len(hist) >= 2:
                    change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                    cap_perf[label] = round(change, 2)
        except Exception as e:
            logger.error(f"Cap perf check failed: {e}")

        # Major index snapshot + sentiment/movers via Finnhub integration
        market_snapshot = {}
        sentiment_proxy = {"score": 50.0, "label": "Neutral", "components": []}
        notable_movers = []
        if self.finnhub.api_key:
            try:
                market_snapshot = self.finnhub.fetch_major_index_snapshot()
                sentiment_proxy = self.finnhub.fetch_market_sentiment_proxy()
                notable_movers = self.finnhub.fetch_notable_movers(limit=6)
                for item in market_snapshot.values():
                    index_perf[item.get('label', item.get('symbol', 'Index'))] = item.get('change_pct', 0.0)
            except Exception as e:
                logger.error(f"Finnhub snapshot check failed: {e}")

        if not index_perf:
            try:
                for symbol, label in [('SPY', 'S&P 500'), ('QQQ', 'NASDAQ 100'), ('DIA', 'Dow Jones'), ('IWM', 'Russell 2000')]:
                    hist = yf.Ticker(symbol).history(period='2d')
                    if len(hist) >= 2:
                        move = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                        index_perf[label] = round(move, 2)
            except Exception as e:
                logger.error(f"Index performance check failed: {e}")

        # 4. Generate Charts
        sector_chart_path = ""
        cap_chart_path = ""
        if sector_perf:
            sector_chart_path = self.visualizer.generate_sector_chart(sector_perf)
        if cap_perf:
            cap_chart_path = self.visualizer.generate_cap_comparison(cap_perf)

        # 5. QotD & Historical insights (Multiple QotDs for PRISM style)
        qotds = []
        for _ in range(3):
            qotds.append(self.ai_agent.generate_qotd())
            
        history_insight = ""
        # 6. Build content (institutional style, concise + actionable)
        content = []
        date_str = datetime.now().strftime('%B %d, %Y')
        top_buys = top_buys or []
        top_sells = top_sells or []
        market_status = market_status or {}

        def _safe_num(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        spy_trend = market_status.get('spy', {}).get('trend', 'NEUTRAL')
        ad_ratio = _safe_num(market_status.get('breadth', {}).get('advance_decline_ratio'))
        pct_above_200 = _safe_num(market_status.get('breadth', {}).get('percent_above_200sma'))
        
        # Header
        content.append("# 🏛️ AlphaIntelligence Capital — Daily Market Brief")
        content.append(f"**Date:** {date_str}")
        content.append("")
        content.append("A professional decision memo combining market structure, risk context, and top opportunities.")
        content.append("")
        
        headline = "Institutional Sentiment Stabilizes Amidst Technical Consolidation"
        if market_news:
            headline = market_news[0]['title']
        content.append(f"## Executive Headline: {headline}")
        
        if market_news:
             content.append(f"{market_news[0].get('summary', 'Market participants are evaluating recent volatility as earnings season developments provide a mixed technical backdrop.')}")
        content.append("")

        # Compact Market Snapshot block near the top
        content.append("## Market Snapshot")
        snapshot_headline = f"Tape check: sentiment proxy at {sentiment_proxy.get('score', 50.0):.1f}/100 ({sentiment_proxy.get('label', 'Neutral')}) with mixed cross-asset leadership."
        if notable_movers:
            dominant = max(notable_movers, key=lambda x: abs(x.get('change_pct', 0.0)))
            snapshot_headline = (
                f"Tape check: {dominant.get('symbol')} is leading notable flow ({dominant.get('change_pct', 0.0):+,.2f}%), "
                f"while sentiment proxy sits at {sentiment_proxy.get('score', 50.0):.1f}/100 ({sentiment_proxy.get('label', 'Neutral')})."
            )
            snapshot_headline = ''.join(snapshot_headline)
        content.append(f"- **Headline:** {snapshot_headline}")

        strip = []
        for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
            data = market_snapshot.get(symbol, {})
            if data:
                change = data.get('change_pct', 0.0)
                arrow = '▲' if change >= 0 else '▼'
                strip.append(f"{symbol} {arrow} {change:+.2f}%")
            elif symbol == 'SPY' and 'S&P 500' in index_perf:
                change = index_perf['S&P 500']
                arrow = '▲' if change >= 0 else '▼'
                strip.append(f"SPY {arrow} {change:+.2f}%")
            elif symbol == 'QQQ' and 'NASDAQ 100' in index_perf:
                change = index_perf['NASDAQ 100']
                arrow = '▲' if change >= 0 else '▼'
                strip.append(f"QQQ {arrow} {change:+.2f}%")
            elif symbol == 'DIA' and 'Dow Jones' in index_perf:
                change = index_perf['Dow Jones']
                arrow = '▲' if change >= 0 else '▼'
                strip.append(f"DIA {arrow} {change:+.2f}%")
            elif symbol == 'IWM' and 'Russell 2000' in index_perf:
                change = index_perf['Russell 2000']
                arrow = '▲' if change >= 0 else '▼'
                strip.append(f"IWM {arrow} {change:+.2f}%")
        if strip:
            content.append(f"- **Index Strip:** {' | '.join(strip)}")

        if sector_perf and len(sector_perf) >= 2:
            best = sector_perf[0]
            worst = sector_perf[-1]
            content.append(
                f"- **Sector Divergence:** {best.get('sector')} leads at {best.get('changesPercentage')}, while {worst.get('sector')} lags at {worst.get('changesPercentage')}."
            )

        if notable_movers:
            mover_bits = []
            for m in notable_movers[:3]:
                mover_bits.append(f"{m.get('symbol')} ({m.get('change_pct', 0.0):+,.2f}%: {m.get('reason', 'Notable move')})")
            content.append(f"- **Notable Movers:** {', '.join(mover_bits)}.")
        content.append("")
        content.append("---")
        content.append("")

        # --- SECTION: MARKET OVERVIEW ---
        content.append("## 1) Market Overview")
        content.append("### Sentiment & Breadth")
        sentiment_score = 55.5
        sentiment_label = "Neutral"
        if trending_entities:
            avg_sent = sum(e.get('sentiment_avg', 0) for e in trending_entities) / len(trending_entities)
            sentiment_score = 50 + (avg_sent * 50)
            sentiment_label = "Greed" if sentiment_score > 60 else "Fear" if sentiment_score < 40 else "Neutral"
        
        content.append(f"- **Sentiment Gauge:** {sentiment_score:.1f}/100 ({sentiment_label})")
        content.append(f"- **SPY Trend Regime:** {spy_trend}")
        if ad_ratio > 0:
            content.append(f"- **Advance/Decline Ratio:** {ad_ratio:.2f}")
        if pct_above_200 > 0:
            content.append(f"- **% of Universe Above 200 SMA:** {pct_above_200:.1f}%")
        content.append("")
        
        content.append("### Market Mood")
        mood_driver = "Balanced risk appetite with mixed conviction"
        if sentiment_score >= 60:
            mood_driver = "Risk-on posture with broad participation"
        elif sentiment_score <= 40:
            mood_driver = "Risk-off posture with defensive bias"
        content.append(f"- **Desk Take:** {mood_driver}.")
        content.append("")

        if index_perf:
            content.append("### Early Tape")
            content.append("| Index | Move |")
            content.append("|---|---:|")
            for label, move in index_perf.items():
                arrow = "▲" if move > 0 else "▼"
                content.append(f"| {label} | {arrow} {move:+.2f}% |")
            content.append("")

        if cap_perf:
            content.append("### Market-Cap Leadership")
            sorted_caps = sorted(cap_perf.items(), key=lambda x: x[1], reverse=True)
            for label, move in sorted_caps:
                content.append(f"- **{label}:** {'▲' if move > 0 else '▼'} {move:+.2f}%")
        content.append("")

        # --- SECTION: MACRO PULSE (NEW: FRED DATA) ---
        if econ_data:
            content.append("### Macro Pulse (FRED)")
            for name, data in econ_data.items():
                trend_icon = "▲" if data['trend'] == "Up" else "▼" if data['trend'] == "Down" else "•"
                content.append(f"- **{name}:** {data['current']} ({trend_icon} from {data['previous']})")
            content.append("")

        # --- SECTION: SECTOR PERFORMANCE ---
        if sector_perf:
            content.append("### Sector Performance")
            leaders = sector_perf[:3]
            laggards = list(reversed(sector_perf[-3:])) if len(sector_perf) >= 3 else []
            content.append("**Leaders**")
            for s in leaders:
                change_str = s.get('changesPercentage', '0.00%')
                content.append(f"- **{s.get('sector')}:** {change_str}")
            if laggards:
                content.append("**Laggards**")
                for s in laggards:
                    change_str = s.get('changesPercentage', '0.00%')
                    content.append(f"- **{s.get('sector')}:** {change_str}")
            content.append("")

        if cap_chart_path and Path(cap_chart_path).exists():
            content.append(f"![Market-cap segment performance]({cap_chart_path})")
            content.append("")
        if sector_chart_path and Path(sector_chart_path).exists():
            content.append(f"![Sector performance chart]({sector_chart_path})")
            content.append("")

        # --- SECTION: TRADE IDEAS ---
        content.append("## 2) Actionable Watchlist")
        if top_buys:
            content.append("### High-Conviction Long Setups")
            for i, idea in enumerate(top_buys[:5], 1):
                ticker = idea.get('ticker', 'N/A')
                score = _safe_num(idea.get('score'))
                price = _safe_num(idea.get('current_price'))
                thesis = idea.get('fundamental_snapshot') or "Technical and fundamental signals are aligned."
                content.append(f"{i}. **{ticker}** — Score {score:.1f} | Price ${price:.2f}")
                content.append(f"   - Thesis: {thesis}")
            content.append("")

        if top_sells:
            content.append("### Risk-Off / Exit Candidates")
            for i, idea in enumerate(top_sells[:5], 1):
                ticker = idea.get('ticker', 'N/A')
                score = _safe_num(idea.get('score'))
                price = _safe_num(idea.get('current_price'))
                reason = idea.get('reason') or "Momentum deterioration or risk-control trigger."
                content.append(f"{i}. **{ticker}** — Score {score:.1f} | Price ${price:.2f}")
                content.append(f"   - Exit Logic: {reason}")
            content.append("")

        if fund_performance_md:
            content.append("## 3) Fund Performance Snapshot")
            content.append(fund_performance_md.strip())
            content.append("")

        if catalysts:
            content.append("## 4) Near-Term Catalysts")
            for catalyst in catalysts[:6]:
                content.append(f"- {catalyst}")
            content.append("")

        content.append("## 5) Notable Movers")
        if notable_movers:
            for mover in notable_movers[:6]:
                direction = '▲' if mover.get('change_pct', 0.0) >= 0 else '▼'
                content.append(
                    f"- **{mover.get('symbol', 'N/A')}** {direction} {mover.get('change_pct', 0.0):+,.2f}% — {mover.get('reason', 'Notable move')}"
                )
        elif top_buys or top_sells:
            for idea in top_buys[:3]:
                ticker = idea.get('ticker', 'N/A')
                score = _safe_num(idea.get('score'))
                content.append(f"- **{ticker}** flagged long with strong composite score ({score:.1f}).")
            for idea in top_sells[:3]:
                ticker = idea.get('ticker', 'N/A')
                score = _safe_num(idea.get('score'))
                content.append(f"- **{ticker}** flagged as risk-off candidate ({score:.1f}); monitor for relative weakness.")
        elif trending_entities:
            for ent in trending_entities[:4]:
                content.append(f"- **{ent.get('key')}** showing elevated narrative flow ({ent.get('total_documents', 0)} documents).")
        else:
            content.append("- No reliable mover data available from configured feeds this run.")
        content.append("")

        # --- SECTION: QUESTIONS OF THE DAY ---
        content.append("## 6) Questions of the Day")
        for i, q in enumerate(qotds, 1):
            content.append(f"### {q.get('question')}")
            content.append(f"📊 **The Answer**: {q.get('answer')}")
            content.append(f"\n*{q.get('insight')}*")
            if i < len(qotds): content.append("\n---")
        content.append("")

        # --- SECTION: WHAT HISTORY SAYS ---
        content.append("## 7) What History Says")
        if top_buys:
             t_stock = top_buys[0].get('ticker')
             hist_comment = self.ai_agent._call_ai(f"Provide professional historical context for {t_stock} relative to its technical setup. 3 sentences.")
             content.append(f"### {t_stock} Context")
             content.append(f"{hist_comment or history_insight}")
        else:
             content.append(f"{history_insight}")
        content.append("")

        # --- SECTION: TOP HEADLINES ---
        if market_news:
            content.append("## 8) Top Headlines")
            for item in market_news[:8]:
                title = item.get('title', 'No Title')
                url = item.get('url', '#')
                site = item.get('site', 'News')
                summary = item.get('summary', '')
                if summary:
                    content.append(f"- [{title}]({url}) — *{site}*\n  - {summary[:180].rstrip()}...")
                else:
                    content.append(f"- [{title}]({url}) — *{site}*")
            content.append("")

        if portfolio_news:
            content.append("## 9) Portfolio-Specific News")
            for item in portfolio_news[:6]:
                title = item.get('title', 'No Title')
                symbol = item.get('symbol', 'N/A')
                url = item.get('url', '#')
                content.append(f"- **{symbol}:** [{title}]({url})")
            content.append("")

        if earnings_cal:
            content.append("## 10) Earnings Radar (Next 5 Days)")
            for event in earnings_cal[:8]:
                symbol = event.get('symbol', 'N/A')
                date = event.get('date', '')
                eps_est = event.get('epsEstimate', 'N/A')
                content.append(f"- **{date}** — {symbol} (EPS est: {eps_est})")
            content.append("")

        if portfolio_news:
            content.append("## 7) Portfolio-Specific News")
            for item in portfolio_news[:6]:
                title = item.get('title', 'No Title')
                symbol = item.get('symbol', 'N/A')
                url = item.get('url', '#')
                content.append(f"- **{symbol}:** [{title}]({url})")
            content.append("")

        if earnings_cal:
            content.append("## 8) Earnings Radar (Next 5 Days)")
            for event in earnings_cal[:8]:
                symbol = event.get('symbol', 'N/A')
                date = event.get('date', '')
                eps_est = event.get('epsEstimate', 'N/A')
                content.append(f"- **{date}** — {symbol} (EPS est: {eps_est})")
            content.append("")

        # --- SECTION: TODAY'S EVENTS ---
        if econ_calendar:
            content.append("## 11) Today's Events")
            for event in econ_calendar[:8]:
                date = event.get('date', '')
                title = event.get('event', 'Economic Event')
                content.append(f"○ **{date}** {title}")
            content.append("")

        # --- GLOSSARY SECTION ---
        content.append("## 📖 Glossary")
        content.append("- **Z-Score**: A statistical measurement of a value's relationship to the mean.")
        content.append("- **RSI**: Momentum indicator measuring speed and change of price movements.")
        content.append("- **Volatility**: Dispersion of returns for a given security or market index.")
        content.append("")
        content.append("---")
        content.append(f"**AlphaIntelligence Capital**")
        content.append(f"This content is for informational purposes only. [Unsubscribe](https://alphaintelligence.capital/unsubscribe)")

        # 3. Enhance whole newsletter with AI for premium feel
        final_md = "\n".join(content)
        if self.ai_agent.api_key:
            logger.info("Enhancing newsletter prose with AI...")
            final_md = self.ai_agent.enhance_newsletter(final_md)

        # Save to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_md)
            
        logger.info(f"Professional Newsletter generated at {output_path}")
        return output_path

    def generate_quarterly_newsletter(self,
                                   portfolio: any,
                                   top_stocks: Dict,
                                   top_etfs: Dict,
                                   output_path: Optional[str] = None) -> str:
        """Generate the comprehensive professional quarterly compounder newsletter."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path("./data/newsletters/quarterly")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"quarterly_compounder_{timestamp}.md")

        logger.info(f"Generating professional quarterly newsletter to {output_path}...")

        # 1. Gather Data context
        quarter_date = datetime.now()
        q = (quarter_date.month - 1) // 3 + 1
        year = quarter_date.year

        trending = self.marketaux.fetch_trending_entities() if self.marketaux.api_key else []
        econ_cal = []
        if self.fetcher and self.fetcher.fmp_available:
            econ_cal = self.fetcher.fmp_fetcher.fetch_economic_calendar(days_forward=30) 

        # Enhanced AI Thesis for Quarterly
        ai_thesis = "Institutional compounder selection focused on capital efficiency and moat depth."
        if self.ai_agent.client:
            prompt = f"Act as a hedge fund macro strategist. Provide a 4-sentence quarterly investment thesis for Q{q} {year} focusing on inflation, interest rates, and sector leadership. Be precise and sophisticated."
            ai_thesis = self.ai_agent._call_ai(prompt)

        # 2. Build Content (PRISM style refactor)
        content = []
        content.append(f"## 🏛️ AlphaIntelligence Capital | STRATEGIC QUARTERLY")
        content.append(f"# Q{q} {year} Compounder Report")
        content.append(f"*High-Conviction Allocation & Multi-Year Growth Framework*")
        content.append(f"**Horizon Date:** {datetime.now().strftime('%B %Y')}")
        content.append("\n" + "---" + "\n")

        # --- SECTION: QUARTERLY MACRO THESIS ---
        content.append("## 📜 Quarterly Investment Thesis")
        content.append(f"{ai_thesis}")
        content.append("")
        
        # Multiple AI QotDs
        content.append("## 💡 Institutional Historical Insights")
        for i in range(2):
            q_data = self.ai_agent.generate_qotd()
            content.append(f"### {q_data.get('question')}")
            content.append(f"📊 **The Answer**: {q_data.get('answer')}")
            content.append(f"\n*{q_data.get('insight')}*")
            if i == 0: content.append("\n---")
        content.append("")

        # --- SECTION: MARKET TRENDING ---
        if trending:
            content.append("## 🛸 Trending Institutional Interest")
            content.append("| Sector/Entity | Sentiment | Volume | Analysis |")
            content.append("|---------------|-----------|--------|----------|")
            for ent in trending[:5]:
                content.append(f"| {ent.get('key')} | {ent.get('sentiment_avg', 0):+.2f} | {ent.get('total_documents')} docs | Market Leader |")
            content.append("")

        # --- SECTION: PORTFOLIO ARCHITECTURE ---
        content.append("## 🧭 Portfolio Governance & Architecture")
        content.append("| Metric | Value | Benchmark |")
        content.append("|--------|-------|-----------|")
        content.append(f"| **Portfolio Quality Score** | {portfolio.total_score:.1f}/100 | > 75.0 |")
        content.append(f"| **Diversification Score** | {(1.0 - portfolio.sector_concentration):.3f} | > 0.700 |")
        content.append(f"| **Total Strategic Positions** | {portfolio.total_positions} | 15-25 |")
        content.append("")

        content.append("### 🛰️ Asset Class Allocation")
        content.append(f"- **Core (60%)**: {len(portfolio.core_allocations)} High-Conviction Individual Compounders")
        content.append(f"- **Satellite (40%)**: {len(portfolio.satellite_allocations)} Thematic/Macro ETF Engines")
        content.append("")

        # --- SECTION: TOP CONVICTION ---
        content.append("## 💎 Top Conviction Picks (Alpha Leaders)")
        content.append("| Rank | Ticker | Allocation | Sector/Theme | Score |")
        content.append("|------|--------|------------|--------------|-------|")
        
        sorted_alloc = sorted(portfolio.allocations.items(), key=lambda x: x[1], reverse=True)
        for rank, (ticker, alloc) in enumerate(sorted_alloc[:10], 1):
            if ticker in top_etfs:
                name = top_etfs[ticker].get('theme', 'Thematic')
                score = top_etfs[ticker].get('score', 0)
                icon = "📦"
            else:
                name = top_stocks.get(ticker, {}).get('sector', 'Unknown')
                score = top_stocks.get(ticker, {}).get('score', 0)
                icon = "🏢"
            
            content.append(f"| {rank} | **{ticker}** {icon} | {alloc:.2%} | {name} | {score:.1f} |")
        content.append("")

        # AI Analysis for Top Pick
        if sorted_alloc:
            top_ticker = sorted_alloc[0][0]
            if top_ticker in top_stocks:
                ai_pick_thesis = self.ai_agent.generate_commentary(top_ticker, {
                    "type": "Quarterly Compounder",
                    "allocation": f"{sorted_alloc[0][1]:.2%}",
                    "details": top_stocks[top_ticker]
                })
                content.append(f"### 🛡️ Strategic Selection Thesis: {top_ticker}")
                content.append(f"*{ai_pick_thesis}*")
                content.append("")

        # --- SECTION: ECONOMIC HORIZON ---
        if econ_cal:
            content.append("## 📅 Event Horizon — Key Quarterly Catalysts")
            content.append("| Date | Event | Impact | Priority |")
            content.append("|------|-------|--------|----------|")
            for event in econ_cal[:8]:
                imp = event.get('impact', 'Medium')
                imp_icon = "🔴" if imp == "High" else "🟡"
                content.append(f"| {event.get('date')} | {event.get('event')} | {imp_icon} {imp} | Strategic |")
            content.append("")

        # --- GLOSSARY & FOOTER ---
        content.append("## 📖 Glossary")
        content.append("- **Compounder**: A high-quality company capable of generating high returns on invested capital over many years.")
        content.append("- **Moat**: A sustainable competitive advantage that protects a company's long-term profits.")
        content.append("- **Alpha**: Excess return relative to a benchmark.")
        content.append("")
        content.append("---")
        content.append(f"*AlphaIntelligence Capital | Strategic Asset Management | {datetime.now().strftime('%Y-%m-%d')}*")
        content.append(f"*Confidential & Proprietary — Wealth Preservation Framework*")
        content.append("[Portal Access](https://alphaintelligence.capital/portal)")

        final_md = "\n".join(content)
        if self.ai_agent.client:
             logger.info("Enhancing quarterly newsletter prose with AI...")
             final_md = self.ai_agent.enhance_newsletter(final_md)

        # Save to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_md)

        return output_path
