
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import yfinance as yf

from ..data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from ..data.finnhub_fetcher import FinnhubFetcher
from ..ai.ai_agent import AIAgent

logger = logging.getLogger(__name__)

class NewsletterGenerator:
    """Generates a high-quality, professional daily market newsletter."""

    def __init__(self, portfolio_path: str = "./data/positions.json", finnhub_fetcher: Optional[FinnhubFetcher] = None):
        self.fetcher = EnhancedFundamentalsFetcher()
        self.finnhub_fetcher = finnhub_fetcher or FinnhubFetcher()
        self.ai_agent = AIAgent()
        self.portfolio_path = Path(portfolio_path)

    @staticmethod
    def _normalize_news_item(item: Dict, *, title_key: str = 'title', url_key: str = 'url', source_key: str = 'site', default_source: str = 'News') -> Dict:
        """Normalize news payloads from multiple providers into one shape."""
        title = (item.get(title_key) or item.get('headline') or 'No Title').strip()
        url = item.get(url_key) or item.get('link') or '#'
        source = item.get(source_key) or item.get('source') or default_source
        return {'title': title, 'url': url, 'site': source}

    @staticmethod
    def _dedupe_news(items: List[Dict], max_items: int = 10) -> List[Dict]:
        """Deduplicate news by URL/title while preserving input order."""
        seen = set()
        result = []
        for raw in items:
            title = (raw.get('title') or '').strip().lower()
            url = (raw.get('url') or '#').strip().lower()
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            result.append(raw)
            if len(result) >= max_items:
                break
        return result

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
                          top_etfs: List[Dict] = None,
                          output_path: Optional[str] = None) -> str:
        """Generate the comprehensive professional daily newsletter."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path("./data/newsletters")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"daily_newsletter_{timestamp}.md")
            
        logger.info(f"Generating professional daily newsletter to {output_path}...")
        
        # 1. Fetch Data
        econ_data = {}
        market_news = []
        portfolio = self.load_portfolio()
        portfolio_tickers = [p['ticker'] for p in portfolio]
        portfolio_news = []
        
        if self.fetcher.fmp_available and self.fetcher.fmp_fetcher:
            try:
                econ_data = self.fetcher.fmp_fetcher.fetch_economic_data()
                market_news = self.fetcher.fmp_fetcher.fetch_market_news(limit=5)
                if portfolio_tickers:
                    portfolio_news = self.fetcher.fmp_fetcher.fetch_stock_news(portfolio_tickers, limit=3)
            except Exception as e:
                logger.error(f"Failed to fetch FMP data: {e}")

        # 1.5 Fetch Finnhub Data
        finnhub_market_news = []
        finnhub_sentiment = {}
        if self.finnhub_fetcher.api_key:
            try:
                finnhub_market_news = (self.finnhub_fetcher.get_market_news(category="general") or [])[:5]
                # Get sentiment for top 5 portfolio stocks
                for t in portfolio_tickers[:5]:
                    sentiment = self.finnhub_fetcher.get_news_sentiment(t)
                    if sentiment:
                        finnhub_sentiment[t] = sentiment
            except Exception as e:
                logger.error(f"Failed to fetch Finnhub data: {e}")

        # Fallback for news if FMP failed or returned nothing
        if not market_news:
            try:
                logger.info("FMP news unavailable, falling back to basic news...")
                # Use a few bellwether stocks for general news
                for t in ['SPY', 'QQQ', 'DIA']:
                    stock = yf.Ticker(t)
                    for n in (stock.news or [])[:2]:
                        market_news.append(self._normalize_news_item(n, url_key='link', source_key='publisher', default_source='Yahoo Finance'))
                market_news = self._dedupe_news(market_news, max_items=8)
            except Exception as e:
                logger.error(f"News fallback failed: {e}")

        if not portfolio_news and portfolio_tickers:
            try:
                for t in portfolio_tickers[:5]: # Max 5 for speed
                    stock = yf.Ticker(t)
                    for n in (stock.news or [])[:1]:
                        normalized = self._normalize_news_item(n, url_key='link', source_key='publisher', default_source='Yahoo Finance')
                        normalized['symbol'] = t
                        portfolio_news.append(normalized)
                portfolio_news = self._dedupe_news(portfolio_news, max_items=8)
            except Exception as e:
                logger.error(f"Portfolio news fallback failed: {e}")

        # Normalize and deduplicate all news streams for cleaner newsletter output
        market_news = [self._normalize_news_item(n) for n in (market_news or [])]
        market_news = self._dedupe_news(market_news, max_items=10)

        finnhub_market_news = [self._normalize_news_item(n, title_key='headline', url_key='url', source_key='source', default_source='Finnhub') for n in (finnhub_market_news or [])]
        finnhub_market_news = self._dedupe_news(finnhub_market_news, max_items=10)

        portfolio_news = [self._normalize_news_item(n, default_source='Portfolio News') | ({'symbol': n.get('symbol')} if n.get('symbol') else {}) for n in (portfolio_news or [])]
        portfolio_news = self._dedupe_news(portfolio_news, max_items=10)

        # 2. Build Content
        content = []
        date_str = datetime.now().strftime('%A, %B %d, %Y')
        
        # Header
        content.append(f"# 🏦 Alpha Intelligence Daily - {date_str}")
        content.append(f"*Institutional-Grade Market Insights & Premium AI Commentary*")
        content.append("")
        
        # --- SECTION 1: MACRO ECONOMY ---
        content.append("## 🌍 Macroeconomic Intelligence")
        if econ_data:
            content.append("| Indicator | Current | Trend | Previous | Analysis |")
            content.append("|-----------|---------|-------|----------|----------|")
            for name, data in econ_data.items():
                trend_icon = "↗️" if data.get('trend') == 'Up' else "↘️"
                val = data.get('current')
                prev = data.get('previous')
                
                # Analysis thinking
                analysis = "Stable"
                if name == 'CPI':
                    analysis = "Inflationary risk" if val > 3.0 else "Within target"
                elif name == 'GDP':
                    analysis = "Strong growth" if val > 2.5 else "Moderate"
                elif name == 'Unemployment':
                    analysis = "Tight labor market" if val < 4.0 else "Loosening"
                
                # Format values
                val_fmt = "N/A"
                if val:
                    val_fmt = f"{val:.2f}%" if name != 'GDP' else f"${val:.1f}T"
                
                prev_fmt = "N/A"
                if prev:
                    prev_fmt = f"{prev:.2f}%" if name != 'GDP' else f"${prev:.1f}T"
                
                content.append(f"| {name} | **{val_fmt}** | {trend_icon} | {prev_fmt} | {analysis} |")
            content.append("")
        else:
            content.append("*Macro data unavailable. Verify FMP API connectivity.*")
            content.append("")

        # --- SECTION 2: PORTFOLIO DASHBOARD ---
        if portfolio:
            content.append("## 💼 Portfolio Intelligence")
            content.append("| Ticker | Qty | Buy Price | Focus |")
            content.append("|--------|-----|-----------|-------|")
            for p in portfolio:
                content.append(f"| **{p['ticker']}** | {p['quantity']} | ${p['average_buy_price']:.2f} | Monitoring |")
            content.append("")
            
            if portfolio_news:
                content.append("### 🗞️ Portfolio News & Catalysts")
                for item in portfolio_news:
                    title = item.get('title', 'No Title')
                    symbol = item.get('symbol', '')
                    url = item.get('url', '#')
                    content.append(f"- **{symbol}**: [{title}]({url})")
                content.append("")

        # --- SECTION 3: MARKET HEALTH & REGIME ---
        if market_status:
            spy = market_status.get('spy', {})
            breadth = market_status.get('breadth', {})
            
            content.append("## 🏥 Market Regime & Health")
            spy_trend = spy.get('trend', 'Unknown')
            spy_price = spy.get('current_price', 0)
            trend_emoji = "🟢" if spy_trend == "UPTREND" else "🔴" if spy_trend == "DOWNTREND" else "🟡"
            
            content.append(f"### Benchmark: {trend_emoji} {spy_trend} (SPY ${(spy_price or 0):.2f})")
            
            adv_dec = breadth.get('advance_decline_ratio') or 0
            stocks_above_ma = breadth.get('percent_above_200sma') or breadth.get('bullish_pct') or 0
            
            content.append(f"- **Market Breadth (AD Ratio)**: {(adv_dec or 0):.2f}")
            content.append(f"- **Participation (> 200 SMA)**: {(stocks_above_ma or 0):.1f}%")
            
            # Smart thinking
            regime_note = "Cautious"
            if spy_trend == "UPTREND" and stocks_above_ma > 60:
                regime_note = "**Aggressive Deployment** - Market environment is highly supportive."
            elif spy_trend == "DOWNTREND":
                regime_note = "**Capital Preservation** - High cash levels recommended."
            else:
                regime_note = "**Selective Bias** - Focus only on elite relative strength."
            
            content.append(f"\n> **Strategic Bias**: {regime_note}")
            content.append("")

        # --- SECTION 4: ELITE OPPORTUNITIES ---
        content.append("## 🎯 Actionable Intelligence - Premium Coverage")
        
        if top_buys:
            content.append(f"### 🟢 Priority Acculmulation ({len(top_buys)} setups detected)")
            for i, signal in enumerate(top_buys[:5], 1): # Top 5 only
                ticker = signal.get('ticker', 'UNKNOWN')
                score = signal.get('score', 0)
                price = signal.get('current_price') or signal.get('breakout_price') or 0
                
                snap = signal.get('fundamental_snapshot', "")
                
                # AI Commentary for Buy signals
                ai_note = ""
                if self.ai_agent.api_key:
                    ai_note = self.ai_agent.generate_commentary(ticker, {
                        "price": price, 
                        "score": score,
                        "ratios": snap.replace("\n", " ")[:500]
                    })

                content.append(f"**{i}. {ticker}** (Tactical Score: {(score or 0):.1f}) - **${(price or 0):.2f}**")
                if ai_note:
                    content.append(f"  - **AI Thesis**: *{ai_note}*")
                
                # Extract DCF
                if "Intrinsic Value (DCF):" in snap:
                    dcf_val = snap.split("Intrinsic Value (DCF):")[1].split("\n")[0].strip()
                    content.append(f"  - *DCF Target: {dcf_val}*")
                
                # Financial Strength
                if "Balance Sheet & Efficiency:" in snap:
                    bs_section = snap.split("Balance Sheet & Efficiency:")[1].split("Overall Assessment")[0].strip()
                    content.append("  - **Financial Strength**:")
                    for line in bs_section.split("\n"):
                        if line.strip():
                            content.append(f"    - {line.strip()}")
                
                content.append("")
        else:
            content.append("*Zero high-conviction buy signals identified.*")
            
        content.append("")
        
        # --- SECTION 5: THEMATIC ETF STRATEGY ---
        if top_etfs:
            content.append("## 🚀 Thematic Strategy (ETF Intelligence)")
            content.append("| Rank | Ticker | Theme | Score | Tailwind | Analysis |")
            content.append("|------|--------|-------|-------|----------|----------|")
            for i, etf in enumerate(top_etfs[:5], 1): # Top 5 ETFs
                ticker = etf.get('ticker', 'N/A')
                theme = etf.get('theme_name', 'Thematic')
                score = etf.get('total_score', 0)
                tailwind = etf.get('tailwind_score', 0)
                
                # Dynamic analysis based on scores
                analysis = "Strong Setup" if score > 75 else "Accumulate"
                if tailwind >= 8.0:
                    analysis = "High Tailwind (Structural Lead)"
                
                content.append(f"| {i} | **{ticker}** | {theme} | {score:.1f} | {tailwind:.1f}/10 | {analysis} |")
            
            content.append("\n> **Thematic View**: Focus remains on high-purity ETFs with persistent relative strength.")
            content.append("")

        if top_sells:
            content.append(f"### 🔴 Risk Management Alerts")
            for i, signal in enumerate(top_sells[:5], 1):
                 ticker = signal.get('ticker')
                 score = signal.get('score')
                 reason = signal.get('reasons', ['Signal triggered'])[0]
                 content.append(f"**{i}. {ticker}** - *{reason}* (Risk Score: {score})")
        else:
            content.append("*No exit alerts triggered.*")

        content.append("")
        content.append("---")
        content.append("## 🗞️ Global Market Intel")
        if market_news:
            for item in market_news:
                title = item.get('title', 'No Title')
                site = item.get('site', 'News')
                url = item.get('url', '#')
                content.append(f"- [{title}]({url}) - *{site}*")
        else:
            content.append("*Global news stream offline.*")

        # --- Finnhub Market Pulse ---
        if finnhub_market_news or finnhub_sentiment:
            content.append("---")
            content.append("## 🛰️ Finnhub Market Intelligence")
            
            if finnhub_market_news:
                content.append("### 🗞️ Global Headlines (via Finnhub)")
                for item in finnhub_market_news:
                    title = item.get('title', 'No Title')
                    url = item.get('url', '#')
                    source = item.get('site', 'Finnhub')
                    content.append(f"- [{title}]({url}) - *{source}*")
                content.append("")

            if finnhub_sentiment:
                content.append("### 🧠 Portfolio Sentiment Scorecard")
                content.append("| Ticker | Sentiment Score | Bullish Buzz | Bearish Buzz | Sector Avg |")
                content.append("|--------|-----------------|--------------|--------------|------------|")
                for t, s in finnhub_sentiment.items():
                    score = s.get('sentiment', {}).get('bullishPercent', 50)
                    buzz = s.get('buzz', {}).get('articlesInLastWeek', 0)
                    sector_avg = s.get('sectorAverageBullishPercent', 50)
                    
                    sentiment_label = "🟢 Bullish" if score > 60 else "🔴 Bearish" if score < 40 else "🟡 Neutral"
                    content.append(f"| {t} | {sentiment_label} ({score:.1f}%) | {buzz} articles | - | {sector_avg:.1f}% |")
                content.append("")

        # Provider telemetry (helps diagnose data quality for newsletter readers)
        provider_metrics = self.fetcher.get_provider_metrics()
        if provider_metrics:
            content.append("---")
            content.append("## 🛠️ Data Provider Health")
            content.append("| Provider | Attempts | Success Rate | Avg Latency (ms) |")
            content.append("|----------|----------|--------------|------------------|")
            for provider, stats in provider_metrics.items():
                content.append(f"| {provider.upper()} | {int(stats.get('attempts', 0))} | {stats.get('success_rate', 0):.1f}% | {stats.get('avg_latency_ms', 0):.1f} |")
            content.append("")

        content.append("")
        content.append("---")
        content.append(f"*Created by AlphaIntelligence Engine | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
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

