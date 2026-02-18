
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from html import escape

import yfinance as yf

from ..data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from ..data.finnhub_fetcher import FinnhubFetcher
from ..data.marketaux_fetcher import MarketauxFetcher
from ..data.fred_fetcher import FredFetcher
from ..data.price_service import PriceService
from ..ai.ai_agent import AIAgent
from .visualizer import ChartArtifact, MarketVisualizer

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
        self.price_service = PriceService()
        self.portfolio_path = Path(portfolio_path)
        self.newsletter_state_path = Path("./data/cache/newsletter_state.json")


    def _authoritative_idea_price(self, idea: Dict) -> float:
        """Resolve idea price from yfinance and reject blocked price sources."""
        ticker = idea.get('ticker', '')
        if not ticker:
            return 0.0

        is_valid, source = self.price_service.validate_price_payload_source(
            idea,
            context=f"newsletter idea {ticker}",
        )
        if not is_valid:
            logger.error("Rejecting newsletter idea payload for %s due to blocked price source=%s", ticker, source)
            return 0.0

        price = self.price_service.get_current_price(ticker)
        return float(price) if price and price > 0 else 0.0

    def _load_newsletter_state(self) -> Dict:
        if not self.newsletter_state_path.exists():
            return {"runs": []}
        try:
            with open(self.newsletter_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                if isinstance(state, dict) and isinstance(state.get("runs"), list):
                    return state
        except Exception as e:
            logger.warning(f"Failed to read newsletter state: {e}")
        return {"runs": []}

    def _save_newsletter_state(self, state: Dict) -> None:
        try:
            self.newsletter_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.newsletter_state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist newsletter state: {e}")

    def _extract_entities_topics(self, items: List[Dict]) -> Dict[str, List[str]]:
        topic_keywords = {
            "rates": ["rate", "yield", "fed", "treasury"],
            "volatility": ["volatility", "vix", "drawdown", "swing"],
            "earnings": ["earnings", "guidance", "eps", "revenue"],
            "macro": ["inflation", "cpi", "jobs", "unemployment", "gdp"],
            "tech": ["ai", "chip", "software", "cloud"],
            "energy": ["oil", "gas", "energy", "opec"],
            "banks": ["bank", "credit", "lending", "financial"]
        }
        entities = set()
        topics = set()
        for item in items:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            for token in re.findall(r"\b[A-Z]{2,5}\b", title):
                entities.add(token)
            lower = title.lower()
            for topic, terms in topic_keywords.items():
                if any(t in lower for t in terms):
                    topics.add(topic)
        return {
            "entities": sorted(entities),
            "topics": sorted(topics)
        }

    def _select_diverse_market_news(self, items: List[Dict], state: Dict, limit: int = 8) -> List[Dict]:
        recent_runs = state.get("runs", [])[-5:]
        recent_titles = {
            t.lower()
            for run in recent_runs
            for t in run.get("headline_titles", [])
            if isinstance(t, str)
        }
        recent_topics = {
            t
            for run in recent_runs
            for t in run.get("topics", [])
            if isinstance(t, str)
        }
        recent_entities = {
            e
            for run in recent_runs
            for e in run.get("entities", [])
            if isinstance(e, str)
        }

        scored = []
        for idx, item in enumerate(items):
            title = (item.get("title") or "")
            analysis = self._extract_entities_topics([item])
            score = 100 - idx
            if title.lower() in recent_titles:
                score -= 40
            topic_overlap = len(set(analysis["topics"]) & recent_topics)
            entity_overlap = len(set(analysis["entities"]) & recent_entities)
            score -= (topic_overlap * 8 + entity_overlap * 3)
            item["_topics"] = analysis["topics"]
            item["_entities"] = analysis["entities"]
            scored.append((score, idx, item))

        scored.sort(key=lambda x: (-x[0], x[1]))

        selected = []
        used_sources = set()
        used_topics = set()
        for _, _, item in scored:
            if len(selected) >= limit:
                break
            source = (item.get("site") or "News").strip().lower()
            topics = set(item.get("_topics", []))

            source_penalty = source in used_sources
            topic_penalty = bool(topics and topics.issubset(used_topics))
            if len(selected) < 3:
                # Early picks favor broad source/topic diversification.
                if source_penalty and topic_penalty:
                    continue
            selected.append(item)
            used_sources.add(source)
            used_topics.update(topics)

        if len(selected) < min(limit, len(scored)):
            selected_keys = {(s.get("title"), s.get("url")) for s in selected}
            for _, _, item in scored:
                key = (item.get("title"), item.get("url"))
                if key in selected_keys:
                    continue
                selected.append(item)
                if len(selected) >= limit:
                    break
        return selected

    def _rotate_optional_sections(self) -> List[str]:
        section_pool = [
            "Volatility Watch",
            "Rates Pulse",
            "Earnings Spotlight",
            "Insider/Flow Watch"
        ]
        day_index = datetime.now().toordinal()
        shift = day_index % len(section_pool)
        count = 2 + (day_index % 2)
        rotated = section_pool[shift:] + section_pool[:shift]
        return rotated[:count]

    def _pick_fresh_text(self, candidates: List[str], recent_texts: List[str]) -> str:
        recent_norm = {(t or '').strip().lower() for t in recent_texts if isinstance(t, str)}
        for text in candidates:
            if text.strip().lower() not in recent_norm:
                return text
        return candidates[0] if candidates else ""

    def _latest_previous_newsletter(self, output_path: str) -> Optional[Path]:
        newsletters_dir = Path("./data/newsletters")
        if not newsletters_dir.exists():
            return None
        candidates = [
            p for p in newsletters_dir.glob("daily_newsletter_*.md")
            if str(p) != str(Path(output_path))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _extract_markdown_links(self, markdown_text: str) -> List[Dict]:
        links = []
        for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", markdown_text):
            links.append({"title": title.strip(), "url": url.strip()})
        return links

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

    def _normalize_topic(self, text: str) -> str:
        """Normalize a headline to a coarse topic key for duplication analysis."""
        cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        words = cleaned.split()
        tokens = [t for t in words if len(t) > 3 and t not in {"today", "market", "stocks", "stock", "news"}]
        if not tokens:
            tokens = words
        return " ".join(tokens[:6])

    def _build_qc_fallback_template(self, date_str: str) -> str:
        """Return a safe fallback newsletter that satisfies required sections."""
        lines = [
            "# 🏛️ AlphaIntelligence Capital — Daily Market Brief",
            f"**Date:** {date_str}",
            "",
            "## Executive Headline",
            "Daily report generation completed with limited confidence checks; use a defensive interpretation.",
            "",
            "## 1) Snapshot",
            "- Market internals are currently being revalidated.",
            "- Maintain risk controls until full signal quality is restored.",
            "",
            "## 2) Top Headlines",
            "- [Market structure update pending](https://alphaintelligence.capital) — *AlphaIntelligence*",
            "- [Macro dashboard refresh pending](https://alphaintelligence.capital) — *AlphaIntelligence*",
            "- [Portfolio monitor refresh pending](https://alphaintelligence.capital) — *AlphaIntelligence*",
            "",
            "## 3) Today's Events",
            "- Economic calendar refresh in progress.",
            "",
            "## Disclaimer",
            "This content is for informational purposes only and is not investment advice.",
        ]
        return "\n".join(lines)

    def _run_newsletter_qc(self, markdown: str) -> Tuple[bool, Dict[str, float], List[str]]:
        """Validate newsletter structure and source quality before final output."""
        errors: List[str] = []
        headings = re.findall(r"^(##\s+.+)$", markdown, flags=re.MULTILINE)
        lowered_headings = [h.lower() for h in headings]

        # Rule: no duplicate section headers
        seen: Set[str] = set()
        dupes: Set[str] = set()
        for h in lowered_headings:
            if h in seen:
                dupes.add(h)
            seen.add(h)
        if dupes:
            errors.append("duplicate_headers")

        # Rule: required sections present
        required_fragments = {
            "headline": "executive headline",
            "snapshot": "snapshot",
            "headlines_list": "top headlines",
            "events": "today's events",
            "disclaimer": "disclaimer",
        }
        for label, fragment in required_fragments.items():
            if fragment not in markdown.lower():
                errors.append(f"missing_{label}")

        # Rule: heading order and numbering consistency
        numbered = re.findall(r"^##\s+(\d+)\)\s+(.+)$", markdown, flags=re.MULTILINE)
        if numbered:
            nums = [int(n) for n, _ in numbered]
            expected = list(range(nums[0], nums[0] + len(nums)))
            if nums != expected:
                errors.append("heading_number_sequence")

        # Rule: minimum source count and max duplicate-topic ratio
        links = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", markdown)
        unique_sources = len(set(links))
        min_source_count = 3
        if unique_sources < min_source_count:
            errors.append("insufficient_sources")

        headlines_section = re.search(r"##\s+\d+\)\s+Top Headlines\n(.+?)(?:\n##\s+|\Z)", markdown, flags=re.DOTALL)
        headline_lines = re.findall(r"^-\s+\[([^\]]+)\]", headlines_section.group(1), flags=re.MULTILINE) if headlines_section else []
        topics = [self._normalize_topic(t) for t in headline_lines if t.strip()]
        topic_total = len(topics)
        topic_unique = len(set(topics)) if topics else 0
        duplicate_ratio = 0.0
        if topic_total:
            duplicate_ratio = max(0.0, (topic_total - topic_unique) / topic_total)
        max_duplicate_ratio = 0.45
        if topic_total >= 2 and duplicate_ratio > max_duplicate_ratio:
            errors.append("duplicate_topic_ratio")

        report = {
            "heading_count": float(len(headings)),
            "duplicate_header_count": float(len(dupes)),
            "source_count": float(unique_sources),
            "duplicate_topic_ratio": round(duplicate_ratio, 3),
        }
        return len(errors) == 0, report, errors

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
        state = self._load_newsletter_state()
        recent_runs = state.get("runs", [])[-5:]

        # 1. Initialize data containers
        econ_data = {}
        macro_panel = {}
        macro_panel_fallback = ""
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
        try:
            logger.info("Fetching FRED macro economic indicators...")
            macro_indicators = {
                'GDP': 'GDP',
                'CPI': 'CPIAUCSL',
                'Unemployment': 'UNRATE',
                'Fed Funds': 'FEDFUNDS'
            }
            if self.fred.api_key:
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
            macro_panel = self.fred.get_fixed_macro_panel()
            if not self.fred.api_key:
                macro_panel_fallback = (
                    "FRED API key missing: macro panel uses fallback narrative. "
                    "Set FRED_API_KEY for live regime reads."
                )
        except Exception as e:
            logger.error(f"FRED fetch failed: {e}")
            macro_panel_fallback = "Macro panel unavailable due to FRED fetch error."

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

        market_news = _dedupe_news(market_news, limit=16)
        market_news = self._select_diverse_market_news(market_news, state, limit=8)
        news_analysis = self._extract_entities_topics(market_news)


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

        def _event_impact_tag(event: Dict) -> str:
            impact_raw = str(event.get('impact') or event.get('importance') or '').lower()
            title = str(event.get('event') or '').lower()
            high_tokens = ['high', 'fed', 'cpi', 'payroll', 'fomc', 'rate decision']
            medium_tokens = ['medium', 'pmi', 'ism', 'consumer confidence', 'jobless claims']
            if any(token in impact_raw for token in ['high', '3']) or any(token in title for token in high_tokens):
                return 'HIGH'
            if any(token in impact_raw for token in ['medium', '2']) or any(token in title for token in medium_tokens):
                return 'MEDIUM'
            return 'LOW'

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

        # 4. Generate Default Chart Suite
        chart_artifacts: List[ChartArtifact] = self.visualizer.generate_default_charts(
            index_perf=index_perf,
            sector_perf=sector_perf,
            market_status=market_status or {},
        )

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

        lead_candidates = [
            (market_news[0].get('summary') or '').strip() if market_news else "",
            "Market participants are evaluating recent volatility as earnings season developments provide a mixed technical backdrop.",
            "Cross-asset signals remain mixed, with institutions leaning selective rather than broadly directional.",
            "Positioning remains tactical as desks balance macro uncertainty with idiosyncratic opportunity."
        ]
        lead_candidates = [x for x in lead_candidates if x]
        lead_sentence = self._pick_fresh_text(lead_candidates, [r.get('lead_sentence', '') for r in recent_runs])
        if lead_sentence:
            content.append(lead_sentence)
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
        desk_take = mood_driver
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

        if macro_panel:
            content.append("### Fixed Macro Panel")
            for panel in macro_panel.values():
                content.append(f"- **{panel.get('name', 'Macro Signal')}:** {panel.get('summary', 'No summary available.')}")
            content.append("")

        if macro_panel_fallback:
            content.append(f"- *{macro_panel_fallback}*")
            content.append("")

        if macro_panel:
            spread_text = (macro_panel.get('yield_spread_regime') or {}).get('summary', '').lower()
            labor_text = (macro_panel.get('labor_stress_proxy') or {}).get('summary', '').lower()
            inflation_text = (macro_panel.get('inflation_momentum_proxy') or {}).get('summary', '').lower()

            risk_bias = "Risk-on" if "steep" in spread_text and "improving" in labor_text else "Risk-off"
            inflation_bias = "inflation cooling" if "cooling" in inflation_text else "inflation pressure"
            labor_bias = "labor conditions firm" if "improving" in labor_text else "labor conditions mixed"

            content.append("### Macro Regime")
            content.append(f"- **Regime Bias:** {risk_bias} context from rates-curve and labor signals.")
            content.append(f"- **Inflation Read:** {inflation_bias}; monitor duration/real-rate sensitivity.")
            content.append(f"- **Growth/Labor Read:** {labor_bias}; position sizing should respect event volatility.")
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

        if chart_artifacts:
            content.append("### Chartbook")
            for chart in chart_artifacts:
                if Path(chart.path).exists():
                    content.append(f"#### {chart.title}")
                    content.append(f"![{chart.title}]({chart.path})")
                    content.append(f"*{chart.caption}*")
                    content.append("")

        # --- SECTION: TRADE IDEAS ---
        content.append("## 2) Actionable Watchlist")
        watchlist_intro = self._pick_fresh_text([
            "Focus list balances asymmetric upside with clearly defined risk controls.",
            "Setups below emphasize favorable reward-to-risk with catalyst visibility.",
            "The desk is prioritizing names with technical confirmation and fundamental support."
        ], [r.get("watchlist_intro", "") for r in recent_runs])
        content.append(watchlist_intro)
        content.append("")
        if top_buys:
            content.append("### High-Conviction Long Setups")
            for i, idea in enumerate(top_buys[:5], 1):
                ticker = idea.get('ticker', 'N/A')
                score = _safe_num(idea.get('score'))
                price = _safe_num(self._authoritative_idea_price(idea))
                thesis = idea.get('fundamental_snapshot') or "Technical and fundamental signals are aligned."
                content.append(f"{i}. **{ticker}** — Score {score:.1f} | Price ${price:.2f}")
                content.append(f"   - Thesis: {thesis}")
            content.append("")

        if top_sells:
            content.append("### Risk-Off / Exit Candidates")
            for i, idea in enumerate(top_sells[:5], 1):
                ticker = idea.get('ticker', 'N/A')
                score = _safe_num(idea.get('score'))
                price = _safe_num(self._authoritative_idea_price(idea))
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
            headlines_intro = self._pick_fresh_text([
                "Selected for source and topic diversity to reduce repeat narrative risk.",
                "Cross-source scan prioritizing fresh narratives over recycled headlines.",
                "Headline tape below reflects both macro and single-name dispersion."
            ], [r.get("headlines_intro", "") for r in recent_runs])
            content.append(headlines_intro)
            content.append("")
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

        # --- SECTION: OPTIONAL ROTATION ---
        optional_sections = self._rotate_optional_sections()
        for section_name in optional_sections:
            content.append(f"## {section_name}")
            if section_name == "Volatility Watch":
                vol_msg = "Volatility remains contained relative to recent highs; continue to size entries tactically."
                if sentiment_score <= 40:
                    vol_msg = "Volatility regime remains elevated; risk budgeting should stay defensive until breadth improves."
                content.append(f"- {vol_msg}")
            elif section_name == "Rates Pulse":
                fed = econ_data.get('Fed Funds', {})
                if fed:
                    content.append(f"- Policy backdrop: Fed Funds at **{fed.get('current')}** (prev {fed.get('previous')}).")
                else:
                    content.append("- Treasury-rate direction is mixed; monitor real-yield sensitivity in duration assets.")
            elif section_name == "Earnings Spotlight":
                if earnings_cal:
                    for event in earnings_cal[:3]:
                        content.append(f"- **{event.get('symbol', 'N/A')}** reports {event.get('date', '')}.")
                else:
                    content.append("- No high-confidence earnings catalyst loaded in this run.")
            elif section_name == "Insider/Flow Watch":
                if trending_entities:
                    for ent in trending_entities[:3]:
                        content.append(f"- Narrative flow elevated in **{ent.get('key')}** ({ent.get('total_documents', 0)} docs).")
                else:
                    content.append("- Flow signals are neutral in configured feeds; stay selective on crowded momentum.")
            content.append("")

        # --- SECTION: DELTA VS YESTERDAY ---
        prev_newsletter = self._latest_previous_newsletter(output_path)
        previous_links = []
        if prev_newsletter and prev_newsletter.exists():
            try:
                previous_links = self._extract_markdown_links(prev_newsletter.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f"Unable to parse previous newsletter for delta block: {e}")
        prev_urls = {x.get('url') for x in previous_links if x.get('url')}
        new_since_yesterday = [item for item in market_news if item.get('url') and item.get('url') not in prev_urls]
        content.append("## New Since Yesterday")
        if new_since_yesterday:
            for item in new_since_yesterday[:5]:
                content.append(f"- [{item.get('title', 'No Title')}]({item.get('url', '#')}) — *{item.get('site', 'News')}*")
        else:
            content.append("- No materially new headline links versus the previous newsletter artifact.")
        content.append("")

        # --- SECTION: TODAY'S EVENTS ---
        if econ_calendar:
            content.append("## 11) Today's Events")
            for event in econ_calendar[:8]:
                date = event.get('date', '')
                event_time = event.get('time') or event.get('hour') or 'TBD'
                title = event.get('event', 'Economic Event')
                impact_tag = _event_impact_tag(event)
                content.append(f"- **{date} {event_time}** — {title} `[{impact_tag}]`")
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
        prior_newsletter_md = self._load_prior_newsletter_text(output_path)
        evidence_payload = self._build_evidence_payload(
            market_news=market_news,
            sector_perf=sector_perf,
            index_perf=index_perf,
            top_buys=top_buys,
            top_sells=top_sells,
            earnings_cal=earnings_cal,
            econ_calendar=econ_calendar,
        )
        if self.ai_agent.api_key:
            logger.info("Enhancing newsletter prose with AI validation...")
            final_md = self.ai_agent.enhance_newsletter_with_validation(
                final_md,
                evidence_payload=evidence_payload,
                prior_newsletter_md=prior_newsletter_md,
            )

        qc_ok, qc_report, qc_errors = self._run_newsletter_qc(final_md)
        if not qc_ok:
            logger.warning(
                "Newsletter QC failed (%s). Report: headings=%s, duplicate_headers=%s, sources=%s, duplicate_topic_ratio=%.3f. Falling back to safe template.",
                ",".join(qc_errors),
                int(qc_report.get("heading_count", 0)),
                int(qc_report.get("duplicate_header_count", 0)),
                int(qc_report.get("source_count", 0)),
                qc_report.get("duplicate_topic_ratio", 0.0),
            )
            final_md = self._build_qc_fallback_template(date_str)
        else:
            logger.info(
                "Newsletter QC passed: headings=%s, sources=%s, duplicate_topic_ratio=%.3f",
                int(qc_report.get("heading_count", 0)),
                int(qc_report.get("source_count", 0)),
                qc_report.get("duplicate_topic_ratio", 0.0),
            )

        # Save markdown archive file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_md)

        state_runs = state.get("runs", [])
        state_runs.append({
            "timestamp": datetime.now().isoformat(),
            "headline_titles": [x.get("title") for x in market_news[:8] if x.get("title")],
            "entities": news_analysis.get("entities", []),
            "topics": news_analysis.get("topics", []),
            "lead_sentence": lead_sentence,
            "watchlist_intro": watchlist_intro,
            "headlines_intro": headlines_intro if market_news else "",
            "optional_sections": optional_sections
        })
        state["runs"] = state_runs[-5:]
        self._save_newsletter_state(state)
            
        logger.info(f"Professional Newsletter generated at {output_path}")
        logger.info(f"Newsletter HTML presentation generated at {html_output}")
        return output_path


    def _read_light_template(self) -> str:
        """Load the default light HTML template for email-safe rendering."""
        default_template = """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body>{{hero_headline}}{{market_mood}}{{indices_strip}}{{sector_table}}{{movers}}{{headlines}}{{events}}{{disclaimer}}</body>
</html>"""
        try:
            if self.template_path.exists():
                return self.template_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to load newsletter template: {e}")
        return default_template

    def build_hero_headline(self, date_str: str, headline: str, summary: str) -> str:
        summary_text = summary or "Institutional desk commentary unavailable for this run; refer to section detail below."
        return (
            '<section class="card hero-card">'
            '<p class="eyebrow">ALPHAINTELLIGENCE CAPITAL · DAILY BRIEF</p>'
            f'<h1>{escape(headline)}</h1>'
            f'<p class="subhead">{escape(summary_text)}</p>'
            f'<p class="meta-date">{escape(date_str)}</p>'
            '</section>'
        )

    def build_market_mood(self, sentiment_score: float, sentiment_label: str, spy_trend: str, desk_take: str) -> str:
        return (
            '<section class="card">'
            '<h2>Market Mood</h2>'
            '<div class="stats-grid">'
            f'<div><span class="stat-label">Sentiment</span><strong>{sentiment_score:.1f}/100 ({escape(sentiment_label)})</strong></div>'
            f'<div><span class="stat-label">SPY Trend</span><strong>{escape(spy_trend)}</strong></div>'
            '</div>'
            f'<p class="compact">{escape(desk_take)}</p>'
            '</section>'
        )

    def build_indices_strip(self, index_perf: Dict[str, float]) -> str:
        if not index_perf:
            return ''
        chips = []
        for label, move in index_perf.items():
            polarity = 'pos' if move >= 0 else 'neg'
            chips.append(f'<span class="chip {polarity}">{escape(label)} {move:+.2f}%</span>')
        return '<section class="card"><h2>Indices</h2><div class="chip-row">' + ''.join(chips) + '</div></section>'

    def build_sector_table(self, sector_perf: List[Dict]) -> str:
        if not sector_perf:
            return ''
        rows = []
        for sector in sector_perf[:8]:
            name = escape(str(sector.get('sector', 'N/A')))
            change = escape(str(sector.get('changesPercentage', '0.00%')))
            rows.append(f'<tr><td>{name}</td><td>{change}</td></tr>')
        return (
            '<section class="card"><h2>Sectors</h2>'
            '<div class="table-wrap"><table><thead><tr><th>Sector</th><th>Move</th></tr></thead><tbody>'
            + ''.join(rows) +
            '</tbody></table></div></section>'
        )

    def build_movers(self, top_buys: List[Dict], top_sells: List[Dict], trending_entities: List[Dict]) -> str:
        items = []
        for idea in (top_buys or [])[:3]:
            items.append(f"<li><strong>{escape(str(idea.get('ticker', 'N/A')))}</strong> long setup · score {float(idea.get('score') or 0):.1f}</li>")
        for idea in (top_sells or [])[:3]:
            items.append(f"<li><strong>{escape(str(idea.get('ticker', 'N/A')))}</strong> risk-off setup · score {float(idea.get('score') or 0):.1f}</li>")
        if not items:
            for ent in (trending_entities or [])[:4]:
                items.append(f"<li><strong>{escape(str(ent.get('key', 'N/A')))}</strong> narrative volume {escape(str(ent.get('total_documents', 0)))} docs</li>")
        if not items:
            items.append('<li>No mover data available in this cycle.</li>')
        return '<section class="card"><h2>Movers</h2><ul>' + ''.join(items) + '</ul></section>'

    def build_headlines(self, market_news: List[Dict]) -> str:
        if not market_news:
            return ''
        items = []
        for item in market_news[:8]:
            title = escape(str(item.get('title', 'No Title')))
            url = escape(str(item.get('url', '#')))
            site = escape(str(item.get('site', 'News')))
            items.append(f'<li><a href="{url}">{title}</a><span class="source">{site}</span></li>')
        return '<section class="card"><h2>Headlines</h2><ul class="link-list">' + ''.join(items) + '</ul></section>'

    def build_events(self, earnings_cal: List[Dict], econ_calendar: List[Dict]) -> str:
        events = []
        for event in (earnings_cal or [])[:5]:
            events.append(f"<li><strong>{escape(str(event.get('date', '')))}</strong> Earnings: {escape(str(event.get('symbol', 'N/A')))}</li>")
        for event in (econ_calendar or [])[:5]:
            events.append(f"<li><strong>{escape(str(event.get('date', '')))}</strong> {escape(str(event.get('event', 'Economic Event')))}</li>")
        if not events:
            events.append('<li>No scheduled catalysts captured.</li>')
        return '<section class="card"><h2>Events</h2><ul>' + ''.join(events) + '</ul></section>'

    def build_disclaimer(self) -> str:
        return (
            '<section class="card disclaimer">'
            '<p><strong>Disclaimer:</strong> This content is for informational purposes only and is not investment advice.</p>'
            '<p><a href="https://alphaintelligence.capital/unsubscribe">Unsubscribe</a></p>'
            '</section>'
        )

    def render_newsletter_html(self, *, date_str: str, headline: str, summary: str, sentiment_score: float,
                               sentiment_label: str, spy_trend: str, desk_take: str, index_perf: Dict[str, float],
                               sector_perf: List[Dict], top_buys: List[Dict], top_sells: List[Dict],
                               trending_entities: List[Dict], market_news: List[Dict], earnings_cal: List[Dict],
                               econ_calendar: List[Dict]) -> str:
        """Render newsletter HTML using explicit section builders and the light template."""
        template = self._read_light_template()
        sections = {
            'hero_headline': self.build_hero_headline(date_str, headline, summary),
            'market_mood': self.build_market_mood(sentiment_score, sentiment_label, spy_trend, desk_take),
            'indices_strip': self.build_indices_strip(index_perf),
            'sector_table': self.build_sector_table(sector_perf),
            'movers': self.build_movers(top_buys, top_sells, trending_entities),
            'headlines': self.build_headlines(market_news),
            'events': self.build_events(earnings_cal, econ_calendar),
            'disclaimer': self.build_disclaimer(),
        }
        html = template
        for key, value in sections.items():
            html = html.replace('{{' + key + '}}', value)
        return html

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
