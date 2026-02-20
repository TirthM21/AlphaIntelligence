
import logging
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path
from html import escape
from urllib.parse import urlparse

import yfinance as yf
import yaml

from ..data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from ..data.finnhub_fetcher import FinnhubFetcher
from ..data.marketaux_fetcher import MarketauxFetcher
from ..data.fred_fetcher import FredFetcher
from ..data.price_service import PriceService
from ..ai.ai_agent import AIAgent
from .visualizer import ChartArtifact, MarketVisualizer

logger = logging.getLogger(__name__)


@dataclass
class SectionQCReport:
    section_title: str
    minimum_item_count: int
    minimum_source_diversity: int
    max_duplicate_topic_ratio: float
    freshness_threshold: float
    item_count: int
    source_diversity: int
    duplicate_topic_ratio: float
    freshness_ratio: float
    provider_attribution: Dict[str, int]
    passed: bool
    errors: List[str]


@dataclass
class SectionDataPlan:
    section_name: str
    primary_provider: str
    fallback_provider: str
    fetch_fn: Callable[[str], Any]
    render_fn: Callable[[Any], str]
    max_items: int
    freshness_sla: str

class NewsletterGenerator:
    """Generates a high-quality, professional daily market newsletter."""

    DEFAULT_PROVIDER_MATRIX = {
        "macro": ["fred", "fmp"],
        "headlines": ["finnhub", "marketaux", "fmp"],
        "sector_performance": ["fmp", "finnhub"],
        "prices": ["yfinance"],
    }
    SUPPORTED_PROVIDERS = {"fred", "fmp", "finnhub", "marketaux", "yfinance"}

    def __init__(self, portfolio_path: str = "./data/positions.json", config_path: str = "config.yaml"):
        try:
            self.fetcher = EnhancedFundamentalsFetcher()
        except Exception as e:
            logger.warning(f"EnhancedFundamentalsFetcher unavailable during init: {e}")
            self.fetcher = None
        self.finnhub = FinnhubFetcher()
        self.marketaux = MarketauxFetcher()
        self.fred = FredFetcher()
        self.ai_agent = AIAgent()
        self.visualizer = MarketVisualizer(output_dir="./data/charts")
        self.price_service = PriceService()
        self.portfolio_path = Path(portfolio_path)
        self.config_path = Path(config_path)
        self.newsletter_state_path = Path("./data/cache/newsletter_state.json")
        self.newsletter_config = self._load_newsletter_config()
        self.provider_matrix = self._build_provider_matrix(self.newsletter_config)
        self.provider_status = self._build_provider_status()
        self._log_provider_diagnostics()

    def _load_newsletter_config(self) -> Dict:
        if not self.config_path.exists():
            logger.warning(
                "Config path %s not found. Newsletter provider matrix will use defaults.",
                self.config_path,
            )
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to parse %s: %s", self.config_path, exc)
            return {}

    def _build_provider_matrix(self, config: Dict) -> Dict[str, List[str]]:
        configured = ((config or {}).get("newsletter") or {}).get("providers") or {}
        matrix: Dict[str, List[str]] = {}
        for section, defaults in self.DEFAULT_PROVIDER_MATRIX.items():
            raw = configured.get(section, defaults)
            if not isinstance(raw, list):
                logger.warning("newsletter.providers.%s must be a list. Falling back to defaults.", section)
                raw = defaults

            normalized = []
            for provider in raw:
                name = str(provider).strip().lower()
                if not name:
                    continue
                if name not in self.SUPPORTED_PROVIDERS:
                    logger.warning("Unsupported provider '%s' in newsletter.providers.%s; skipping.", name, section)
                    continue
                normalized.append(name)

            if not normalized:
                logger.warning("No valid providers configured for section '%s'; using defaults %s.", section, defaults)
                normalized = defaults.copy()
            matrix[section] = normalized
        return matrix

    def _build_provider_status(self) -> Dict[str, Dict]:
        fmp_key = None
        if self.fetcher and self.fetcher.fmp_fetcher:
            fmp_key = self.fetcher.fmp_fetcher.api_key
        elif self.fetcher and self.fetcher.fmp_available:
            fmp_key = os.getenv("FMP_API_KEY")

        status = {
            "finnhub": {"active": bool(self.finnhub.api_key), "missing_key": not bool(self.finnhub.api_key), "key_env": "FINNHUB_API_KEY"},
            "marketaux": {"active": bool(self.marketaux.api_key), "missing_key": not bool(self.marketaux.api_key), "key_env": "MARKETAUX_API_KEY"},
            "fred": {"active": bool(self.fred.api_key), "missing_key": not bool(self.fred.api_key), "key_env": "FRED_API_KEY"},
            "fmp": {"active": bool(fmp_key), "missing_key": not bool(fmp_key), "key_env": "FMP_API_KEY"},
            "yfinance": {"active": True, "missing_key": False, "key_env": None},
        }
        return status

    def _get_runtime_providers(self, section: str) -> List[str]:
        providers = self.provider_matrix.get(section, [])
        return [p for p in providers if self.provider_status.get(p, {}).get("active", False)]

    def _log_provider_diagnostics(self) -> None:
        logger.info("Newsletter provider diagnostics at startup:")
        for section in self.DEFAULT_PROVIDER_MATRIX:
            configured = self.provider_matrix.get(section, [])
            active = [p for p in configured if self.provider_status.get(p, {}).get("active", False)]
            logger.info(" - %s fallback order: %s", section, " -> ".join(configured) if configured else "(none)")
            logger.info(" - %s active providers: %s", section, ", ".join(active) if active else "(none)")

        missing = []
        for name, details in self.provider_status.items():
            if details.get("missing_key"):
                missing.append(f"{name} ({details.get('key_env')})")
        if missing:
            logger.warning("Newsletter providers with missing API keys: %s", ", ".join(missing))
        else:
            logger.info("Newsletter providers with API keys are fully configured.")


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

    def _load_prior_newsletter_text(self, output_path: str) -> str:
        """Load previous newsletter markdown for anti-repetition checks."""
        prev = self._latest_previous_newsletter(output_path)
        if not prev or not prev.exists():
            return ""
        try:
            return prev.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Unable to read prior newsletter '{prev}': {e}")
            return ""

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

    def _source_domain(self, url: str, site: str = "") -> str:
        parsed = urlparse((url or "").strip())
        domain = (parsed.netloc or "").lower().replace("www.", "")
        if domain:
            return domain
        return (site or "unknown").strip().lower()

    def _rank_and_dedupe_news(
        self,
        items: List[Dict],
        limit: int = 8,
        max_source_ratio: float = 0.4,
    ) -> List[Dict]:
        """Deterministically rank and dedupe headlines by relevance+recency."""
        now_ts = datetime.now().timestamp()
        normalized: List[Dict] = []
        seen_urls = set()
        seen_topic_keys = set()
        seen_domain_topic = set()

        for item in items:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if not title or not url:
                continue
            domain = self._source_domain(url, item.get("site", ""))
            topic_key = self._normalize_topic(title)
            if url in seen_urls or topic_key in seen_topic_keys or (domain, topic_key) in seen_domain_topic:
                continue

            raw_ts = item.get("datetime")
            try:
                ts = float(raw_ts or 0)
            except (TypeError, ValueError):
                ts = 0.0
            age_hours = max(0.0, (now_ts - ts) / 3600.0) if ts else 72.0
            recency = max(0.0, 120.0 - min(120.0, age_hours))
            relevance = min(40.0, len(title) * 0.45) + min(20.0, len((item.get("summary") or "")) * 0.04)
            score = recency + relevance

            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "site": (item.get("site") or "News").strip(),
                    "summary": (item.get("summary") or "").strip(),
                    "datetime": item.get("datetime"),
                    "domain": domain,
                    "topic_key": topic_key,
                    "score": round(score, 2),
                }
            )
            seen_urls.add(url)
            seen_topic_keys.add(topic_key)
            seen_domain_topic.add((domain, topic_key))

        ranked = sorted(normalized, key=lambda x: (-x.get("score", 0.0), x.get("title", ""), x.get("url", "")))
        if not ranked:
            return []

        source_counts: Dict[str, int] = {}
        selected: List[Dict] = []
        for item in ranked:
            source = item.get("domain") or item.get("site", "news").lower()
            projected_total = len(selected) + 1
            projected_count = source_counts.get(source, 0) + 1
            if projected_total >= 3 and (projected_count / projected_total) > max_source_ratio:
                continue
            source_counts[source] = projected_count
            selected.append(item)
            if len(selected) >= limit:
                break

        if len(selected) < min(limit, len(ranked)):
            existing_urls = {x.get("url") for x in selected}
            for item in ranked:
                if item.get("url") in existing_urls:
                    continue
                selected.append(item)
                if len(selected) >= limit:
                    break
        return selected

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

    @staticmethod
    def _format_macro_value(indicator: Dict) -> str:
        value = indicator.get("value")
        if value is None:
            return "N/A"
        unit = indicator.get("unit", "")
        if unit == "%":
            return f"{value:.2f}%"
        if unit == "binary":
            return "Recession" if value >= 0.5 else "Expansion"
        return f"{value:.2f}"

    def _build_macro_section_payload(self, bundle: Dict) -> Dict[str, List[str]]:
        source = bundle.get("source", "fallback")
        fetched_at = bundle.get("fetched_at", "unknown")
        indicators = bundle.get("indicators", {})
        derived = bundle.get("derived", {})

        def line_for(key: str) -> str:
            indicator = indicators.get(key, {})
            label = indicator.get("label", key)
            value = self._format_macro_value(indicator)
            date = indicator.get("date", "unknown")
            age = indicator.get("age_label", "age unknown")
            trend = indicator.get("trend", "stable")
            return f"- **{label}:** {value} ({trend}, obs {date}, {age})"

        lines = {
            "meta": [
                f"- **Macro Bundle Source:** `{source}`",
                f"- **Bundle Timestamp (UTC):** {fetched_at}",
            ],
            "snapshot": [],
            "rates": [],
            "risk": [],
            "warning": [],
        }

        warning = bundle.get("warning")
        if warning:
            lines["warning"].append(f"- ⚠️ {warning}")

        if source == "fallback":
            fallback = bundle.get("fallback_template", {})
            lines["snapshot"] = [f"- {msg}" for msg in fallback.get("macro_snapshot", [])]
            lines["rates"] = [f"- {msg}" for msg in fallback.get("rates_pulse", [])]
            lines["risk"] = [f"- {msg}" for msg in fallback.get("risk_regime", [])]
            return lines

        lines["snapshot"] = [
            line_for("fed_funds"),
            line_for("cpi_yoy"),
            line_for("unemployment_rate"),
            line_for("recession_proxy"),
        ]

        spread = derived.get("spread_2s10s")
        curve = derived.get("curve_regime", "unknown")
        spread_txt = f"{spread:.2f}pp" if isinstance(spread, (int, float)) else "N/A"
        lines["rates"] = [
            line_for("treasury_2y"),
            line_for("treasury_10y"),
            f"- **2s10s Spread:** {spread_txt} ({curve} curve)",
        ]

        recession = (indicators.get("recession_proxy") or {}).get("value")
        labor_trend = (indicators.get("unemployment_rate") or {}).get("trend", "stable")
        inflation_trend = (indicators.get("cpi_yoy") or {}).get("trend", "stable")
        risk_regime = "Balanced"
        if recession and recession >= 0.5:
            risk_regime = "Defensive"
        elif curve == "steep" and labor_trend in {"down", "stable"} and inflation_trend != "up":
            risk_regime = "Constructive"
        elif curve == "inverted" or labor_trend == "up":
            risk_regime = "Cautious"

        lines["risk"] = [
            f"- **Risk Regime:** {risk_regime}",
            f"- **Curve/Labor/Inflation Read:** curve={curve}, labor trend={labor_trend}, inflation trend={inflation_trend}.",
        ]
        return lines

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

        section_reports = self._run_section_qc_suite(markdown)
        providers = self._extract_provider_attribution(markdown)
        report = {
            "heading_count": float(len(headings)),
            "duplicate_header_count": float(len(dupes)),
            "source_count": float(unique_sources),
            "duplicate_topic_ratio": round(duplicate_ratio, 3),
            "provider_attribution": providers,
            "section_qc": [asdict(r) for r in section_reports],
            "section_qc_failures": float(sum(1 for r in section_reports if not r.passed)),
        }
        return len(errors) == 0, report, errors

    def _build_section_qc_fallback(self, section_title: str, report: SectionQCReport) -> List[str]:
        provider_bits = ", ".join(
            f"{name}: {count}" for name, count in sorted(report.provider_attribution.items(), key=lambda x: (-x[1], x[0]))
        ) or "No provider attribution available"
        return [
            f"## {section_title}",
            (
                "- Section quality checks flagged this section for low confidence "
                f"({', '.join(report.errors) if report.errors else 'unknown_qc_issue'})."
            ),
            (
                f"- QC snapshot: items={report.item_count}/{report.minimum_item_count}, "
                f"source_diversity={report.source_diversity}/{report.minimum_source_diversity}, "
                f"duplicate_topic_ratio={report.duplicate_topic_ratio:.2f}/{report.max_duplicate_topic_ratio:.2f}, "
                f"freshness_ratio={report.freshness_ratio:.2f}/{report.freshness_threshold:.2f}."
            ),
            f"- Provider attribution: {provider_bits}.",
            "- Action: defer to other sections while this feed is revalidated.",
            "",
        ]

    def _extract_provider_attribution(self, section_markdown: str) -> Dict[str, int]:
        providers: Dict[str, int] = {}
        for provider in re.findall(r"—\s*\*([^*]+)\*", section_markdown):
            key = provider.strip()
            if not key:
                continue
            providers[key] = providers.get(key, 0) + 1
        return providers

    def _parse_sections(self, markdown: str) -> List[Tuple[str, str]]:
        matches = list(re.finditer(r"^##\s+(.+)$", markdown, flags=re.MULTILINE))
        sections: List[Tuple[str, str]] = []
        for idx, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            body = markdown[start:end].rstrip() + "\n"
            sections.append((title, body))
        return sections

    def _resolve_section_qc_thresholds(self, section_title: str) -> Tuple[int, int, float, float]:
        normalized = section_title.lower()
        if "top headlines" in normalized:
            return 3, 2, 0.45, 0.5
        if "portfolio-specific news" in normalized or "earnings radar" in normalized:
            return 2, 1, 0.6, 0.3
        if "today's events" in normalized or "new since yesterday" in normalized:
            return 1, 1, 0.75, 0.2
        return 1, 1, 0.75, 0.2

    def _run_section_qc(self, section_title: str, section_markdown: str) -> SectionQCReport:
        minimum_item_count, minimum_source_diversity, max_duplicate_topic_ratio, freshness_threshold = (
            self._resolve_section_qc_thresholds(section_title)
        )
        item_titles = re.findall(r"^-\s+\[([^\]]+)\]", section_markdown, flags=re.MULTILINE)
        if not item_titles:
            item_titles = re.findall(r"^-\s+\*\*([^:*\n]+)", section_markdown, flags=re.MULTILINE)
        item_count = len(item_titles)
        provider_attribution = self._extract_provider_attribution(section_markdown)
        source_diversity = len(provider_attribution)

        topics = [self._normalize_topic(t) for t in item_titles if t.strip()]
        topic_total = len(topics)
        topic_unique = len(set(topics)) if topics else 0
        duplicate_topic_ratio = 0.0
        if topic_total:
            duplicate_topic_ratio = max(0.0, (topic_total - topic_unique) / topic_total)
        freshness_ratio = 0.0
        if topic_total:
            freshness_ratio = topic_unique / topic_total

        errors: List[str] = []
        if item_count < minimum_item_count:
            errors.append("minimum_item_count")
        if source_diversity < minimum_source_diversity:
            errors.append("minimum_source_diversity")
        if topic_total >= 2 and duplicate_topic_ratio > max_duplicate_topic_ratio:
            errors.append("max_duplicate_topic_ratio")
        if topic_total >= 1 and freshness_ratio < freshness_threshold:
            errors.append("freshness_threshold")

        return SectionQCReport(
            section_title=section_title,
            minimum_item_count=minimum_item_count,
            minimum_source_diversity=minimum_source_diversity,
            max_duplicate_topic_ratio=max_duplicate_topic_ratio,
            freshness_threshold=freshness_threshold,
            item_count=item_count,
            source_diversity=source_diversity,
            duplicate_topic_ratio=round(duplicate_topic_ratio, 3),
            freshness_ratio=round(freshness_ratio, 3),
            provider_attribution=provider_attribution,
            passed=len(errors) == 0,
            errors=errors,
        )

    def _run_section_qc_suite(self, markdown: str) -> List[SectionQCReport]:
        section_reports: List[SectionQCReport] = []
        for section_title, section_markdown in self._parse_sections(markdown):
            section_reports.append(self._run_section_qc(section_title, section_markdown))
        return section_reports

    def _apply_section_qc_fallbacks(self, markdown: str) -> Tuple[str, List[SectionQCReport]]:
        sections = self._parse_sections(markdown)
        reports: List[SectionQCReport] = []
        rebuilt_chunks: List[str] = []
        for section_title, section_md in sections:
            report = self._run_section_qc(section_title, section_md)
            reports.append(report)
            if report.passed:
                rebuilt_chunks.append(section_md.rstrip())
            else:
                logger.warning(
                    "Section QC failed for '%s' (%s): items=%s, source_diversity=%s, duplicate_topic_ratio=%.3f, freshness_ratio=%.3f",
                    section_title,
                    ",".join(report.errors),
                    report.item_count,
                    report.source_diversity,
                    report.duplicate_topic_ratio,
                    report.freshness_ratio,
                )
                rebuilt_chunks.append("\n".join(self._build_section_qc_fallback(section_title, report)).rstrip())
        prefix = re.split(r"^##\s+.+$", markdown, maxsplit=1, flags=re.MULTILINE)[0].rstrip()
        body = "\n\n".join(chunk for chunk in rebuilt_chunks if chunk.strip())
        assembled = "\n\n".join(x for x in [prefix, body] if x).rstrip() + "\n"
        return assembled, reports

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
        market_news = []
        portfolio = self.load_portfolio()
        portfolio_tickers = [p['ticker'] for p in portfolio]
        portfolio_news = []
        econ_calendar = []
        trending_entities = []
        earnings_cal = []
        ipo_calendar = []
        sentiment_snaps = {"top_buys": [], "top_sells": []}
        
        # 2. Section registry: primary + fallback provider execution per section
        fmp_fetcher = self.fetcher.fmp_fetcher if (self.fetcher and self.fetcher.fmp_available and self.fetcher.fmp_fetcher) else None
        section_status: Dict[str, Dict[str, str]] = {}
        headline_providers = self._get_runtime_providers("headlines")
        price_providers = self._get_runtime_providers("prices")

        def _has_data(data: Any) -> bool:
            return data not in (None, {}, [])

        def _diag_renderer(payload: Any) -> str:
            if isinstance(payload, list):
                return f"items={len(payload)}"
            if isinstance(payload, dict):
                return f"keys={len(payload.keys())}"
            return f"type={type(payload).__name__}"

        # Canonical FRED macro bundle for macro/rates/risk sections
        macro_bundle = {}
        macro_render = {"meta": [], "snapshot": [], "rates": [], "risk": [], "warning": []}
        date_str = datetime.now().strftime('%B %d, %Y')
        try:
            logger.info("Fetching canonical FRED macro bundle...")
            macro_bundle = self.fred.fetch_canonical_macro_bundle()
            macro_render = self._build_macro_section_payload(macro_bundle)
        except Exception as e:
            logger.error(f"FRED bundle fetch failed: {e}")
            macro_bundle = {
                "source": "fallback",
                "fetched_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "warning": "Canonical macro bundle failed to load.",
                "fallback_template": {
                    "macro_snapshot": ["Macro inputs unavailable; maintain neutral positioning."],
                    "rates_pulse": ["Rates inputs unavailable; avoid aggressive duration bets."],
                    "risk_regime": ["Risk regime unavailable; tighten risk controls."],
                },
            }
            macro_render = self._build_macro_section_payload(macro_bundle)

        def _fetch_macro_rates(provider: str) -> Dict[str, Any]:
            if provider == "fred":
                return {
                    "econ_data": macro_bundle,
                    "macro_panel": macro_render,
                    "macro_panel_fallback": macro_bundle.get("warning", ""),
                }
            if provider == "fmp" and fmp_fetcher:
                return {
                    "econ_data": fmp_fetcher.fetch_economic_data() or {},
                    "macro_panel": macro_render,
                    "macro_panel_fallback": "",
                }
            return {}

        def _fetch_economic_calendar(provider: str) -> List[Dict]:
            if provider == "fred":
                return self.finnhub.fetch_economic_calendar() if self.finnhub.api_key else []
            if provider == "fmp" and fmp_fetcher:
                return fmp_fetcher.fetch_economic_calendar(days_forward=3) or []
            return []

        def _fetch_market_headlines(provider: str) -> List[Dict]:
            if provider == "finnhub":
                return self.finnhub.fetch_top_market_news(limit=10, category="general") or []
            if provider == "marketaux":
                return self.marketaux.fetch_market_news(limit=10) or []
            if provider == "fmp" and fmp_fetcher:
                raw = fmp_fetcher.fetch_market_news(limit=10) or []
                return [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "site": item.get("site") or "FMP News",
                        "summary": item.get("text") or item.get("summary", ""),
                        "datetime": item.get("publishedDate") or item.get("date"),
                    }
                    for item in raw
                    if item.get("title")
                ]
            return []

        def _fetch_market_sentiment(provider: str) -> Dict[str, Any]:
            if provider == "finnhub":
                return self.finnhub.fetch_market_sentiment_proxy() or {}
            return {"score": 50.0, "label": "Neutral", "components": []}

        def _fetch_earnings_calendar(provider: str) -> List[Dict]:
            if provider == "finnhub":
                return self.finnhub.fetch_earnings_calendar_standardized(days_forward=5, limit=20) or []
            return []

        def _fetch_sector_performance(provider: str) -> List[Dict]:
            if provider == "fmp" and fmp_fetcher:
                return fmp_fetcher.fetch_sector_performance() or []
            return []

        def _fetch_movers(provider: str) -> List[Dict]:
            if provider == "finnhub":
                return self.finnhub.fetch_notable_movers(limit=6) or []
            return []

        def _fetch_fundamentals_snippets(provider: str) -> List[Dict]:
            snippets: List[Dict] = []
            if provider == "fmp" and self.fetcher:
                for ticker in portfolio_tickers[:3]:
                    if self.fetcher.fetch_all_metrics(ticker):
                        snippets.append({"symbol": ticker, "summary": "FMP fundamentals retrieved"})
            elif provider == "finnhub" and self.finnhub.api_key:
                for ticker in portfolio_tickers[:3]:
                    if self.finnhub.fetch_basic_financials(ticker):
                        snippets.append({"symbol": ticker, "summary": "Finnhub basic financials retrieved"})
            return snippets

        def _execute_section_plan(plan: SectionDataPlan) -> Any:
            for provider in [plan.primary_provider, plan.fallback_provider]:
                try:
                    data = plan.fetch_fn(provider)
                    if _has_data(data):
                        if isinstance(data, list):
                            data = data[:plan.max_items]
                        section_status[plan.section_name] = {"status": "success", "provider": provider}
                        return data
                except Exception as e:
                    logger.error(
                        "Section %s provider %s failed: %s",
                        plan.section_name,
                        provider,
                        e,
                    )

            section_status[plan.section_name] = {"status": "failed", "provider": "none"}
            return [] if plan.section_name not in {"macro", "rates", "market_sentiment"} else {}

        section_plans: List[SectionDataPlan] = [
            SectionDataPlan("macro", "fred", "fmp", _fetch_macro_rates, _diag_renderer, 6, "24h"),
            SectionDataPlan("rates", "fred", "fmp", _fetch_macro_rates, _diag_renderer, 4, "24h"),
            SectionDataPlan("economic_calendar", "fred", "fmp", _fetch_economic_calendar, _diag_renderer, 8, "4h"),
            SectionDataPlan("market_headlines", "finnhub", "marketaux", _fetch_market_headlines, _diag_renderer, 8, "1h"),
            SectionDataPlan("market_sentiment", "finnhub", "fmp", _fetch_market_sentiment, _diag_renderer, 1, "15m"),
            SectionDataPlan("earnings_calendar", "finnhub", "fmp", _fetch_earnings_calendar, _diag_renderer, 8, "12h"),
            SectionDataPlan("sector_performance", "fmp", "finnhub", _fetch_sector_performance, _diag_renderer, 11, "1h"),
            SectionDataPlan("movers", "fmp", "finnhub", _fetch_movers, _diag_renderer, 6, "15m"),
            SectionDataPlan("fundamentals_snippets", "fmp", "finnhub", _fetch_fundamentals_snippets, _diag_renderer, 3, "24h"),
        ]

        section_results: Dict[str, Any] = {}
        for plan in section_plans:
            section_results[plan.section_name] = _execute_section_plan(plan)

        # Extra Finnhub data for newsletter sections (when key is available)
        if self.finnhub.api_key:
            try:
                ipo_calendar = self.finnhub.fetch_ipo_calendar(days_forward=10, limit=10)
                buy_tickers = [x.get('ticker', '') for x in (top_buys or []) if x.get('ticker')]
                sell_tickers = [x.get('ticker', '') for x in (top_sells or []) if x.get('ticker')]
                sentiment_snaps = self.finnhub.fetch_top_ticker_sentiment_snapshots(
                    top_buy_tickers=buy_tickers,
                    top_sell_tickers=sell_tickers,
                    per_side=3,
                )
                if portfolio_tickers:
                    for t in portfolio_tickers[:3]:
                        for item in (self.finnhub.fetch_company_news(t) or [])[:2]:
                            portfolio_news.append({
                                'title': item.get('headline') or item.get('title'),
                                'symbol': t,
                                'url': item.get('url'),
                                'summary': item.get('summary', ''),
                            })
            except Exception as e:
                logger.error(f"Finnhub supplemental fetch failed: {e}")

        macro_payload = section_results.get("macro") or {}
        econ_data = macro_payload.get("econ_data", {})
        macro_panel = macro_payload.get("macro_panel", {})
        macro_panel_fallback = macro_payload.get("macro_panel_fallback", "")
        market_news = section_results.get("market_headlines") or []
        earnings_cal = section_results.get("earnings_calendar") or []
        econ_calendar = section_results.get("economic_calendar") or []

        if self.marketaux.api_key:
            try:
                logger.info("Fetching Marketaux trending entities...")
                trending_entities = self.marketaux.fetch_trending_entities()
            except Exception as e:
                logger.error(f"Marketaux fetch failed: {e}")

        if fmp_fetcher:
            try:
                # FMP used for DAILY news as requested
                if self.fetcher.fmp_fetcher.is_endpoint_cooldown_active('stock_news'):
                    logger.warning("FMP cooldown active for market news section; switching to fallback provider.")
                else:
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
                    if self.fetcher.fmp_fetcher.is_endpoint_cooldown_active('economic_calendar'):
                        logger.warning("FMP cooldown active for economic calendar; skipping to existing/fallback sources.")
                    else:
                        econ_calendar = self.fetcher.fmp_fetcher.fetch_economic_calendar(days_forward=3)
                
                # FMP for portfolio news DAILY 
                if not portfolio_news and portfolio_tickers:
                    if self.fetcher.fmp_fetcher.is_endpoint_cooldown_active('stock_news'):
                        logger.warning("FMP cooldown active for portfolio news; switching to fallback provider.")
                    else:
                        portfolio_news = self.fetcher.fmp_fetcher.fetch_stock_news(portfolio_tickers, limit=3)
            except Exception as e:
                logger.error(f"Failed to fetch FMP supplemental data: {e}")

        if not market_news and "yfinance" in headline_providers:
            try:
                logger.info("Falling back to basic yfinance news...")
                for t in ['SPY', 'QQQ', 'DIA']:
                    stock = yf.Ticker(t)
                    for n in (stock.news or [])[:2]:
                        market_news.append({
                            'title': n.get('title'),
                            'url': n.get('link'),
                            'site': 'Yahoo Finance',
                            'summary': '',
                        })
            except Exception as e:
                logger.error(f"News fallback failed: {e}")

        market_news = self._rank_and_dedupe_news(market_news, limit=12, max_source_ratio=0.4)
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
        sector_perf = section_results.get("sector_performance") or []
        cap_perf = {}
        index_perf = {}
        if sector_perf:
            try:
                if self.fetcher.fmp_fetcher.is_endpoint_cooldown_active('sector-performance'):
                    logger.warning("FMP cooldown active for sector performance section; using fallback providers.")
                else:
                    sector_perf = self.fetcher.fmp_fetcher.fetch_sector_performance()
                # Sort sectors by performance
                if sector_perf:
                    sector_perf = sorted(sector_perf, key=lambda x: float(x.get('changesPercentage', '0').replace('%','')), reverse=True)
            except Exception as e:
                logger.warning(f"Sector perf sort failed: {e}")
        
        # Cap Segment Analysis (SPY, MDY, IWM)
        if "yfinance" in price_providers:
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
        sentiment_proxy = section_results.get("market_sentiment") or {"score": 50.0, "label": "Neutral", "components": []}
        notable_movers = section_results.get("movers") or []
        if not isinstance(notable_movers, list):
            try:
                notable_movers = list(notable_movers)
            except Exception:
                notable_movers = []
        if self.finnhub.api_key:
            try:
                market_snapshot = self.finnhub.fetch_major_index_snapshot()
                for item in market_snapshot.values():
                    index_perf[item.get('label', item.get('symbol', 'Index'))] = item.get('change_pct', 0.0)
            except Exception as e:
                logger.error(f"Finnhub snapshot check failed: {e}")

        if not index_perf and "yfinance" in price_providers:
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

        # 5. Build content (institutional style, concise + actionable)
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

        def _ai_section_insight(title: str, payload: Dict[str, Any], fallback: str) -> str:
            if not getattr(self.ai_agent, 'client', None):
                return fallback
            prompt = (
                f"Act as a buy-side strategist. Write exactly 2 concise bullet points for '{title}'. "
                "Use only facts present in the payload. Keep it tactical and risk-aware. "
                "No markdown header, no hype, no trivia. Return only bullets.\n\n"
                f"Payload:\n{json.dumps(payload, default=str)[:4000]}"
            )
            ai_text = self.ai_agent._call_ai(prompt, low_temp=True)
            if not ai_text:
                return fallback
            lines = [ln.strip() for ln in ai_text.splitlines() if ln.strip()]
            bullets = [ln if ln.startswith('-') else f"- {ln.lstrip('•- ')}" for ln in lines[:2]]
            return "\n".join(bullets) if bullets else fallback

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
        sentiment_proxy_score = _safe_num(sentiment_proxy.get('score', 50.0), 50.0)
        sentiment_proxy_label = sentiment_proxy.get('label', 'Neutral')
        snapshot_headline = f"Tape check: sentiment proxy at {sentiment_proxy_score:.1f}/100 ({sentiment_proxy_label}) with mixed cross-asset leadership."
        if notable_movers and any(isinstance(x, dict) for x in notable_movers):
            valid_movers = [x for x in notable_movers if isinstance(x, dict)]
            dominant = max(valid_movers, key=lambda x: abs(_safe_num(x.get('change_pct', 0.0))))
            snapshot_headline = (
                f"Tape check: {dominant.get('symbol')} is leading notable flow ({dominant.get('change_pct', 0.0):+,.2f}%), "
                f"while sentiment proxy sits at {sentiment_proxy_score:.1f}/100 ({sentiment_proxy_label})."
            )
            snapshot_headline = ''.join(snapshot_headline)
        content.append(f"- **Headline:** {snapshot_headline}")

        strip = []
        for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
            data = market_snapshot.get(symbol, {})
            if data:
                change = _safe_num(data.get('change_pct', 0.0), 0.0)
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
                m_change = _safe_num(m.get('change_pct', 0.0), 0.0)
                mover_bits.append(f"{m.get('symbol')} ({m_change:+,.2f}%: {m.get('reason', 'Notable move')})")
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
        quick_view_payload = {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'index_perf': index_perf,
            'notable_movers': notable_movers[:3],
        }
        default_view = '- Market internals are mixed; stay selective with entry timing.\n- Respect volatility in single-name moves around macro and earnings catalysts.'
        content.append(_ai_section_insight('Market Overview Quick View', quick_view_payload, default_view))
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

        # --- SECTION: CANONICAL MACRO BUNDLE ---
        content.append("### Macro Snapshot")
        content.extend(macro_render.get("meta", []))
        content.extend(macro_render.get("warning", []))
        content.extend(macro_render.get("snapshot", []))
        content.append("")

        content.append("### Rates Pulse")
        content.extend(macro_render.get("meta", []))
        content.extend(macro_render.get("warning", []))
        content.extend(macro_render.get("rates", []))
        content.append("")

        content.append("### Risk Regime")
        content.extend(macro_render.get("meta", []))
        content.extend(macro_render.get("warning", []))
        content.extend(macro_render.get("risk", []))
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
                snap = next((x for x in sentiment_snaps.get("top_buys", []) if x.get("ticker") == ticker), None)
                if snap:
                    content.append(
                        f"   - Finnhub Signal Check: bullish {snap.get('bullish_pct', 0.0):.1f}% | bearish {snap.get('bearish_pct', 0.0):.1f}% | buzz {snap.get('buzz', 0.0):.2f}"
                    )
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
                snap = next((x for x in sentiment_snaps.get("top_sells", []) if x.get("ticker") == ticker), None)
                if snap:
                    content.append(
                        f"   - Finnhub Signal Check: bullish {snap.get('bullish_pct', 0.0):.1f}% | bearish {snap.get('bearish_pct', 0.0):.1f}% | news score {snap.get('company_news_score', 0.0):.2f}"
                    )
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
                mover_change = _safe_num(mover.get('change_pct', 0.0), 0.0)
                direction = '▲' if mover_change >= 0 else '▼'
                content.append(
                    f"- **{mover.get('symbol', 'N/A')}** {direction} {mover_change:+,.2f}% — {mover.get('reason', 'Notable move')}"
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

        # --- SECTION: ECONOMIC CALENDAR & DESK PLAYBOOK ---
        content.append("## 6) Economic Calendar (Next 72 Hours)")
        if econ_calendar:
            for event in econ_calendar[:8]:
                date_val = event.get('date') or event.get('time') or event.get('datetime') or 'TBD'
                event_name = event.get('event') or event.get('title') or event.get('name') or 'Economic Event'
                impact = event.get('impact') or event.get('importance') or 'N/A'
                actual = event.get('actual') or event.get('value') or 'N/A'
                forecast = event.get('forecast') or event.get('estimate') or 'N/A'
                content.append(f"- **{date_val}** | {event_name} | Impact: {impact} | Actual: {actual} | Forecast: {forecast}")
        else:
            content.append("- No economic-calendar records available from configured providers for this run.")
        content.append("")

        content.append("## 7) Institutional Desk Playbook")
        macro_payload = {
            "spy_trend": spy_trend,
            "ad_ratio": ad_ratio,
            "pct_above_200": pct_above_200,
            "index_perf": index_perf,
            "sector_leaders": sector_perf[:3] if sector_perf else [],
            "sector_laggards": sector_perf[-3:] if sector_perf else [],
        }
        default_playbook = "- Keep gross exposure balanced until breadth confirms trend persistence.\n- Prioritize liquid leaders with clear catalyst windows and disciplined stop placement."
        content.append(_ai_section_insight("Institutional Desk Playbook", macro_payload, default_playbook))
        content.append("")

        # --- SECTION: TOP HEADLINES ---
        content.append("## 8) Top Headlines")
        headlines_intro = self._pick_fresh_text([
            "Finnhub-first headline tape ranked on relevance + recency, then constrained for source/topic concentration.",
            "Cross-source scan prioritizing fresh narratives over recycled headlines.",
            "Headline tape below reflects both macro and single-name dispersion."
        ], [r.get("headlines_intro", "") for r in recent_runs])
        content.append(headlines_intro)
        content.append("")
        if market_news:
            for item in market_news[:8]:
                title = item.get('title', 'No Title')
                url = item.get('url', '#')
                site = item.get('site', 'News')
                summary = item.get('summary', '')
                if summary:
                    content.append(f"- [{title}]({url}) — *{site}*\n  - {summary[:180].rstrip()}...")
                else:
                    content.append(f"- [{title}]({url}) — *{site}*")
        else:
            content.append("- Finnhub headline feed unavailable in this run; fallback feeds did not pass ranking/dedupe gates.")
        content.append("")

        if portfolio_news:
            content.append("## 9) Portfolio-Specific News")
        # Removed redundant sections (Portfolio News / Earnings Radar) - moved further down


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
                if macro_render.get("rates"):
                    content.append("- Optional rates addendum sourced from canonical macro bundle:")
                    content.extend(macro_render.get("rates", [])[:2])
                else:
                    content.append("- Canonical macro bundle unavailable; rates addendum suppressed.")
            elif section_name == "Earnings Spotlight":
                if earnings_cal:
                    for event in earnings_cal[:3]:
                        content.append(
                            f"- **{event.get('symbol', 'N/A')}** reports {event.get('date', '')} ({event.get('hour', 'TBD')})."
                        )
                else:
                    content.append("- Finnhub earnings calendar returned no high-confidence catalyst in this run.")
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
        content.append("## 11) Today's Events")
        if econ_calendar:
            for event in econ_calendar[:8]:
                date = event.get('date', '')
                event_time = event.get('time') or event.get('hour') or 'TBD'
                title = event.get('event', 'Economic Event')
                impact_tag = _event_impact_tag(event)
                content.append(f"- **{date} {event_time}** — {title} `[{impact_tag}]`")
        elif earnings_cal or ipo_calendar:
            for event in earnings_cal[:4]:
                content.append(
                    f"- **{event.get('date', '')} {event.get('hour', 'TBD')}** — {event.get('symbol', 'N/A')} earnings `[{_event_impact_tag({'event': 'earnings'})}]`"
                )
            for event in ipo_calendar[:3]:
                content.append(f"- **{event.get('date', '')}** — IPO watch: {event.get('symbol', 'N/A')} ({event.get('exchange', 'N/A')})")
        else:
            content.append("- Finnhub calendars unavailable for this run; no validated event tape to publish.")
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

        final_md, section_qc_reports = self._apply_section_qc_fallbacks(final_md)

        qc_ok, qc_report, qc_errors = self._run_newsletter_qc(final_md)
        qc_report["section_qc"] = [asdict(r) for r in section_qc_reports]
        qc_report["section_qc_failures"] = float(sum(1 for r in section_qc_reports if not r.passed))
        qc_report["provider_attribution"] = self._extract_provider_attribution(final_md)
        if not qc_ok:
            logger.warning(
                "Newsletter QC failed (%s). Report: headings=%s, duplicate_headers=%s, sources=%s, duplicate_topic_ratio=%.3f, section_failures=%s. Falling back to safe template.",
                ",".join(qc_errors),
                int(qc_report.get("heading_count", 0)),
                int(qc_report.get("duplicate_header_count", 0)),
                int(qc_report.get("source_count", 0)),
                qc_report.get("duplicate_topic_ratio", 0.0),
                int(qc_report.get("section_qc_failures", 0)),
            )
            final_md = self._build_qc_fallback_template(date_str)
        else:
            logger.info(
                "Newsletter QC passed: headings=%s, sources=%s, duplicate_topic_ratio=%.3f, section_failures=%s",
                int(qc_report.get("heading_count", 0)),
                int(qc_report.get("source_count", 0)),
                qc_report.get("duplicate_topic_ratio", 0.0),
                int(qc_report.get("section_qc_failures", 0)),
            )

        diagnostics_lines = ["", "## Internal Diagnostics"]
        for plan in section_plans:
            status_row = section_status.get(plan.section_name, {"status": "failed", "provider": "none"})
            diag_summary = plan.render_fn(section_results.get(plan.section_name))
            diagnostics_lines.append(
                f"- **{plan.section_name}**: {status_row['status']} via `{status_row['provider']}` "
                f"(primary={plan.primary_provider}, fallback={plan.fallback_provider}, "
                f"max_items={plan.max_items}, sla={plan.freshness_sla}, {diag_summary})"
            )
        final_md = final_md.rstrip() + "\n" + "\n".join(diagnostics_lines) + "\n"

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
            "optional_sections": optional_sections,
            "section_status": section_status
        })
        state["runs"] = state_runs[-5:]
        self._save_newsletter_state(state)
            
        logger.info(f"Professional Newsletter generated at {output_path}")
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
        trending_window_minutes = 1440
        min_trending_entities = 3
        min_documents = 3
        max_entity_age_minutes = 72 * 60
        min_sentiment_confidence = 0.20
        econ_cal = []
        if self.fetcher and self.fetcher.fmp_available:
            econ_cal = self.fetcher.fmp_fetcher.fetch_economic_calendar(days_forward=30) 

        # Enhanced AI Thesis for Quarterly
        ai_thesis = "Institutional compounder selection focused on capital efficiency and moat depth."
        if self.ai_agent.client:
            prompt = f"Act as a hedge fund macro strategist. Provide a 4-sentence quarterly investment thesis for Q{q} {year} focusing on inflation, interest rates, and sector leadership. Be precise and sophisticated."
            ai_thesis = self.ai_agent._call_ai(prompt)

        state = self._load_newsletter_state()
        quarterly_history = state.get("quarterly_diagnostics", []) if isinstance(state, dict) else []
        prior_quarter_metrics = quarterly_history[-1] if isinstance(quarterly_history, list) and quarterly_history else {}

        def _safe_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _format_pct(value: Optional[float]) -> str:
            return f"{value:.2%}" if value is not None else "N/A"

        def _format_num(value: Optional[float], decimals: int = 2) -> str:
            return f"{value:.{decimals}f}" if value is not None else "N/A"

        sector_allocations: Dict[str, float] = {}
        equity_alloc, etf_alloc = 0.0, 0.0
        for ticker, allocation in portfolio.allocations.items():
            alloc = float(allocation)
            if ticker in top_stocks:
                sector = top_stocks.get(ticker, {}).get("sector") or "Unknown"
                equity_alloc += alloc
            elif ticker in top_etfs:
                sector = top_etfs.get(ticker, {}).get("theme") or "Thematic"
                etf_alloc += alloc
            else:
                sector = "Unknown"
            sector_allocations[sector] = sector_allocations.get(sector, 0.0) + alloc

        current_metrics: Dict[str, Optional[float]] = {
            "sector_concentration": _safe_float(getattr(portfolio, "sector_concentration", None)),
            "top3_weight": None,
            "sector_dispersion": None,
            "regime_sentiment": None,
            "high_impact_events": None,
            "equity_exposure": equity_alloc if portfolio.allocations else None,
            "etf_exposure": etf_alloc if portfolio.allocations else None,
            "top_conviction_score": None,
        }

        sorted_alloc = sorted(portfolio.allocations.items(), key=lambda x: x[1], reverse=True)
        if sorted_alloc:
            current_metrics["top3_weight"] = sum(float(a) for _, a in sorted_alloc[:3])

            top_conviction = sorted_alloc[:5]
            total_top_alloc = sum(float(a) for _, a in top_conviction)
            if total_top_alloc > 0:
                weighted_score = 0.0
                for ticker, alloc in top_conviction:
                    meta = top_stocks.get(ticker, top_etfs.get(ticker, {}))
                    weighted_score += float(alloc) * float(meta.get("score", 0))
                current_metrics["top_conviction_score"] = weighted_score / total_top_alloc

        if len(sector_allocations) > 1:
            weights = list(sector_allocations.values())
            mean_weight = sum(weights) / len(weights)
            variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
            current_metrics["sector_dispersion"] = variance ** 0.5

        if trending:
            sentiments: List[float] = []
            for item in trending:
                sentiment = _safe_float(item.get("sentiment_avg"))
                if sentiment is not None:
                    sentiments.append(sentiment)
            if sentiments:
                current_metrics["regime_sentiment"] = sum(sentiments) / len(sentiments)

        if econ_cal:
            high_impact_count = sum(1 for event in econ_cal if str(event.get("impact", "")).lower() == "high")
            current_metrics["high_impact_events"] = float(high_impact_count)

        strategy_universe = {
            str(sym).upper()
            for sym in list(portfolio.allocations.keys()) + list(top_stocks.keys()) + list(top_etfs.keys())
            if str(sym).strip()
        }
        supported_exchanges = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}
        supported_instruments = {"stock", "etf", "index", "adr", "equity"}

        def _entity_timestamp(item: Dict[str, Any]) -> Optional[datetime]:
            for key in ("updated_at", "published_at", "last_seen_at", "most_recent_document_at", "created_at"):
                value = item.get(key)
                if not value:
                    continue
                normalized = str(value).strip().replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(normalized)
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return None

        def _entity_confidence(item: Dict[str, Any]) -> float:
            for key in ("sentiment_confidence", "confidence", "score", "match_score"):
                value = _safe_float(item.get(key))
                if value is not None:
                    return value
            return 0.0

        def _in_supported_universe(item: Dict[str, Any]) -> bool:
            key = str(item.get("key", "")).upper().strip()
            if not key:
                return False
            if key in strategy_universe:
                return True

            exchange = str(item.get("exchange") or item.get("exchange_name") or "").upper().strip()
            instrument = str(item.get("instrument_type") or item.get("type") or item.get("asset_type") or "").lower().strip()
            symbol_like = bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", key))
            return symbol_like and ((exchange in supported_exchanges) or (instrument in supported_instruments))

        def _quality_labels(item: Dict[str, Any], age_minutes: float, confidence: float) -> str:
            labels: List[str] = []
            doc_count = int(_safe_float(item.get("total_documents")) or 0)
            sentiment = _safe_float(item.get("sentiment_avg")) or 0.0
            if doc_count >= 8:
                labels.append("High mention velocity")
            if sentiment >= 0.25:
                labels.append("Positive sentiment skew")
            elif sentiment <= -0.25:
                labels.append("Negative sentiment skew")
            if age_minutes <= 120:
                labels.append("Fresh coverage burst")
            if confidence < (min_sentiment_confidence + 0.10):
                labels.append("Low-confidence mention")
            return ", ".join(labels[:2]) if labels else "Balanced mention profile"

        filtered_trending_entities: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc)
        for item in trending:
            if not _in_supported_universe(item):
                continue

            docs = int(_safe_float(item.get("total_documents")) or 0)
            if docs < min_documents:
                continue

            ts = _entity_timestamp(item)
            if ts is None:
                continue
            age_minutes = (now_utc - ts.astimezone(timezone.utc)).total_seconds() / 60.0
            if age_minutes > max_entity_age_minutes:
                continue

            confidence = _entity_confidence(item)
            if confidence < min_sentiment_confidence:
                continue

            enriched = dict(item)
            enriched["_age_minutes"] = age_minutes
            enriched["_confidence"] = confidence
            enriched["_label"] = _quality_labels(item, age_minutes, confidence)
            filtered_trending_entities.append(enriched)

        filtered_trending_entities.sort(
            key=lambda ent: (
                -int(_safe_float(ent.get("total_documents")) or 0),
                -(_safe_float(ent.get("sentiment_avg")) or 0.0),
                ent.get("_age_minutes", float("inf")),
            )
        )

        regime_score = current_metrics.get("regime_sentiment")
        regime_label = "Neutral"
        if regime_score is not None:
            if regime_score >= 0.15:
                regime_label = "Constructive"
            elif regime_score <= -0.15:
                regime_label = "Defensive"

        diagnostic_rows: List[Dict[str, str]] = []

        def _add_row(metric: str, current: str, prior: str, interpretation: str) -> None:
            if current == "N/A":
                return
            diagnostic_rows.append({
                "metric": metric,
                "current": current,
                "prior": prior,
                "interpretation": interpretation,
            })

        conc = current_metrics.get("sector_concentration")
        if conc is not None:
            prior_conc = _safe_float(prior_quarter_metrics.get("sector_concentration")) if prior_quarter_metrics else None
            conc_prior_str = _format_num(prior_conc, 3)
            conc_view = "Concentration risk elevated" if conc > 0.18 else "Diversification profile stable"
            _add_row(
                "Portfolio Concentration (Herfindahl)",
                _format_num(conc, 3),
                conc_prior_str,
                conc_view,
            )

        top3 = current_metrics.get("top3_weight")
        if top3 is not None:
            prior_top3 = _safe_float(prior_quarter_metrics.get("top3_weight")) if prior_quarter_metrics else None
            drift = "Top positions dominate risk budget" if top3 > 0.45 else "Top positions remain balanced"
            _add_row(
                "Top-3 Position Weight",
                _format_pct(top3),
                _format_pct(prior_top3),
                drift,
            )

        dispersion = current_metrics.get("sector_dispersion")
        if dispersion is not None:
            prior_dispersion = _safe_float(prior_quarter_metrics.get("sector_dispersion")) if prior_quarter_metrics else None
            interp = "Sector weights are widely dispersed" if dispersion > 0.08 else "Sector exposures are tightly clustered"
            _add_row(
                "Sector Allocation Dispersion (σ)",
                _format_pct(dispersion),
                _format_pct(prior_dispersion),
                interp,
            )

        regime_sent = current_metrics.get("regime_sentiment")
        if regime_sent is not None:
            prior_regime = _safe_float(prior_quarter_metrics.get("regime_sentiment")) if prior_quarter_metrics else None
            regime_interp = "Newsflow supports constructive risk regime" if regime_sent >= 0 else "Newsflow points to cautious regime"
            _add_row(
                "Regime Sentiment Indicator",
                _format_num(regime_sent, 2),
                _format_num(prior_regime, 2),
                regime_interp,
            )

        high_impact = current_metrics.get("high_impact_events")
        if high_impact is not None:
            prior_events = _safe_float(prior_quarter_metrics.get("high_impact_events")) if prior_quarter_metrics else None
            events_interp = "Dense high-impact macro calendar" if high_impact >= 3 else "Macro event pressure remains moderate"
            _add_row(
                "High-Impact Macro Events (30d)",
                _format_num(high_impact, 0),
                _format_num(prior_events, 0),
                events_interp,
            )

        conviction_score = current_metrics.get("top_conviction_score")
        if conviction_score is not None:
            prior_conviction = _safe_float(prior_quarter_metrics.get("top_conviction_score")) if prior_quarter_metrics else None
            conviction_interp = "Top convictions retain high factor quality" if conviction_score >= 75 else "Top conviction quality has softened"
            _add_row(
                "Top-5 Conviction Score (Weighted)",
                _format_num(conviction_score, 1),
                _format_num(prior_conviction, 1),
                conviction_interp,
            )

        equity_exp = current_metrics.get("equity_exposure")
        if equity_exp is not None:
            prior_equity = _safe_float(prior_quarter_metrics.get("equity_exposure")) if prior_quarter_metrics else None
            _add_row(
                "Single-Name Equity Exposure",
                _format_pct(equity_exp),
                _format_pct(prior_equity),
                "Core stock exposure defines primary return engine" if equity_exp >= 0.55 else "ETF sleeve carries larger portfolio influence",
            )

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
        
        # Deterministic diagnostics block (no AI trivia fallback)
        content.append("## 🧪 Portfolio & Macro Signal Diagnostics")
        if diagnostic_rows:
            content.append("| Metric | Current Value | Prior Quarter | Interpretation |")
            content.append("|--------|---------------|---------------|----------------|")
            for row in diagnostic_rows:
                content.append(
                    f"| {row['metric']} | {row['current']} | {row['prior']} | {row['interpretation']} |"
                )
        else:
            content.append("*Insufficient data to render portfolio/macro diagnostics for this quarter.*")
        content.append("")

        # --- SECTION: MARKET TRENDING ---
        if len(filtered_trending_entities) >= min_trending_entities:
            content.append("## 🛸 Trending Institutional Interest")
            content.append(
                f"*Source: Marketaux trending entities (US), lookback {trending_window_minutes // 60}h, "
                f"filters: docs≥{min_documents}, age≤{max_entity_age_minutes // 60}h, confidence≥{min_sentiment_confidence:.2f}.*"
            )
            content.append("| Sector/Entity | Sentiment | Volume | Analysis |")
            content.append("|---------------|-----------|--------|----------|")
            for ent in filtered_trending_entities[:5]:
                content.append(
                    f"| {ent.get('key')} | {ent.get('sentiment_avg', 0):+.2f} | "
                    f"{ent.get('total_documents')} docs | {ent.get('_label')} |"
                )
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

        top_holdings_stats: List[Dict[str, Any]] = []
        for ticker, alloc in sorted_alloc[:10]:
            meta = top_stocks.get(ticker, top_etfs.get(ticker, {}))
            top_holdings_stats.append(
                {
                    "ticker": ticker,
                    "allocation_pct": round(float(alloc) * 100, 2),
                    "score": _safe_float(meta.get("score")),
                    "label": meta.get("sector") or meta.get("theme") or "Unknown",
                }
            )

        allowed_percentages: Set[str] = set()
        for _, alloc in sorted_alloc[:10]:
            allowed_percentages.add(f"{float(alloc) * 100:.2f}%")
        for val in current_metrics.values():
            if val is not None and abs(val) <= 1:
                allowed_percentages.add(f"{float(val) * 100:.2f}%")

        named_entities: Set[str] = {f"Q{q}", str(year)}
        for ticker, _ in sorted_alloc[:10]:
            named_entities.add(str(ticker).upper())
        for item in filtered_trending_entities[:8]:
            key = str(item.get("key", "")).strip()
            if key:
                named_entities.add(key)
        for event in econ_cal[:10]:
            name = str(event.get("event", "")).strip()
            if name:
                named_entities.add(name)

        quarterly_evidence_payload = {
            "quarter": f"Q{q} {year}",
            "regime_label": regime_label,
            "portfolio_metrics": {k: v for k, v in current_metrics.items() if v is not None},
            "top_holdings_stats": top_holdings_stats,
            "macro_calendar_items": [
                {
                    "date": event.get("date"),
                    "event": event.get("event"),
                    "impact": event.get("impact", "Medium"),
                }
                for event in econ_cal[:10]
            ],
            "allowed_percentages": sorted(allowed_percentages),
            "allowed_named_entities": sorted(named_entities),
        }

        if self.ai_agent.client:
            logger.info("Enhancing quarterly newsletter prose with AI + evidence validation...")
            final_md = self.ai_agent.enhance_quarterly_newsletter_with_validation(
                final_md,
                evidence_payload=quarterly_evidence_payload,
                prior_newsletter_md="",
            )

        state_payload = state if isinstance(state, dict) else {"runs": []}
        history_payload = state_payload.get("quarterly_diagnostics")
        if not isinstance(history_payload, list):
            history_payload = []
        history_payload.append(
            {
                "quarter": f"Q{q} {year}",
                "captured_at": datetime.now().isoformat(),
                **{k: v for k, v in current_metrics.items() if v is not None},
            }
        )
        state_payload["quarterly_diagnostics"] = history_payload[-8:]
        self._save_newsletter_state(state_payload)

        # Save to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_md)

        return output_path
