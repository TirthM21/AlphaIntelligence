"""AI Analysis Engine using NVIDIA Integrated API (z-ai/glm5) for premium financial commentary."""

import logging
import os
import json
import re
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AIAgent:
    """Uses LLM to generate professional financial commentary and newsletter content."""
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        # Use the specific NVIDIA key for institutional reasoning (z-ai/glm4.7)
        self.api_key = api_key or os.getenv('NVIDIA_API_KEY') or "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"
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = "z-ai/glm4.7"
        
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            logger.info(f"AIAgent initialized using NVIDIA {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            self.client = None

    def _sanitize_data(self, data: any) -> any:
        """Recursively convert non-serializable objects (like pd.Timestamp) to strings."""
        if isinstance(data, dict):
            return {str(k): self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(i) for i in data]
        elif hasattr(data, 'isoformat'): # Handles pd.Timestamp and datetime
            return data.isoformat()
        return data

    def generate_commentary(self, ticker: str, data: Dict) -> str:
        """Generate AI commentary for a specific stock breakout."""
        if not self.client:
            return "AI Commentary unavailable (Client Error)."

        sanitized_data = self._sanitize_data(data)

        prompt = f"""
        Act as a senior quantitative equity analyst. Analyze this stock data for {ticker}:
        
        Data: {json.dumps(sanitized_data, indent=2)}
        
        Provide a concise, professional 3-sentence summary of the investment thesis, focusing on the quality of the breakout and financial strength.
        """
        
        return self._call_ai(prompt) or "Breakout confirmed by technical indicators with supporting fundamental growth."

    def enhance_newsletter(self, newsletter_md: str) -> str:
        """Improve the language and structure of the newsletter."""
        return self.enhance_newsletter_with_validation(newsletter_md)

    def enhance_newsletter_with_validation(
        self,
        newsletter_md: str,
        evidence_payload: Optional[Dict] = None,
        prior_newsletter_md: str = ""
    ) -> str:
        """Enhance newsletter prose with validation and constrained regeneration."""
        if not self.client:
            return newsletter_md

        evidence_payload = evidence_payload or {}
        base_prompt = self._build_newsletter_prompt(
            newsletter_md,
            evidence_payload,
            prior_newsletter_md=prior_newsletter_md,
            stricter=False,
        )
        enhanced = self._call_ai(base_prompt, low_temp=True)
        if not enhanced:
            return newsletter_md

        issues = self._validate_newsletter(enhanced, evidence_payload, prior_newsletter_md)
        if not issues:
            return enhanced

        logger.warning("Newsletter validation failed (%s). Regenerating with stricter prompt.", "; ".join(issues))
        retry_prompt = self._build_newsletter_prompt(
            newsletter_md,
            evidence_payload,
            prior_newsletter_md=prior_newsletter_md,
            stricter=True,
            validation_issues=issues,
        )
        retry = self._call_ai(retry_prompt, temperature=0.05)
        if not retry:
            return enhanced

        retry_issues = self._validate_newsletter(retry, evidence_payload, prior_newsletter_md)
        return retry if not retry_issues else enhanced

    def _build_newsletter_prompt(
        self,
        newsletter_md: str,
        evidence_payload: Dict,
        prior_newsletter_md: str = "",
        stricter: bool = False,
        validation_issues: Optional[List[str]] = None,
    ) -> str:
        """Construct constrained newsletter editing prompt."""
        max_reused_sentences = 0 if stricter else 1
        extra_guardrails = ""
        if validation_issues:
            extra_guardrails = f"\nFailed checks from prior draft: {', '.join(validation_issues)}. Resolve every failed check."

        prompt = f"""
        Act as a professional financial editor for AlphaIntelligence Capital. 
        Enhance the following newsletter to make it sound institutional, elite, and authoritative.
        Keep voice concise, analyst-style, and evidence-led (short declarative sentences, no hype, no rhetorical questions, no clichés).
        
        CRITICAL STRUCTURAL RULES:
        1. MAINTAIN the '## 🏛️ AlphaIntelligence Capital BRIEF' header.
        2. KEEP all vertical lists (e.g. Sector Performance, Market Sentiment, Today's Events) EXACTLY as they are. DO NOT convert them into tables.
        3. Do NOT add new sections that weren't in the original text.
        4. Maintain all technical data points, URLs, and markdown formatting.
        5. Improve narrative transitions using concise analyst language.

        REQUIRED EVIDENCE SLOTS (must be explicit, each on its own bullet in the market overview narrative):
        - Index Move: cite index and exact move.
        - Sector Leader/Laggard: identify both with numeric spread.
        - Mover Stats: include at least one advancing and one declining ticker move.
        - Event References: reference upcoming/active macro or earnings events.

        SENTENCE REUSE CONSTRAINT:
        - Reuse at most {max_reused_sentences} full sentence(s) from prior day's newsletter text.

        DATA PAYLOAD (authoritative facts only):
        {json.dumps(self._sanitize_data(evidence_payload), indent=2)}

        PRIOR DAY NEWSLETTER (for anti-repetition only):
        {prior_newsletter_md[:6000] if prior_newsletter_md else 'N/A'}
        {extra_guardrails}
        
        Newsletter:
        {newsletter_md}
        
        Return ONLY the updated markdown newsletter.
        """
        return prompt

    def _validate_newsletter(self, text: str, evidence_payload: Dict, prior_newsletter_md: str) -> List[str]:
        """Validate generated newsletter for evidence anchors, repetition, and unsupported claims."""
        issues = []

        lower_text = text.lower()
        required_slot_terms = ["index move", "sector leader", "laggard", "mover", "event"]
        if not all(term in lower_text for term in required_slot_terms):
            issues.append("missing explicit evidence slots")

        numeric_anchors = re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)
        if len(numeric_anchors) < 8:
            issues.append("missing numeric anchors")

        if prior_newsletter_md:
            repeated = self._find_reused_phrases(text, prior_newsletter_md)
            if repeated:
                issues.append("repeated phrases from prior run")

        unsupported = self._find_unsupported_claims(text, evidence_payload)
        if unsupported:
            issues.append("unsupported claims not present in fetched data payload")

        return issues

    def _find_reused_phrases(self, current_text: str, prior_text: str) -> List[str]:
        """Return long repeated phrases reused from prior run."""
        current_sentences = {s.strip().lower() for s in re.split(r"[\n\.!?]+", current_text) if len(s.split()) >= 9}
        prior_sentences = {s.strip().lower() for s in re.split(r"[\n\.!?]+", prior_text) if len(s.split()) >= 9}
        return sorted(current_sentences.intersection(prior_sentences))[:5]

    def _find_unsupported_claims(self, text: str, evidence_payload: Dict) -> List[str]:
        """Detect referenced symbols/events that are not in payload allow-list."""
        payload_blob = json.dumps(self._sanitize_data(evidence_payload)).lower()
        unsupported_tokens = []

        for token in re.findall(r"\b[A-Z]{2,5}\b", text):
            if token.lower() not in payload_blob and token not in {"SMA", "RSI", "GDP", "CPI", "EPS"}:
                unsupported_tokens.append(token)

        return unsupported_tokens[:5]

    def generate_qotd(self) -> Dict[str, str]:
        """Generate a 'Question of the Day' with institutional insight."""
        if not self.client:
            return {
                "question": "What is the historical average return of February?",
                "answer": "Historically, February is one of the weakest months for the S&P 500, often showing a negative average return.",
                "insight": "Investors should be selective and look for relative strength during seasonal weakness."
            }

        prompt = """
        Generate a 'Question of the Day' about stock market history, seasonality, or quantitative indicators.
        
        Return JSON with:
        "question": A compelling question.
        "answer": A factual answer.
        "insight": A professional takeaway.
        
        Ensure it is factual and high-quality.
        """
        
        try:
            resp = self._call_ai(prompt, low_temp=False)
            if resp:
                clean_resp = resp.strip().replace('```json', '').replace('```', '')
                return json.loads(clean_resp)
        except Exception as e:
            logger.error(f"Failed to generate QotD: {e}")
            
        return {
            "question": "What happens after a 1% market drop?",
            "answer": "It typically takes about four weeks on average for the market to fully recover from a 1% single-day drop.",
            "insight": "Market pullbacks are often temporary mean-reversion events within a larger trend."
        }

    def _call_ai(self, prompt: str, low_temp: bool = False, temperature: Optional[float] = None) -> Optional[str]:
        """Call NVIDIA Integrated API."""
        if not self.client:
            return None

        try:
            # Non-streaming for better stability and simpler processing
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else (0.1 if low_temp else 1.0),
                top_p=1,
                max_tokens=16384,
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
                stream=False
            )
            
            return completion.choices[0].message.content.strip() if completion.choices else None
            
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            return None
