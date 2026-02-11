"""AI Analysis Engine using NVIDIA NIM for elite financial commentary."""

import logging
import os
import json
import time
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIAgent:
    """Uses NVIDIA NIM (GLM-4.7) to generate professional financial commentary and newsletter content."""
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        # Primary key from user's snippet, fallback to env
        self.api_key = api_key or os.getenv('FREE_LLM_API_KEY')
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = "z-ai/glm4.7"
        self.request_timeout_seconds = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "60"))
        self.max_attempts = int(os.getenv("AI_MAX_ATTEMPTS", "2"))
        
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.request_timeout_seconds
            )
            logger.info(f"AIAgent initialized using NVIDIA NIM with model {self.model}")
        else:
            logger.warning("AI API Key not set. AI features will be disabled.")

    def _sanitize_data(self, data: Any) -> Any:
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
            return "AI Commentary unavailable (No API Key)."

        # Sanitize data to remove non-serializable objects like pd.Timestamp
        sanitized_data = self._sanitize_data(data)

        prompt = f"""
        Act as a senior quantitative equity analyst at a top-tier hedge fund. 
        Analyze this stock data for {ticker}:
        
        Data: {json.dumps(sanitized_data, indent=2)}
        
        Provide a concise, professional 3-sentence summary of the investment thesis.
        Focus on:
        1. The quality of the technical breakout.
        2. Financial strength and momentum.
        3. A clear "Quant's Take" verdict.
        """
        
        return self._call_ai(prompt) or "AI analysis failed to generate."

    def enhance_newsletter(self, newsletter_md: str) -> str:
        """Improve the language and structure of the newsletter to make it 'Premium'."""
        if not self.client:
            return newsletter_md

        prompt = f"""
        Act as a professional financial editor for a top-tier investment firm (e.g., Goldman Sachs or BlackRock). 
        Enhance the following newsletter to make it sound institutional, elite, and highly authoritative. 
        Maintain all technical data points but improve the prose, impact, and "premium" feel.
        
        Newsletter:
        {newsletter_md}
        
        Return ONLY the updated markdown newsletter.
        """
        
        enhanced = self._call_ai(prompt)
        return enhanced if enhanced else newsletter_md


    def generate_deep_dive_report(self, scan_context: str) -> str:
        """Generate deep-dive report body with deterministic fallback if AI is unavailable."""
        prompt = f"""
        Act as a Head of Quantitative Strategy at a Tier-1 Hedge Fund.

        Data from Scan:
        {scan_context}

        Structure your report as follows:
        1. Strategic Market Regime
        2. Top 3 Alpha Picks
        3. Structural Risks
        4. Final Verdict
        """
        report = self._call_ai(prompt)
        if report:
            return report

        return (
            "## Strategic Market Regime\n"
            "AI endpoint unavailable (timeout/5xx). Use scanner breadth + SPY trend section as primary regime signal.\n\n"
            "## Top Alpha Picks\n"
            "Select highest-score Phase 2 names from latest scan and validate with fundamentals + volume confirmation.\n\n"
            "## Structural Risks\n"
            "Watch for margin contraction, inventory build, and broad-market distribution signals.\n\n"
            "## Final Verdict\n"
            "Maintain selective risk-on posture only where technical trend and fundamental momentum align."
        )

    def _call_ai(self, prompt: str) -> Optional[str]:
        """Call NVIDIA NIM API with bounded retries and timeout."""
        if not self.client:
            return None

        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"content": prompt, "role": "user"}],
                    temperature=0.7,
                    top_p=1,
                    max_tokens=4096,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                            "clear_thinking": False
                        }
                    },
                    stream=False,
                    timeout=self.request_timeout_seconds,
                )
                if not response or not response.choices:
                    return None
                content = response.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                last_error = e
                logger.warning(f"AI attempt {attempt}/{self.max_attempts} failed: {e}")
                if attempt < self.max_attempts:
                    time.sleep(1.5 * attempt)

        logger.error(f"AI Generation failed after {self.max_attempts} attempts: {last_error}")
        return None
