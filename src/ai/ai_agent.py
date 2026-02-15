"""AI Analysis Engine using NVIDIA Integrated API (z-ai/glm5) for premium financial commentary."""

import logging
import os
import json
import time
from typing import Dict, List, Optional
from pathlib import Path

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
        if not self.client:
            return newsletter_md

        prompt = f"""
        Act as a professional financial editor for AlphaIntelligence Capital. 
        Enhance the following newsletter to make it sound institutional, elite, and authoritative. 
        
        CRITICAL STRUCTURAL RULES:
        1. MAINTAIN the '## 🏛️ AlphaIntelligence Capital BRIEF' header.
        2. KEEP all vertical lists (e.g. Sector Performance, Market Sentiment, Today's Events) EXACTLY as they are. DO NOT convert them into tables.
        3. Do NOT add new sections that weren't in the original text.
        4. Maintain all technical data points, URLs, and markdown formatting.
        5. Improve the narrative transitions and sophisticated terminology.
        
        Newsletter:
        {newsletter_md}
        
        Return ONLY the updated markdown newsletter.
        """
        
        enhanced = self._call_ai(prompt, low_temp=True)
        return enhanced if enhanced else newsletter_md

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

    def _call_ai(self, prompt: str, low_temp: bool = False) -> Optional[str]:
        """Call NVIDIA Integrated API."""
        if not self.client:
            return None

        try:
            # Non-streaming for better stability and simpler processing
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 if low_temp else 1.0,
                top_p=1,
                max_tokens=16384,
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
                stream=False
            )
            
            return completion.choices[0].message.content.strip() if completion.choices else None
            
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            return None
