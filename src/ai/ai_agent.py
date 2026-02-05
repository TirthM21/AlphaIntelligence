"""AI Analysis Engine using apifreellm for premium financial commentary."""

import logging
import os
import requests
import json
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AIAgent:
    """Uses LLM to generate professional financial commentary and newsletter content."""
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('FREE_LLM_API_KEY')
        self.base_url = "https://apifreellm.com/api/v1/chat"
        self.model = "apifreellm"
        
        if not self.api_key:
            logger.warning("FREE_LLM_API_KEY not set. AI features will be disabled.")

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
        if not self.api_key:
            return "AI Commentary unavailable (No API Key)."

        # Sanitize data to remove non-serializable objects like pd.Timestamp
        sanitized_data = self._sanitize_data(data)

        prompt = f"""
        Act as a senior quantitative equity analyst. Analyze this stock data for {ticker}:
        
        Data: {json.dumps(sanitized_data, indent=2)}
        
        Provide a concise, professional 3-sentence summary of the investment thesis, focusing on the quality of the breakout and financial strength.
        """
        
        return self._call_ai(prompt)

    def enhance_newsletter(self, newsletter_md: str) -> str:
        """Improve the language and structure of the newsletter to make it 'Premium'."""
        if not self.api_key:
            return newsletter_md

        prompt = f"""
        Act as a professional financial editor for a top-tier investment firm. 
        Enhance the following newsletter to make it sound institutional, elite, and highly authoritative. 
        Maintain all technical data points but improve the prose and impact.
        
        Newsletter:
        {newsletter_md}
        
        Return ONLY the updated markdown newsletter.
        """
        
        enhanced = self._call_ai(prompt)
        return enhanced if enhanced else newsletter_md

    def _call_ai(self, prompt: str) -> Optional[str]:
        """Call the apifreellm API."""
        if not self.api_key:
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "message": prompt,
            "model": self.model
        }
        
        try:
            # Free tier has a 5s delay/rate limit
            time.sleep(1) 
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 429:
                logger.warning("AI API Rate limit hit. Waiting 5s...")
                time.sleep(5)
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            response.raise_for_status()
            data = response.json()
            logger.info(f"AI API Response status: {response.status_code}")
            
            if data.get('success'):
                return data.get('response')
            
            logger.error(f"AI API returned unsuccessful response: {data}")
            return None
            
        except Exception as e:
            logger.error(f"AI Generation failed: {e}")
            return None
