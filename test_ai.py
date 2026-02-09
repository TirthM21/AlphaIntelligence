import os
import logging
from src.ai.ai_agent import AIAgent
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ai():
    print("--- Testing AIAgent (NVIDIA NIM) ---")
    ai = AIAgent()
    
    if not ai.api_key:
        print("❌ AI API Key not found in environment")
        return

    # Test 1: Simple Commentary
    print("\nTest 1: Generating Market Commentary for NVDA...")
    data = {
        "price": 700.50,
        "score": 85.5,
        "ratios": "Net Margin: 45%, Revenue Growth: 200%, ROE: 60%"
    }
    commentary = ai.generate_commentary("NVDA", data)
    print(f"Commentary Result: {commentary}")
    
    # Test 2: Newsletter Enhancement
    print("\nTest 2: Enhancing a sample Newsletter...")
    sample_md = """
    # Market Update
    The market was up today. Microsoft and Apple did well.
    - AAPL: breakout at 190.
    - MSFT: strong earnings.
    """
    enhanced = ai.enhance_newsletter(sample_md)
    print(f"Enhanced Newsletter (partial): {enhanced[:200]}...")

if __name__ == "__main__":
    test_ai()
