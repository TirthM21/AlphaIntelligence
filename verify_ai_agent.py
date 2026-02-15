import sys
import os
import logging
from src.ai.ai_agent import AIAgent

logging.basicConfig(level=logging.INFO)

def test_ai_agent_logic():
    print("Testing AIAgent High-Level Methods...")
    agent = AIAgent()
    
    # 1. Test QotD
    print("\n[1/3] Testing generate_qotd()...")
    qotd = agent.generate_qotd()
    print("QotD Result:", qotd)
    
    # 2. Test Commentary
    print("\n[2/3] Testing generate_commentary()...")
    data = {
        "ticker": "NVDA",
        "current_price": 725.10,
        "score": 95.5,
        "details": {"volume_score": 9.0}
    }
    commentary = agent.generate_commentary("NVDA", data)
    print("Commentary Result:", commentary)
    
    # 3. Test Enhancement (Smaller string)
    print("\n[3/3] Testing enhance_newsletter() with small snippet...")
    sample_md = "## Market Update\nSPY is up 1%. Breadth is positive.\n"
    enhanced = agent.enhance_newsletter(sample_md)
    print("Enhanced Snippet:", enhanced)

if __name__ == "__main__":
    test_ai_agent_logic()
