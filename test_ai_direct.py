import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('NVIDIA_API_KEY') or "z-ai/glm5-BbrtwJCRkdzAgrjeeRMAVNt3aUN5FWB"

def test_ai():
    print(f"Testing AI with key: {api_key[:10]}...")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    try:
        completion = client.chat.completions.create(
            model="z-ai/glm5",
            messages=[{"role":"user","content":"Provide a 1-sentence institutional market fact about the S&P 500."}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=100,
        )
        print("\nAI Response:")
        print(completion.choices[0].message.content)
        return True
    except Exception as e:
        print(f"\nAI Error: {e}")
        return False

if __name__ == "__main__":
    test_ai()
