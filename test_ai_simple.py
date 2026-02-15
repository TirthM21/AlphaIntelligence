import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('NVIDIA_API_KEY') or "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"
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

def test_simple():
    print("Testing simple AI call (no streaming)...")
    try:
        completion = client.chat.completions.create(
            model="z-ai/glm4.7",
            messages=[{"role": "user", "content": "Hello, provide a 1-sentence market greeting."}],
            temperature=0.1,
            max_tokens=100
        )
        print(f"Response: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simple()
