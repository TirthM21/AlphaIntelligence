from openai import OpenAI
import os
import sys

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

def test_glm5():
    print("Testing NVIDIA z-ai/glm5...")
    try:
        completion = client.chat.completions.create(
          model="z-ai/glm5",
          messages=[{"role":"user","content":"Provide a brief institutional market outlook for Q1 2026."}],
          temperature=0.1,
          max_tokens=500,
          extra_body={"chat_template_kwargs":{"enable_thinking":True,"clear_thinking":False}},
          stream=True
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n\nGLM5 Success.")
    except Exception as e:
        print(f"\nGLM5 Error: {e}")

if __name__ == "__main__":
    test_glm5()
