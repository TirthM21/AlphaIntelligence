from openai import OpenAI
import os
import sys

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

print("Initializing NVIDIA z-ai/glm4.7 client...")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

try:
    print("Sending request...")
    completion = client.chat.completions.create(
      model="z-ai/glm4.7",
      messages=[{"role":"user","content":"Provide a brief institutional market outlook for Q1 2026."}],
      temperature=1,
      top_p=1,
      max_tokens=1024,
      extra_body={"chat_template_kwargs":{"enable_thinking":True,"clear_thinking":False}},
      stream=True
    )

    print("Response received (streaming):\n")
    for chunk in completion:
      if not getattr(chunk, "choices", None):
        continue
      if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
        continue
      delta = chunk.choices[0].delta
      reasoning = getattr(delta, "reasoning_content", None)
      if reasoning:
        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="")
      if getattr(delta, "content", None) is not None:
        print(delta.content, end="")
    print("\n\nTest complete.")
except Exception as e:
    print(f"\nError encountered: {e}")
