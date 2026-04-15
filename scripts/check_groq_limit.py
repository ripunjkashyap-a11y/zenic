"""Minimal Groq API call to check if tokens are available today."""
import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq, RateLimitError

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

try:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
    )
    print(f"OK — API is responding.")
    print(f"Tokens used in this call: {resp.usage.total_tokens}")
    print(f"Prompt tokens: {resp.usage.prompt_tokens}, Completion tokens: {resp.usage.completion_tokens}")
except RateLimitError as e:
    print(f"RATE LIMIT HIT — Groq TPD exhausted.")
    print(f"Error: {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
