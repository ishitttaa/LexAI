"""
Shared Groq LLM client — used by all detection methods.
Model is configurable via GROQ_MODEL env var.
"""
import os
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Sync client (for simple calls)
groq_client = Groq(api_key=GROQ_API_KEY)

# Async client (for parallel detection methods)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Synchronous Groq chat call. Returns the response text."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


async def achat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Async Groq chat call. Returns the response text."""
    response = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
