"""
Claim Extractor — for Document Audit Mode (Mode 2).
Uses Groq/LLaMA to extract factual claims from a document.
"""
import os
import json
import re
from backend.groq_client import chat

EXTRACTION_PROMPT = """You are a factual claim extractor. Given the following document, extract ALL verifiable factual claims.

A factual claim is any statement that:
- Asserts a specific fact, statistic, date, event, person, place, or measurement
- Can be verified as true or false against external sources
- Is NOT an opinion or subjective statement

For each claim, return the exact original text fragment from the document and a clean, standalone claim statement.

Return ONLY a valid JSON array with this exact format:
[
  {{
    "claim": "Clean standalone factual claim",
    "original_text": "Exact phrase from the document"
  }}
]

If no factual claims are found, return an empty array: []

DOCUMENT:
{document}
"""


def extract_claims(document: str) -> list[dict]:
    """Extract factual claims from a document using Groq."""
    prompt = EXTRACTION_PROMPT.format(document=document[:8000])
    try:
        text = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        claims = json.loads(text)
        return claims if isinstance(claims, list) else []
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return []
    except Exception as e:
        print(f"[claim_extractor] Error: {e}")
        return []
