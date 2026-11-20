"""
Confidence Scorer — for Document Audit Mode.
Uses Groq/LLaMA to judge each claim against web search results.
"""
import json
from backend.groq_client import chat
from backend.models import Source, Verdict

SCORING_PROMPT = """You are a fact-checking expert. Evaluate whether this claim is supported by the search results.

CLAIM: "{claim}"

SEARCH RESULTS:
{sources_text}

Classify as:
- VERIFIED: Search results directly confirm the claim
- PLAUSIBLE: Results are related but don't fully confirm or deny
- HALLUCINATION: Results contradict the claim OR no relevant results found

Return ONLY valid JSON:
{{
  "status": "verified" | "plausible" | "hallucination",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one sentence>"
}}
"""


def score_claim(claim: str, sources: list[Source]) -> dict:
    if not sources:
        return {
            "status": Verdict.HALLUCINATION,
            "confidence": 0.05,
            "explanation": "No sources found for this claim.",
        }

    sources_text = "\n\n".join([
        f"Source {i+1}: {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
        for i, s in enumerate(sources)
    ])

    try:
        text = chat(
            messages=[{"role": "user", "content": SCORING_PROMPT.format(
                claim=claim, sources_text=sources_text
            )}],
            temperature=0.1,
            max_tokens=300,
        )
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        result = json.loads(text)
        status_map = {
            "verified":     Verdict.VERIFIED,
            "plausible":    Verdict.PLAUSIBLE,
            "hallucination":Verdict.HALLUCINATION,
        }
        result["status"]     = status_map.get(result.get("status","plausible").lower(), Verdict.PLAUSIBLE)
        result["confidence"] = float(result.get("confidence", 0.5))
        return result
    except Exception as e:
        print(f"[confidence_scorer] Error: {e}")
        return {"status": Verdict.PLAUSIBLE, "confidence": 0.4, "explanation": "Scoring error."}
