"""
Method B — LLM-as-Judge (Groundedness)
Uses Groq/LLaMA to judge whether the answer is grounded in the given context.
Each factual claim in the answer is evaluated against the context.

Hallucination score returned = proportion of claims NOT grounded in context.
"""
import json
import re
from backend.groq_client import achat

GROUNDEDNESS_PROMPT = """You are a strict fact-checking judge. Your job is to evaluate whether an AI answer is grounded in the provided context.

QUESTION: {question}

CONTEXT (Source of Truth):
{context}

AI ANSWER TO EVALUATE:
{answer}

Instructions:
1. Identify each factual claim in the AI Answer.
2. For each claim, check if it is SUPPORTED, CONTRADICTED, or NOT_ADDRESSED by the Context.
3. Be strict — if the context doesn't mention something, it's NOT_ADDRESSED (not verified).

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "claim": "exact claim from answer",
      "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_ADDRESSED",
      "reason": "one sentence explanation"
    }}
  ],
  "overall_groundedness": <float 0.0 to 1.0>,
  "summary": "one sentence overall assessment"
}}

Rules:
- overall_groundedness: fraction of claims that are SUPPORTED (0.0 = none, 1.0 = all)
- Be concise and precise in reasons
"""


async def groundedness_check_async(question: str, context: str, answer: str) -> dict:
    """
    Async LLM-as-judge groundedness check via Groq.
    Returns hallucination_score, confidence, and per-claim breakdown.
    """
    prompt = GROUNDEDNESS_PROMPT.format(
        question=question,
        context=context[:4000],   # limit context to avoid token overflow
        answer=answer[:2000],
    )

    try:
        text = await achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
        )

        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        result = json.loads(text)
        groundedness = float(result.get("overall_groundedness", 0.5))
        groundedness = max(0.0, min(1.0, groundedness))
        hallucination_score = round(1.0 - groundedness, 4)

        claims = result.get("claims", [])
        n_total       = len(claims)
        n_supported   = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
        n_contradicted= sum(1 for c in claims if c.get("verdict") == "CONTRADICTED")

        confidence = 0.85 if n_total > 0 else 0.4

        return {
            "hallucination_score": hallucination_score,
            "confidence": confidence,
            "explanation": result.get("summary", "LLM judge evaluation complete."),
            "claims_breakdown": claims,
            "n_claims": n_total,
            "n_supported": n_supported,
            "n_contradicted": n_contradicted,
        }

    except (json.JSONDecodeError, Exception) as e:
        print(f"[groundedness] Error: {e}")
        return {
            "hallucination_score": 0.5,
            "confidence": 0.3,
            "explanation": "Could not complete groundedness check due to parsing error.",
            "claims_breakdown": [],
            "n_claims": 0,
            "n_supported": 0,
            "n_contradicted": 0,
        }
