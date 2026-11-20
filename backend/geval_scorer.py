"""
Method D — G-Eval Style Scoring
Uses Groq to score the answer on 3 dimensions:
  1. Factual Accuracy  — does it match the context?
  2. Reasoning Quality — is the logic sound?
  3. Relevance         — does it answer the question?

Each dimension scored 1–10. Final hallucination score derived from combined low scores.
"""
import json
from backend.groq_client import achat

GEVAL_PROMPT = """You are an expert evaluator for LLM-generated answers. Evaluate the following answer on three dimensions.

QUESTION: {question}

CONTEXT (ground truth reference):
{context}

ANSWER TO EVALUATE:
{answer}

Rate the answer on each dimension from 1 (very poor) to 10 (excellent):

1. **Factual Accuracy** — Are all facts in the answer consistent with the context? Does it introduce unsupported claims?
2. **Reasoning Quality** — Is the logic and reasoning in the answer sound and coherent?
3. **Relevance** — Does the answer actually address what was asked?

Return ONLY valid JSON in this exact format:
{{
  "factual_accuracy": <int 1-10>,
  "reasoning_quality": <int 1-10>,
  "relevance": <int 1-10>,
  "factual_accuracy_reason": "brief explanation",
  "reasoning_quality_reason": "brief explanation",
  "relevance_reason": "brief explanation",
  "overall_assessment": "one sentence summary"
}}
"""


async def geval_score_async(question: str, context: str, answer: str) -> dict:
    """
    Async G-Eval scoring via Groq.
    Returns hallucination_score derived from weighted dimension scores.
    """
    prompt = GEVAL_PROMPT.format(
        question=question,
        context=context[:4000],
        answer=answer[:2000],
    )

    try:
        text = await achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )

        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        result = json.loads(text)

        factual   = int(result.get("factual_accuracy",  5))
        reasoning = int(result.get("reasoning_quality", 5))
        relevance = int(result.get("relevance",          5))

        # Clamp scores to valid range
        factual   = max(1, min(10, factual))
        reasoning = max(1, min(10, reasoning))
        relevance = max(1, min(10, relevance))

        # Weighted combined score (factual accuracy weighted highest)
        combined = (0.50 * factual + 0.30 * reasoning + 0.20 * relevance)
        normalized = combined / 10.0  # 0-1 where 1 = perfect

        # Hallucination score = inverse of quality
        hallucination_score = round(1.0 - normalized, 4)
        confidence = 0.88  # G-Eval is generally reliable

        return {
            "hallucination_score": hallucination_score,
            "confidence": confidence,
            "explanation": result.get("overall_assessment", "G-Eval scoring complete."),
            "dimensions": {
                "factual_accuracy":       factual,
                "reasoning_quality":      reasoning,
                "relevance":              relevance,
                "factual_accuracy_reason":   result.get("factual_accuracy_reason", ""),
                "reasoning_quality_reason":  result.get("reasoning_quality_reason", ""),
                "relevance_reason":          result.get("relevance_reason", ""),
            },
        }

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"[geval] Error: {e}")
        return {
            "hallucination_score": 0.5,
            "confidence": 0.3,
            "explanation": "G-Eval scoring failed due to a parsing error.",
            "dimensions": {
                "factual_accuracy": 5, "reasoning_quality": 5, "relevance": 5,
                "factual_accuracy_reason": "", "reasoning_quality_reason": "", "relevance_reason": "",
            },
        }
