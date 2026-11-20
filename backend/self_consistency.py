"""
Method C — Self-Consistency Check
Ask Groq the same question 3× with different phrasings/temperatures.
Measure answer consistency using embedding similarity.

Logic:
  Consistent answers → model is confident → lower hallucination risk
  Inconsistent answers → model is uncertain → higher hallucination risk

Hallucination score = 1 - avg pairwise similarity between 3 answers
"""
import asyncio
from itertools import combinations
from backend.groq_client import achat

REPHRASINGS = [
    # Original phrasing template
    "Based on the following context, answer this question:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    # Rephrase 1 — instruction-following style
    "You are a helpful assistant. Use only the provided context to answer.\n\nContext:\n{context}\n\nQ: {question}\nA:",
    # Rephrase 2 — direct concise style
    "Context: {context}\n\nUsing only the context above, provide a concise answer to: {question}\n\nAnswer:",
]


async def _generate_answer(template: str, question: str, context: str, temp: float) -> str:
    prompt = template.format(question=question, context=context[:3000])
    try:
        return await achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=512,
        )
    except Exception as e:
        print(f"[self_consistency] Error generating answer: {e}")
        return ""


def _pairwise_similarity(texts: list[str]) -> float:
    """Compute average pairwise cosine similarity between a list of texts."""
    if len(texts) < 2:
        return 1.0

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        def cosine_sim(a, b):
            a = np.asarray(a).flatten()
            b = np.asarray(b).flatten()
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)

        pairs = list(combinations(range(len(embeddings)), 2))
        sims  = [float(cosine_sim(embeddings[i], embeddings[j])) for i, j in pairs]
        return float(np.mean(sims))
    except Exception as e:
        print(f"[self_consistency] Similarity error: {e}")
        return 0.5


async def self_consistency_check_async(question: str, context: str) -> dict:
    """
    Generate 3 answers to the same question with different phrasings,
    then measure consistency via embedding similarity.
    """
    temperatures = [0.1, 0.5, 0.8]

    # Run all 3 LLM calls in parallel
    tasks = [
        _generate_answer(template, question, context, temp)
        for template, temp in zip(REPHRASINGS, temperatures)
    ]
    answers = await asyncio.gather(*tasks)
    answers = [a for a in answers if a.strip()]  # filter empty

    if len(answers) < 2:
        return {
            "hallucination_score": 0.5,
            "confidence": 0.2,
            "explanation": "Could not generate enough answer variants for consistency check.",
            "answers_generated": len(answers),
            "avg_pairwise_similarity": 0.5,
        }

    # Run similarity in executor (CPU-bound)
    loop = asyncio.get_event_loop()
    avg_sim = await loop.run_in_executor(None, _pairwise_similarity, answers)
    avg_sim = max(0.0, min(1.0, avg_sim))

    # Inconsistency = hallucination signal
    hallucination_score = round(1.0 - avg_sim, 4)
    confidence = round(0.6 + abs(avg_sim - 0.5) * 0.5, 4)

    if avg_sim >= 0.85:
        explanation = f"Answers are highly consistent across 3 phrasings (similarity={avg_sim:.2f}) — model is confident."
    elif avg_sim >= 0.60:
        explanation = f"Moderate consistency across 3 phrasings (similarity={avg_sim:.2f}) — some uncertainty detected."
    else:
        explanation = f"Low consistency across 3 phrasings (similarity={avg_sim:.2f}) — model gives conflicting answers, strong hallucination signal."

    return {
        "hallucination_score": hallucination_score,
        "confidence": confidence,
        "explanation": explanation,
        "answers_generated": len(answers),
        "avg_pairwise_similarity": round(avg_sim, 4),
        "sample_answers": answers[:2],  # include first 2 for UI display
    }
