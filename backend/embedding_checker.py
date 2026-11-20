"""
Method A — Embedding Similarity
Uses sentence-transformers to compute cosine similarity between
the LLM answer and the provided context.

Score interpretation:
  High similarity (close to 1.0) → answer is grounded in context
  Low similarity (close to 0.0)  → answer diverges from context → hallucination risk
  
Hallucination score returned = 1 - similarity (so high = more hallucination)
"""
import asyncio
import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Lazy-load to avoid slow startup — model loads on first use
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embedding_similarity_check(answer: str, context: str) -> dict:
    """
    Compute cosine similarity between answer and context embeddings.
    Returns hallucination_score (0=grounded, 1=hallucination), confidence, explanation.
    """
    if not answer.strip() or not context.strip():
        return {
            "hallucination_score": 0.5,
            "confidence": 0.3,
            "explanation": "Empty answer or context — could not compute embedding similarity.",
        }

    model = _get_model()
    emb_answer  = model.encode([answer])
    emb_context = model.encode([context])
    similarity  = float(cosine_similarity(emb_answer, emb_context))
    similarity  = max(0.0, min(1.0, similarity))  # clamp to [0, 1]

    # Convert to hallucination score: low similarity = high hallucination risk
    hallucination_score = round(1.0 - similarity, 4)
    confidence = round(0.5 + abs(similarity - 0.5), 4)  # more confident near extremes

    if similarity >= 0.80:
        explanation = f"Answer is highly similar to context (similarity={similarity:.2f}) — strongly grounded."
    elif similarity >= 0.55:
        explanation = f"Moderate similarity to context (similarity={similarity:.2f}) — partially grounded."
    elif similarity >= 0.30:
        explanation = f"Low similarity to context (similarity={similarity:.2f}) — likely diverges from context."
    else:
        explanation = f"Very low similarity to context (similarity={similarity:.2f}) — strong hallucination signal."

    return {
        "hallucination_score": hallucination_score,
        "confidence": confidence,
        "explanation": explanation,
        "raw_similarity": similarity,
    }


async def embedding_similarity_check_async(answer: str, context: str) -> dict:
    """Async wrapper — runs in executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedding_similarity_check, answer, context)
