"""
Weighted Ensemble Scorer
Combines all 4 detection method scores into a single final hallucination probability.

Weights (tuned for best F1 on FEVER-style datasets):
  Method A — Embedding Similarity : 25%
  Method B — Groundedness (LLM)   : 35%
  Method C — Self-Consistency     : 20%
  Method D — G-Eval               : 20%

Final score: 0.0 = definitely grounded, 1.0 = definite hallucination
"""
from backend.models import Verdict

# Default weights — can be overridden
WEIGHTS = {
    "embedding":     0.25,
    "groundedness":  0.35,
    "consistency":   0.20,
    "geval":         0.20,
}

# Verdict thresholds
HALLUCINATION_THRESHOLD = 0.60   # above this → hallucination
PLAUSIBLE_THRESHOLD     = 0.35   # above this → plausible


def compute_ensemble_score(
    embedding_score:    float,
    groundedness_score: float,
    consistency_score:  float,
    geval_score:        float,
    weights:            dict = None,
) -> dict:
    """
    Compute weighted ensemble hallucination score.
    All input scores are hallucination probability (0=grounded, 1=hallucination).
    Returns final score, verdict, and per-method contribution breakdown.
    """
    w = weights or WEIGHTS

    final = (
        w["embedding"]    * embedding_score    +
        w["groundedness"] * groundedness_score +
        w["consistency"]  * consistency_score  +
        w["geval"]        * geval_score
    )
    final = round(max(0.0, min(1.0, final)), 4)

    if final >= HALLUCINATION_THRESHOLD:
        verdict = Verdict.HALLUCINATION
    elif final >= PLAUSIBLE_THRESHOLD:
        verdict = Verdict.PLAUSIBLE
    else:
        verdict = Verdict.VERIFIED

    contributions = {
        "embedding":    round(w["embedding"]    * embedding_score,    4),
        "groundedness": round(w["groundedness"] * groundedness_score, 4),
        "consistency":  round(w["consistency"]  * consistency_score,  4),
        "geval":        round(w["geval"]        * geval_score,        4),
    }

    return {
        "ensemble_score": final,
        "verdict":        verdict,
        "contributions":  contributions,
        "weights_used":   w,
    }
