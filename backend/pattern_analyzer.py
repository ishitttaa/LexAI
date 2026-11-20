"""
Pattern Analyzer — Galileo "Signals" equivalent.
Automatically detects failure patterns from the audit trail.

Answers questions like:
  - What types of questions cause most hallucinations?
  - At what context length does accuracy drop?
  - What topics have the highest failure rate?
"""
import re
from collections import Counter


def analyze_patterns(records: list[dict]) -> dict:
    """
    Analyze audit records and return failure patterns + signals.
    """
    if not records:
        return {
            "total_audits": 0,
            "overall_hallucination_rate": 0.0,
            "avg_ensemble_score": 0.0,
            "patterns": [],
            "topic_failure_rates": {},
            "score_distribution": {"low": 0, "medium": 0, "high": 0},
        }

    total = len(records)
    hallucinations = [r for r in records if r["verdict"] == "hallucination"]
    hallucination_rate = round(len(hallucinations) / total, 4)
    avg_score = round(sum(r["ensemble_score"] for r in records) / total, 4)

    # ── Score distribution ─────────────────────────────────────
    score_distribution = {"low": 0, "medium": 0, "high": 0}
    for r in records:
        s = r["ensemble_score"]
        if s < 0.35:
            score_distribution["low"]    += 1
        elif s < 0.60:
            score_distribution["medium"] += 1
        else:
            score_distribution["high"]   += 1

    # ── Topic failure rates ────────────────────────────────────
    topics = {}
    for r in records:
        t = r.get("topic", "general") or "general"
        if t not in topics:
            topics[t] = {"total": 0, "hallucinations": 0}
        topics[t]["total"] += 1
        if r["verdict"] == "hallucination":
            topics[t]["hallucinations"] += 1

    topic_failure_rates = {
        t: round(v["hallucinations"] / v["total"], 4)
        for t, v in topics.items() if v["total"] > 0
    }

    # ── Context length analysis ────────────────────────────────
    flagged_ctx_lengths   = [len(r["context"]) for r in hallucinations]
    unflagged_ctx_lengths = [len(r["context"]) for r in records if r["verdict"] != "hallucination"]
    avg_flagged_len       = int(sum(flagged_ctx_lengths) / len(flagged_ctx_lengths))   if flagged_ctx_lengths   else 0
    avg_unflagged_len     = int(sum(unflagged_ctx_lengths) / len(unflagged_ctx_lengths)) if unflagged_ctx_lengths else 0

    # ── Common words in hallucinated questions ─────────────────
    stop_words = {"the","a","an","is","are","was","were","what","how","why","when","where","who","which","does","do","did","in","of","to","for","on","at","by","with","from","and","or","not","be","been","have","has","had"}
    all_words = []
    for r in hallucinations:
        words = re.findall(r'\b[a-z]{4,}\b', r["question"].lower())
        all_words.extend([w for w in words if w not in stop_words])
    top_words = [w for w, _ in Counter(all_words).most_common(5)]

    # ── Which method detects most hallucinations ───────────────
    method_scores = {
        "Embedding Similarity": sum(r["embedding_score"] for r in records) / total,
        "Groundedness (LLM)":   sum(r["groundedness_score"] for r in records) / total,
        "Self-Consistency":     sum(r["consistency_score"] for r in records) / total,
        "G-Eval":               sum(r["geval_score"] for r in records) / total,
    }
    highest_method = max(method_scores, key=method_scores.get)

    # ── Build signals list ─────────────────────────────────────
    patterns = []

    if hallucination_rate > 0.4:
        patterns.append({
            "label":       "⚠️ High Hallucination Rate",
            "description": f"{round(hallucination_rate*100, 1)}% of audited answers are hallucinations — review your LLM's context retrieval pipeline.",
            "value":       f"{round(hallucination_rate*100, 1)}%",
        })

    if avg_flagged_len < avg_unflagged_len and flagged_ctx_lengths:
        patterns.append({
            "label":       "📏 Short Context = More Hallucinations",
            "description": f"Hallucinated answers had shorter contexts on average ({avg_flagged_len} chars) vs correct answers ({avg_unflagged_len} chars).",
            "value":       f"{avg_flagged_len} chars avg",
        })
    elif flagged_ctx_lengths and avg_flagged_len > avg_unflagged_len:
        patterns.append({
            "label":       "📏 Long Context = More Hallucinations",
            "description": f"Hallucinated answers appeared with longer contexts ({avg_flagged_len} chars) — model may be losing focus in long contexts.",
            "value":       f"{avg_flagged_len} chars avg",
        })

    if top_words:
        patterns.append({
            "label":       "🔑 Common Failure Keywords",
            "description": f"Questions containing these words most often lead to hallucinations.",
            "value":       ", ".join(top_words),
        })

    if topic_failure_rates:
        worst_topic = max(topic_failure_rates, key=topic_failure_rates.get)
        patterns.append({
            "label":       f"📌 Riskiest Topic: {worst_topic}",
            "description": f"The topic '{worst_topic}' has the highest hallucination rate ({round(topic_failure_rates[worst_topic]*100,1)}%).",
            "value":       f"{round(topic_failure_rates[worst_topic]*100,1)}%",
        })

    patterns.append({
        "label":       f"🔍 Strongest Detector: {highest_method}",
        "description": f"{highest_method} gives the highest average hallucination signal across all evaluations.",
        "value":       f"avg score {round(method_scores[highest_method], 2)}",
    })

    return {
        "total_audits":               total,
        "overall_hallucination_rate": hallucination_rate,
        "avg_ensemble_score":         avg_score,
        "patterns":                   patterns,
        "topic_failure_rates":        topic_failure_rates,
        "score_distribution":         score_distribution,
    }
