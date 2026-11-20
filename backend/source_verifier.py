"""
Source Verifier — for Document Audit Mode and QA Auto-Context.
Searches Google via Serper API to find sources for each claim,
and auto-builds context for the QA Evaluator from real web results.
"""
import os
import httpx
from dotenv import load_dotenv
from backend.models import Source

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL     = "https://google.serper.dev/search"


def search_sources(claim: str, num_results: int = 3) -> list[Source]:
    if not SERPER_API_KEY:
        return []
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": claim, "num": num_results, "gl": "us", "hl": "en"}
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(SERPER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return [
            Source(
                title=r.get("title", "No title"),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
            )
            for r in data.get("organic", [])[:num_results]
        ]
    except Exception as e:
        print(f"[source_verifier] Error: {e}")
        return []


async def fetch_context_for_question(question: str, num_results: int = 5) -> tuple[str, list[Source]]:
    """
    Auto-fetch real-world ground-truth context for a question via Google Search.

    Returns:
        (context_str, sources) — a combined context string built from top search
        snippets, and the list of Source objects used (for UI display).
    """
    if not SERPER_API_KEY:
        return "", []

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": question, "num": num_results, "gl": "us", "hl": "en"}

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(SERPER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        organic = data.get("organic", [])[:num_results]
        knowledge_graph = data.get("knowledgeGraph", {})

        sources: list[Source] = []
        context_parts: list[str] = []

        # Add knowledge graph info first (most authoritative)
        if knowledge_graph:
            kg_title = knowledge_graph.get("title", "")
            kg_desc  = knowledge_graph.get("description", "")
            if kg_title and kg_desc:
                context_parts.append(f"{kg_title}: {kg_desc}")

        for r in organic:
            title   = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            url     = r.get("link", "")
            if snippet:
                context_parts.append(f"{title}: {snippet}" if title else snippet)
                sources.append(Source(title=title or url, url=url, snippet=snippet))

        if not context_parts:
            return "", []

        context_str = "\n\n".join(context_parts)
        return context_str, sources

    except Exception as e:
        print(f"[source_verifier] fetch_context error: {e}")
        return "", []
