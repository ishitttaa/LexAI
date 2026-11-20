import sys
import os
import asyncio
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models import (
    Verdict,
    QARequest, QAReport, MethodScore,
    DocAuditRequest, DocAuditReport, Claim,
    AuditTrailResponse, SignalsResponse
)
from backend.embedding_checker import embedding_similarity_check_async
from backend.groundedness_checker import groundedness_check_async
from backend.self_consistency import self_consistency_check_async
from backend.geval_scorer import geval_score_async
from backend.ensemble_scorer import compute_ensemble_score, WEIGHTS
from backend.audit_store import init_db, save_audit, get_all_audits
from backend.pattern_analyzer import analyze_patterns
from backend.claim_extractor import extract_claims
from backend.source_verifier import search_sources, fetch_context_for_question
from backend.confidence_scorer import score_claim
import io
from pypdf import PdfReader

app = FastAPI(
    title="Galileo-Style Hallucination Audit System",
    description="Multi-method hallucination detection platform powered by Groq.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.on_event("startup")
async def startup_event():
    await init_db()


@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


# ── Mode 1: QA Evaluation (4 Methods Parallel) ──────────────────────────

@app.post("/api/qa-audit", response_model=QAReport)
async def run_qa_audit(req: QARequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    if not req.question or not req.answer:
        raise HTTPException(status_code=400, detail="Missing question or answer.")

    # Auto-fetch context from Google if not provided by user
    context_sources = []
    fetched_context = None
    if not req.context or not req.context.strip():
        fetched_context, context_sources = await fetch_context_for_question(req.question)
        if not fetched_context:
            raise HTTPException(
                status_code=503,
                detail="Could not auto-fetch context from web. Please paste context manually or check your SERPER_API_KEY."
            )
        context = fetched_context
    else:
        context = req.context

    # Run all 4 methods concurrently
    (emb_res, grd_res, cons_res, gev_res) = await asyncio.gather(
        embedding_similarity_check_async(req.answer, context),
        groundedness_check_async(req.question, context, req.answer),
        self_consistency_check_async(req.question, context),
        geval_score_async(req.question, context, req.answer)
    )

    # Compute weighted ensemble
    ensemble = compute_ensemble_score(
        emb_res["hallucination_score"],
        grd_res["hallucination_score"],
        cons_res["hallucination_score"],
        gev_res["hallucination_score"]
    )

    # Save to SQLite in background
    topic = req.topic or "general"
    audit_id = await save_audit(
        question=req.question,
        context=context,
        answer=req.answer,
        topic=topic,
        embedding_score=emb_res["hallucination_score"],
        groundedness_score=grd_res["hallucination_score"],
        consistency_score=cons_res["hallucination_score"],
        geval_score=gev_res["hallucination_score"],
        ensemble_score=ensemble["ensemble_score"],
        verdict=ensemble["verdict"]
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return QAReport(
        question=req.question,
        context=context,
        answer=req.answer,
        topic=topic,
        embedding_score=MethodScore(
            name="Embedding Similarity",
            score=emb_res["hallucination_score"],
            confidence=emb_res["confidence"],
            explanation=emb_res["explanation"],
            weight=WEIGHTS["embedding"]
        ),
        groundedness_score=MethodScore(
            name="Groundedness (LLM Judge)",
            score=grd_res["hallucination_score"],
            confidence=grd_res["confidence"],
            explanation=grd_res["explanation"],
            weight=WEIGHTS["groundedness"]
        ),
        consistency_score=MethodScore(
            name="Self-Consistency",
            score=cons_res["hallucination_score"],
            confidence=cons_res["confidence"],
            explanation=cons_res["explanation"],
            weight=WEIGHTS["consistency"]
        ),
        geval_score=MethodScore(
            name="G-Eval",
            score=gev_res["hallucination_score"],
            confidence=gev_res["confidence"],
            explanation=gev_res["explanation"],
            weight=WEIGHTS["geval"]
        ),
        ensemble_score=ensemble["ensemble_score"],
        verdict=ensemble["verdict"],
        processing_ms=elapsed_ms,
        audit_id=audit_id,
        fetched_context=fetched_context,
        context_sources=context_sources,
    )


# ── Mode 2: Document Audit (Existing) ───────────────────────────────────

@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Must be a PDF file.")
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")


@app.post("/api/doc-audit", response_model=DocAuditReport)
def run_doc_audit(req: DocAuditRequest):
    if not req.document.strip():
        raise HTTPException(status_code=400, detail="Document cannot be empty.")

    raw_claims = extract_claims(req.document)
    if not raw_claims:
        raise HTTPException(status_code=422, detail="No factual claims found.")

    claims: list[Claim] = []
    for i, raw in enumerate(raw_claims[:20]):
        claim_text = raw.get("claim", "").strip()
        orig_text  = raw.get("original_text", "").strip()
        if not claim_text: continue

        sources = search_sources(claim_text)
        verdict = score_claim(claim_text, sources)

        claims.append(Claim(
            id=i + 1,
            claim_text=claim_text,
            original_text=orig_text,
            status=verdict["status"],
            confidence=round(verdict["confidence"], 2),
            sources=sources,
            explanation=verdict["explanation"],
        ))

    total = len(claims)
    verified = sum(1 for c in claims if c.status == Verdict.VERIFIED)
    plausible = sum(1 for c in claims if c.status == Verdict.PLAUSIBLE)
    hallucination = sum(1 for c in claims if c.status == Verdict.HALLUCINATION)

    def pct(n): return round((n / total) * 100, 1) if total > 0 else 0.0
    trust_score = round(sum(c.confidence for c in claims) / total * 100, 1) if total > 0 else 0.0

    return DocAuditReport(
        document=req.document,
        total_claims=total,
        verified_count=verified,
        plausible_count=plausible,
        hallucination_count=hallucination,
        verified_pct=pct(verified),
        plausible_pct=pct(plausible),
        hallucination_pct=pct(hallucination),
        claims=claims,
        overall_trust_score=trust_score,
    )


# ── Audit Trail & Signals ───────────────────────────────────────────────

@app.get("/api/audit-trail", response_model=AuditTrailResponse)
async def get_trail(limit: int = 200):
    records = await get_all_audits(limit)
    formatted = []
    for r in records:
        formatted.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "question": r["question"],
            "context_preview": r["context"][:100] + "...",
            "answer_preview": r["answer"][:100] + "...",
            "topic": r["topic"],
            "embedding_score": round(r["embedding_score"], 2),
            "groundedness_score": round(r["groundedness_score"], 2),
            "consistency_score": round(r["consistency_score"], 2),
            "geval_score": round(r["geval_score"], 2),
            "ensemble_score": round(r["ensemble_score"], 2),
            "verdict": r["verdict"]
        })
    return AuditTrailResponse(records=formatted, total=len(formatted))


@app.get("/api/signals", response_model=SignalsResponse)
async def get_signals():
    records = await get_all_audits(1000)
    analysis = analyze_patterns(records)
    return SignalsResponse(**analysis)
