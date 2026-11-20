from pydantic import BaseModel
from typing import Optional
from enum import Enum


# ── Shared Enums ──────────────────────────────────────────────

class Verdict(str, Enum):
    VERIFIED      = "verified"
    PLAUSIBLE     = "plausible"
    HALLUCINATION = "hallucination"
    PENDING       = "pending"


# ── Document Audit Mode (Mode 2) ──────────────────────────────

class Source(BaseModel):
    title:   str
    url:     str
    snippet: str

class Claim(BaseModel):
    id:            int
    claim_text:    str
    original_text: str
    status:        Verdict = Verdict.PENDING
    confidence:    float   = 0.0
    sources:       list[Source] = []
    explanation:   str = ""

class DocAuditRequest(BaseModel):
    document: str

class DocAuditReport(BaseModel):
    document:           str
    total_claims:       int
    verified_count:     int
    plausible_count:    int
    hallucination_count:int
    verified_pct:       float
    plausible_pct:      float
    hallucination_pct:  float
    claims:             list[Claim]
    overall_trust_score:float


# ── QA Evaluation Mode (Mode 1 — new) ────────────────────────

class QARequest(BaseModel):
    question: str
    context:  Optional[str] = None   # if None/empty, auto-fetched from web
    answer:   str
    topic:    Optional[str] = "general"

class MethodScore(BaseModel):
    name:        str
    score:       float          # 0.0 = no hallucination, 1.0 = definite hallucination
    confidence:  float
    explanation: str
    weight:      float

class QAReport(BaseModel):
    question:           str
    context:            str
    answer:             str
    topic:              str
    embedding_score:    MethodScore
    groundedness_score: MethodScore
    consistency_score:  MethodScore
    geval_score:        MethodScore
    ensemble_score:     float      # final weighted hallucination probability 0-1
    verdict:            Verdict
    processing_ms:      int
    audit_id:           Optional[int] = None
    fetched_context:    Optional[str] = None    # auto-fetched context string (if used)
    context_sources:    list[Source] = []       # web sources used as ground truth


# ── Audit Trail ───────────────────────────────────────────────

class AuditRecord(BaseModel):
    id:                 int
    timestamp:          str
    question:           str
    context_preview:    str
    answer_preview:     str
    topic:              str
    embedding_score:    float
    groundedness_score: float
    consistency_score:  float
    geval_score:        float
    ensemble_score:     float
    verdict:            str

class AuditTrailResponse(BaseModel):
    records: list[AuditRecord]
    total:   int


# ── Signals (Pattern Analysis) ────────────────────────────────

class SignalPattern(BaseModel):
    label:       str
    description: str
    value:       str

class SignalsResponse(BaseModel):
    total_audits:               int
    overall_hallucination_rate: float
    avg_ensemble_score:         float
    patterns:                   list[SignalPattern]
    topic_failure_rates:        dict[str, float]
    score_distribution:         dict[str, int]
