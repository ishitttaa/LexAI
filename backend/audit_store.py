"""
Audit Trail Store — SQLite-backed persistent storage for all QA evaluations.
Supports async reads/writes via aiosqlite.
"""
import aiosqlite
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "audit_trail.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS qa_audits (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    question            TEXT NOT NULL,
    context             TEXT NOT NULL,
    answer              TEXT NOT NULL,
    topic               TEXT DEFAULT 'general',
    embedding_score     REAL DEFAULT 0.0,
    groundedness_score  REAL DEFAULT 0.0,
    consistency_score   REAL DEFAULT 0.0,
    geval_score         REAL DEFAULT 0.0,
    ensemble_score      REAL DEFAULT 0.0,
    verdict             TEXT DEFAULT 'pending'
);
"""


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE_SQL)
        await db.commit()


async def save_audit(
    question: str,
    context: str,
    answer: str,
    topic: str,
    embedding_score: float,
    groundedness_score: float,
    consistency_score: float,
    geval_score: float,
    ensemble_score: float,
    verdict: str,
) -> int:
    """Insert a new audit record and return its ID."""
    ts = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO qa_audits
               (timestamp, question, context, answer, topic,
                embedding_score, groundedness_score, consistency_score,
                geval_score, ensemble_score, verdict)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, question, context, answer, topic,
             embedding_score, groundedness_score, consistency_score,
             geval_score, ensemble_score, verdict),
        )
        await db.commit()
        return cursor.lastrowid


async def get_all_audits(limit: int = 200) -> list[dict]:
    """Fetch all audit records ordered by most recent first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM qa_audits ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_audit_count() -> int:
    """Return total number of audit records."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM qa_audits")
        row = await cursor.fetchone()
        return row[0] if row else 0
