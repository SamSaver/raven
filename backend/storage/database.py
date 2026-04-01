import json
import sqlite3
from datetime import datetime
from pathlib import Path

import structlog

from backend.config import settings

logger = structlog.get_logger()

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER DEFAULT 0,
    upload_time TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    query_type TEXT,
    model_used TEXT,
    retrieval_config_json TEXT,
    response_preview TEXT,
    latency_ms REAL,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    eval_id TEXT PRIMARY KEY,
    eval_type TEXT NOT NULL,
    test_set_id TEXT,
    total_samples INTEGER,
    aggregate_metrics_json TEXT NOT NULL,
    per_sample_json TEXT,
    model_used TEXT,
    duration_seconds REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_log_id INTEGER REFERENCES query_logs(id),
    rating INTEGER CHECK(rating IN (-1, 1)),
    comment TEXT,
    created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(CREATE_TABLES)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def insert_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        page_count: int | None,
        chunk_count: int,
        metadata: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents (doc_id, filename, file_type, file_size, page_count,
                   chunk_count, upload_time, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    filename,
                    file_type,
                    file_size,
                    page_count,
                    chunk_count,
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, file_type, file_size, chunk_count, upload_time "
                "FROM documents ORDER BY upload_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def log_query(
        self,
        query: str,
        query_type: str | None,
        model_used: str | None,
        retrieval_config: dict | None,
        response_preview: str | None,
        latency_ms: float | None,
        confidence: float | None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO query_logs (query, query_type, model_used, retrieval_config_json,
                   response_preview, latency_ms, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query,
                    query_type,
                    model_used,
                    json.dumps(retrieval_config or {}),
                    response_preview,
                    latency_ms,
                    confidence,
                    datetime.utcnow().isoformat(),
                ),
            )
            return cursor.lastrowid

    def add_feedback(self, query_log_id: int, rating: int, comment: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (query_log_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
                (query_log_id, rating, comment, datetime.utcnow().isoformat()),
            )

    # ── Evaluation results ──────────────────────────────

    def save_eval_result(
        self,
        eval_id: str,
        eval_type: str,
        aggregate_metrics: dict,
        total_samples: int = 0,
        test_set_id: str | None = None,
        per_sample: list | None = None,
        model_used: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO eval_results
                   (eval_id, eval_type, test_set_id, total_samples,
                    aggregate_metrics_json, per_sample_json, model_used,
                    duration_seconds, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eval_id, eval_type, test_set_id, total_samples,
                    json.dumps(aggregate_metrics),
                    json.dumps(per_sample) if per_sample else None,
                    model_used, duration_seconds,
                    datetime.utcnow().isoformat(),
                ),
            )

    def list_eval_results(self, eval_type: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if eval_type:
                rows = conn.execute(
                    "SELECT eval_id, eval_type, test_set_id, total_samples, "
                    "aggregate_metrics_json, model_used, duration_seconds, created_at "
                    "FROM eval_results WHERE eval_type = ? ORDER BY created_at DESC",
                    (eval_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT eval_id, eval_type, test_set_id, total_samples, "
                    "aggregate_metrics_json, model_used, duration_seconds, created_at "
                    "FROM eval_results ORDER BY created_at DESC"
                ).fetchall()

            results = []
            for r in rows:
                d = dict(r)
                d["aggregate_metrics"] = json.loads(d.pop("aggregate_metrics_json", "{}"))
                results.append(d)
            return results

    def get_eval_result(self, eval_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM eval_results WHERE eval_id = ?", (eval_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["aggregate_metrics"] = json.loads(d.pop("aggregate_metrics_json", "{}"))
            d["per_sample"] = json.loads(d.pop("per_sample_json", "null"))
            return d
