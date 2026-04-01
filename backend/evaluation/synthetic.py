"""Synthetic test set generation for RAG evaluation.

Generates question-answer pairs from ingested documents using Ollama.
These synthetic QA pairs serve as regression tests and evaluation datasets
without requiring manual annotation.
"""

import json
import time
import uuid
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from backend.config import settings
from backend.generation.llm import generate
from backend.storage.database import Database

logger = structlog.get_logger()

QA_GENERATION_PROMPT = """Given the following text passage, generate {count} question-answer pairs.
Each question should be answerable ONLY from the provided text.
Generate diverse question types: factual, analytical, and inferential.

Return a JSON array where each element has:
- "question": the question string
- "answer": the ground-truth answer from the text
- "type": one of "factual", "analytical", "inferential"
- "difficulty": one of "easy", "medium", "hard"

Text passage:
---
{text}
---

Return ONLY valid JSON array, no explanation."""

MULTI_HOP_PROMPT = """Given the following two text passages, generate {count} questions that
require information from BOTH passages to answer correctly.

Return a JSON array where each element has:
- "question": the multi-hop question
- "answer": the ground-truth answer combining info from both passages
- "type": "multi_hop"
- "difficulty": "hard"
- "required_passages": [1, 2]

Passage 1:
---
{text1}
---

Passage 2:
---
{text2}
---

Return ONLY valid JSON array, no explanation."""


class SyntheticQA(BaseModel):
    qa_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    qa_type: str = "factual"
    difficulty: str = "medium"
    source_doc_id: str = ""
    source_chunks: list[str] = Field(default_factory=list)
    context_texts: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SyntheticTestSet(BaseModel):
    test_set_id: str = Field(default_factory=lambda: f"testset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    samples: list[SyntheticQA] = Field(default_factory=list)
    total_samples: int = 0
    generation_model: str = ""
    duration_seconds: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def _parse_qa_json(response: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage truncated JSON by closing open brackets
        salvaged = text.rstrip()
        # If it ends mid-object, try to close it
        if salvaged and not salvaged.endswith("]"):
            # Remove trailing incomplete object
            last_brace = salvaged.rfind("}")
            if last_brace > 0:
                salvaged = salvaged[:last_brace + 1] + "]"
                try:
                    return json.loads(salvaged)
                except json.JSONDecodeError:
                    pass
        logger.warning("synthetic.json_parse_failed", response=text[:200])
        return []


def generate_qa_from_chunk(
    text: str,
    doc_id: str = "",
    chunk_id: str = "",
    count: int = 3,
) -> list[SyntheticQA]:
    """Generate QA pairs from a single text chunk."""
    if len(text.strip()) < 100:
        return []

    prompt = QA_GENERATION_PROMPT.format(count=count, text=text[:3000])

    try:
        response = generate(prompt, temperature=0.4, max_tokens=2048)
        qa_list = _parse_qa_json(response)
    except Exception as e:
        logger.warning("synthetic.generation_failed", error=str(e))
        return []

    results = []
    for qa in qa_list:
        if not qa.get("question") or not qa.get("answer"):
            continue
        results.append(SyntheticQA(
            question=qa["question"],
            answer=qa["answer"],
            qa_type=qa.get("type", "factual"),
            difficulty=qa.get("difficulty", "medium"),
            source_doc_id=doc_id,
            source_chunks=[chunk_id] if chunk_id else [],
            context_texts=[text],
        ))

    return results


def generate_multi_hop_qa(
    text1: str,
    text2: str,
    doc_id: str = "",
    chunk_ids: list[str] | None = None,
    count: int = 2,
) -> list[SyntheticQA]:
    """Generate multi-hop QA pairs requiring both passages."""
    if len(text1.strip()) < 100 or len(text2.strip()) < 100:
        return []

    prompt = MULTI_HOP_PROMPT.format(count=count, text1=text1[:2000], text2=text2[:2000])

    try:
        response = generate(prompt, temperature=0.4, max_tokens=1000)
        qa_list = _parse_qa_json(response)
    except Exception as e:
        logger.warning("synthetic.multi_hop_failed", error=str(e))
        return []

    results = []
    for qa in qa_list:
        if not qa.get("question") or not qa.get("answer"):
            continue
        results.append(SyntheticQA(
            question=qa["question"],
            answer=qa["answer"],
            qa_type="multi_hop",
            difficulty="hard",
            source_doc_id=doc_id,
            source_chunks=chunk_ids or [],
            context_texts=[text1, text2],
        ))

    return results


def generate_test_set(
    doc_id: str | None = None,
    max_chunks: int = 20,
    qa_per_chunk: int = 3,
    include_multi_hop: bool = True,
) -> SyntheticTestSet:
    """Generate a full synthetic test set from ingested documents.

    Args:
        doc_id: Specific document to generate from. If None, uses all documents.
        max_chunks: Maximum number of chunks to process.
        qa_per_chunk: Number of QA pairs per chunk.
        include_multi_hop: Whether to generate multi-hop questions from chunk pairs.
    """
    from backend.ingestion.embedder import embed_query
    from backend.storage.vector import VectorStore

    start = time.time()
    vs = VectorStore()
    db = Database()

    # Check collection exists first
    collection_info = vs.get_collection_info()
    if collection_info.get("points_count", 0) == 0:
        logger.warning("synthetic.empty_collection")
        return SyntheticTestSet(
            generation_model=settings.ollama_model,
            total_samples=0,
        )

    # Get chunks from Qdrant
    try:
        query_vector = embed_query("document content overview")
        chunks = vs.search(
            query_vector=query_vector,
            top_k=max_chunks,
            doc_id_filter=doc_id,
        )
    except Exception as e:
        logger.error("synthetic.search_failed", error=str(e))
        return SyntheticTestSet(generation_model=settings.ollama_model, total_samples=0)

    if not chunks:
        logger.warning("synthetic.no_chunks_found", doc_id=doc_id)
        return SyntheticTestSet(generation_model=settings.ollama_model, total_samples=0)

    all_qa: list[SyntheticQA] = []

    # Generate single-chunk QA pairs
    for chunk in chunks:
        content = chunk.get("content", "")
        c_doc_id = chunk.get("doc_id", "")
        c_chunk_id = chunk.get("_chunk_id", "")

        qa_pairs = generate_qa_from_chunk(
            text=content,
            doc_id=c_doc_id,
            chunk_id=c_chunk_id,
            count=qa_per_chunk,
        )
        all_qa.extend(qa_pairs)
        logger.info("synthetic.chunk_done", chunk_id=c_chunk_id[:8], qa_count=len(qa_pairs))

    # Generate multi-hop QA from chunk pairs
    if include_multi_hop and len(chunks) >= 2:
        import random
        pairs = []
        for i in range(min(5, len(chunks) - 1)):
            j = random.randint(i + 1, len(chunks) - 1)
            pairs.append((chunks[i], chunks[j]))

        for c1, c2 in pairs:
            multi_qa = generate_multi_hop_qa(
                text1=c1.get("content", ""),
                text2=c2.get("content", ""),
                doc_id=c1.get("doc_id", ""),
                chunk_ids=[c1.get("_chunk_id", ""), c2.get("_chunk_id", "")],
                count=2,
            )
            all_qa.extend(multi_qa)

    duration = time.time() - start

    test_set = SyntheticTestSet(
        samples=all_qa,
        total_samples=len(all_qa),
        generation_model=settings.ollama_model,
        duration_seconds=round(duration, 1),
    )

    logger.info(
        "synthetic.test_set_complete",
        total=len(all_qa),
        single_hop=sum(1 for q in all_qa if q.qa_type != "multi_hop"),
        multi_hop=sum(1 for q in all_qa if q.qa_type == "multi_hop"),
        duration=round(duration, 1),
    )

    # Persist to SQLite
    _save_test_set(test_set)

    return test_set


def _save_test_set(test_set: SyntheticTestSet) -> None:
    """Save test set to SQLite for tracking over time."""
    db = Database()
    with db._connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS synthetic_test_sets (
                test_set_id TEXT PRIMARY KEY,
                total_samples INTEGER,
                generation_model TEXT,
                duration_seconds REAL,
                data_json TEXT,
                created_at TEXT
            )"""
        )
        conn.execute(
            "INSERT OR REPLACE INTO synthetic_test_sets VALUES (?, ?, ?, ?, ?, ?)",
            (
                test_set.test_set_id,
                test_set.total_samples,
                test_set.generation_model,
                test_set.duration_seconds,
                test_set.model_dump_json(),
                test_set.created_at,
            ),
        )


def load_test_set(test_set_id: str) -> SyntheticTestSet | None:
    """Load a saved test set from SQLite."""
    db = Database()
    with db._connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS synthetic_test_sets (
                test_set_id TEXT PRIMARY KEY,
                total_samples INTEGER,
                generation_model TEXT,
                duration_seconds REAL,
                data_json TEXT,
                created_at TEXT
            )"""
        )
        row = conn.execute(
            "SELECT data_json FROM synthetic_test_sets WHERE test_set_id = ?",
            (test_set_id,),
        ).fetchone()

    if not row:
        return None

    return SyntheticTestSet.model_validate_json(row["data_json"])


def list_test_sets() -> list[dict]:
    """List all saved test sets."""
    db = Database()
    with db._connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS synthetic_test_sets (
                test_set_id TEXT PRIMARY KEY,
                total_samples INTEGER,
                generation_model TEXT,
                duration_seconds REAL,
                data_json TEXT,
                created_at TEXT
            )"""
        )
        rows = conn.execute(
            "SELECT test_set_id, total_samples, generation_model, duration_seconds, created_at "
            "FROM synthetic_test_sets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
