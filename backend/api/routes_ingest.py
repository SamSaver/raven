import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import structlog
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.config import settings
from backend.ingestion.chunker import ChunkingStrategy, chunk_document
from backend.ingestion.embedder import embed_chunks, get_embedding_dimension
from backend.ingestion.parser import parse_document, save_upload
from backend.models.documents import DocumentListItem, IngestResponse
from backend.storage.database import Database
from backend.storage.vector import VectorStore

logger = structlog.get_logger()
router = APIRouter(tags=["ingestion"])

db = Database()
vector_store = VectorStore()

_executor = ThreadPoolExecutor(max_workers=2)

# In-memory task status tracker for background ingestion
_task_status: dict[str, dict] = {}


def _do_ingest(
    file_path: Path,
    doc_id: str,
    filename: str,
    file_size: int,
    chunking_strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
    task_id: str | None = None,
) -> IngestResponse:
    """Synchronous ingestion work — runs in thread pool."""
    try:
        if task_id:
            _task_status[task_id]["status"] = "parsing"

        parsed = parse_document(file_path)

        if task_id:
            _task_status[task_id]["status"] = "chunking"

        chunks = chunk_document(
            parsed_doc=parsed,
            doc_id=doc_id,
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=settings.embedding_model,
        )

        if not chunks:
            raise ValueError("No content could be extracted from file")

        if task_id:
            _task_status[task_id]["status"] = "embedding"
            _task_status[task_id]["chunk_count"] = len(chunks)

        vectors = embed_chunks(chunks, settings.embedding_model)

        dim = get_embedding_dimension(settings.embedding_model)
        vector_store.ensure_collection(dimension=dim)

        if task_id:
            _task_status[task_id]["status"] = "storing"

        chunk_ids = [c.metadata.chunk_id for c in chunks]
        payloads = [
            {
                "doc_id": doc_id,
                "content": c.content,
                "source": c.metadata.source,
                "chunk_type": c.metadata.chunk_type.value,
                "page_number": c.metadata.page_number,
                "section_title": c.metadata.section_title,
                "token_count": c.metadata.token_count,
            }
            for c in chunks
        ]
        vector_store.upsert_chunks(chunk_ids, vectors, payloads)

        db.insert_document(
            doc_id=doc_id,
            filename=filename,
            file_type=parsed.metadata.get("file_type", "unknown"),
            file_size=file_size,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
        )

        result = IngestResponse(doc_id=doc_id, filename=filename, chunk_count=len(chunks))
        if task_id:
            _task_status[task_id]["status"] = "completed"
            _task_status[task_id]["result"] = result.model_dump()
        return result

    except Exception as e:
        if task_id:
            _task_status[task_id]["status"] = "failed"
            _task_status[task_id]["error"] = str(e)
        raise


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Query(default=ChunkingStrategy.RECURSIVE),
    chunk_size: int = Query(default=1000, ge=100, le=10000),
    chunk_overlap: int = Query(default=200, ge=0, le=2000),
    background: bool = Query(default=False, description="Run ingestion in background and return task ID"),
):
    doc_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    content = await file.read()
    file_path = await save_upload(content, file.filename, settings.upload_dir)

    _task_status[task_id] = {
        "task_id": task_id,
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "queued",
        "chunk_count": 0,
        "error": None,
        "result": None,
    }

    if background:
        # Fire-and-forget: return task_id immediately, client polls /ingest/status/{task_id}
        def _run():
            _do_ingest(file_path, doc_id, file.filename, len(content),
                       chunking_strategy, chunk_size, chunk_overlap, task_id)
        threading.Thread(target=_run, daemon=True).start()
        return {"task_id": task_id, "status": "queued", "doc_id": doc_id}

    # Synchronous mode: wait for completion (still in thread pool so event loop is free)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _do_ingest,
            file_path, doc_id, file.filename, len(content),
            chunking_strategy, chunk_size, chunk_overlap, task_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("ingest.failed", error=str(e), file=file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str):
    """Poll ingestion task progress."""
    if task_id not in _task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_status[task_id]


@router.get("/documents", response_model=list[DocumentListItem])
async def list_documents():
    docs = db.list_documents()
    return [
        DocumentListItem(
            doc_id=d["doc_id"],
            filename=d["filename"],
            file_type=d["file_type"],
            chunk_count=d["chunk_count"],
            upload_time=d["upload_time"],
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store.delete_by_doc_id(doc_id)
    db.delete_document(doc_id)

    file_path = settings.upload_dir / doc["filename"]
    if file_path.exists():
        file_path.unlink()

    return {"status": "deleted", "doc_id": doc_id}


@router.get("/collections/info")
async def collection_info():
    return vector_store.get_collection_info()
