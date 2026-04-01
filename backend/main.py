import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models.responses import HealthResponse

logger = structlog.get_logger()

app = FastAPI(
    title="Raven RAG",
    description="Zero-budget, production-grade RAG system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    settings.ensure_directories()
    logger.info("raven.startup", host=settings.api_host, port=settings.api_port)

    # Preload embedding model at startup so first ingestion/query is fast.
    # This avoids the "loading weights" delay on every request.
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _preload():
        try:
            from backend.ingestion.embedder import get_model
            get_model()
            logger.info("startup.embedding_model_loaded", model=settings.embedding_model)
        except Exception as e:
            logger.warning("startup.embedding_preload_failed", error=str(e))

    loop = asyncio.get_event_loop()
    loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _preload)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    ollama_ok = False
    available_models: list[str] = []
    chroma_ok = False

    # Check Ollama
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        models = client.list()
        available_models = [m.model for m in models.models]
        ollama_ok = True
    except Exception as e:
        logger.warning("ollama.health_check_failed", error=str(e))

    # Check ChromaDB (embedded — always available if import works)
    try:
        from backend.storage.vector import VectorStore
        vs = VectorStore()
        info = vs.get_collection_info()
        chroma_ok = "error" not in str(info.get("status", ""))
    except Exception as e:
        logger.warning("chroma.health_check_failed", error=str(e))

    return HealthResponse(
        status="ok" if (ollama_ok and chroma_ok) else "degraded",
        ollama_connected=ollama_ok,
        qdrant_connected=chroma_ok,  # Reusing field name for backward compat
        available_models=available_models,
    )


# Import and register routers
from backend.api.routes_ingest import router as ingest_router
from backend.api.routes_query import router as query_router
from backend.api.routes_eval import router as eval_router
from backend.api.routes_admin import router as admin_router
from backend.api.routes_agent import router as agent_router

app.include_router(ingest_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(eval_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
