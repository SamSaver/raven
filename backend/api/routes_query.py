import json
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.generation.citations import compute_confidence, extract_citations
from backend.generation.context import assemble_messages
from backend.generation.llm import chat, chat_stream, list_models
from backend.models.queries import ChatRequest, QueryRequest, QueryType
from backend.models.responses import ChatResponse, RetrievalResult, SourceReference
from backend.retrieval.hybrid import HybridSearcher
from backend.retrieval.postprocess import postprocess_results
from backend.retrieval.query import QueryProcessor
from backend.retrieval.reranker import rerank
from backend.ingestion.embedder import embed_query
from backend.storage.database import Database

logger = structlog.get_logger()
router = APIRouter(tags=["query"])

db = Database()
searcher = HybridSearcher()


def _run_retrieval_pipeline(
    query: str,
    config: dict,
    enable_query_transform: bool = True,
) -> tuple[list[dict], dict]:
    """Run the full retrieval pipeline: query transform → hybrid search → GraphRAG → rerank → postprocess."""
    query_info = {"original_query": query, "query_type": QueryType.FACTUAL, "search_queries": [query]}

    if enable_query_transform:
        processor = QueryProcessor(enable_classification=True, enable_hyde=False)
        query_info = processor.process(query)

    # Search with multiple queries if expanded/decomposed
    search_queries = query_info.get("search_queries", [query])
    hyde_doc = query_info.get("hyde_document")

    if len(search_queries) > 1:
        results = searcher.multi_query_search(
            queries=search_queries,
            top_k=config.get("top_k", settings.default_top_k),
            similarity_threshold=config.get("similarity_threshold"),
            hybrid_weight=config.get("hybrid_weight"),
        )
    else:
        results = searcher.search(
            query=search_queries[0],
            top_k=config.get("top_k", settings.default_top_k),
            similarity_threshold=config.get("similarity_threshold"),
            hybrid_weight=config.get("hybrid_weight"),
            hyde_document=hyde_doc,
        )

    # GraphRAG augmentation: find related chunks via knowledge graph
    try:
        from backend.agents.graph_rag import KnowledgeGraph
        kg = KnowledgeGraph()
        if kg.graph.number_of_nodes() > 0:
            graph_results = kg.graph_search(query, top_k=3)
            if graph_results:
                # Fetch the actual chunk content from Qdrant for graph-found chunks
                existing_chunk_ids = {r.get("_chunk_id") for r in results}
                for gr in graph_results:
                    if gr.get("chunk_id") and gr["chunk_id"] not in existing_chunk_ids:
                        # Search Qdrant for this specific chunk by doc_id
                        graph_hits = searcher.vector_store.search(
                            query_vector=embed_query(query),
                            top_k=1,
                            doc_id_filter=gr.get("doc_id"),
                        )
                        for hit in graph_hits:
                            hit["graph_boosted"] = True
                            hit["graph_entity"] = gr.get("entity", "")
                            results.append(hit)
                logger.info("retrieval.graph_augmented", graph_chunks=len(graph_results))
    except Exception as e:
        logger.debug("retrieval.graph_skipped", reason=str(e))

    # Rerank
    if config.get("reranker_enabled", settings.reranker_enabled) and results:
        results = rerank(query, results, top_k=config.get("top_k", settings.default_top_k))

    # Post-process
    results = postprocess_results(
        results,
        max_context_tokens=config.get("max_context_tokens", 4096),
    )

    return results, query_info


@router.post("/query", response_model=RetrievalResult)
async def query_documents(request: QueryRequest):
    """Retrieve relevant chunks for a query (no generation)."""
    start = time.time()

    config = request.retrieval_config.model_dump()
    results, query_info = _run_retrieval_pipeline(request.query, config)

    sources = [
        SourceReference(
            chunk_id=r.get("_chunk_id", ""),
            doc_id=r.get("doc_id", ""),
            filename=r.get("source", "unknown"),
            page_number=r.get("page_number"),
            section_title=r.get("section_title"),
            content_preview=r.get("content", "")[:200],
            relevance_score=r.get("rerank_score", r.get("score", r.get("rrf_score", 0.0))),
        )
        for r in results
    ]

    latency = (time.time() - start) * 1000
    logger.info("query.complete", latency_ms=round(latency, 1), results=len(sources))

    return RetrievalResult(
        chunks=sources,
        query_type=query_info.get("query_type", QueryType.FACTUAL).value,
        transformed_query=str(query_info.get("search_queries", [request.query])),
    )


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Full RAG pipeline: retrieve → generate → cite. Supports streaming."""
    start = time.time()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    model = request.model or settings.ollama_model

    # Retrieval
    config = request.retrieval_config.model_dump()
    results, query_info = _run_retrieval_pipeline(request.query, config)
    query_type = query_info.get("query_type", QueryType.FACTUAL)

    # Assemble LLM messages
    messages = assemble_messages(
        query=request.query,
        results=results,
        query_type=query_type,
        history=request.history or None,
    )

    if request.stream:
        return _stream_response(
            messages, model, results, conversation_id, query_type, start,
            original_query=request.query, retrieval_config=config,
        )

    # Non-streaming
    answer = chat(messages, model=model)
    latency = (time.time() - start) * 1000

    citations = extract_citations(answer, results)
    confidence = compute_confidence(results, answer)

    # Log query
    db.log_query(
        query=request.query,
        query_type=query_type.value,
        model_used=model,
        retrieval_config=config,
        response_preview=answer[:200],
        latency_ms=latency,
        confidence=confidence,
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        model_used=model,
        conversation_id=conversation_id,
    )


def _stream_response(
    messages: list[dict],
    model: str,
    results: list[dict],
    conversation_id: str,
    query_type: QueryType,
    start_time: float,
    original_query: str = "",
    retrieval_config: dict | None = None,
):
    """Stream the LLM response as Server-Sent Events."""

    def event_stream():
        # Send sources first
        sources = [
            {
                "chunk_id": r.get("_chunk_id", ""),
                "doc_id": r.get("doc_id", ""),
                "filename": r.get("source", "unknown"),
                "page_number": r.get("page_number"),
                "content_preview": r.get("content", "")[:200],
                "relevance_score": r.get("rerank_score", r.get("score", r.get("rrf_score", 0.0))),
            }
            for r in results
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        # Stream answer tokens
        full_answer = []
        for token in chat_stream(messages, model=model):
            full_answer.append(token)
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        # Send final metadata
        answer_text = "".join(full_answer)
        citations = extract_citations(answer_text, results)
        confidence = compute_confidence(results, answer_text)
        latency = (time.time() - start_time) * 1000

        # Log query to SQLite (was previously missing for streaming!)
        try:
            db.log_query(
                query=original_query,
                query_type=query_type.value if hasattr(query_type, "value") else str(query_type),
                model_used=model,
                retrieval_config=retrieval_config,
                response_preview=answer_text[:200],
                latency_ms=latency,
                confidence=confidence,
            )
        except Exception as e:
            logger.warning("stream.log_query_failed", error=str(e))

        final = {
            "type": "done",
            "data": {
                "confidence": confidence,
                "model_used": model,
                "conversation_id": conversation_id,
                "latency_ms": round(latency, 1),
                "citations": [c.model_dump() for c in citations],
            },
        }
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/models")
async def get_available_models():
    """List available Ollama models."""
    models = list_models()
    return {"models": models, "default": settings.ollama_model}
