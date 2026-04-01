import json
import time

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.graph_rag import KnowledgeGraph
from backend.agents.planner import run_agent, run_agent_stream

logger = structlog.get_logger()
router = APIRouter(tags=["agent"])


class AgentRequest(BaseModel):
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    stream: bool = False


class AgentResponse(BaseModel):
    answer: str
    iterations: int
    tool_trace: list[dict]
    latency_ms: float


class GraphQueryRequest(BaseModel):
    entity: str
    max_hops: int = 2


class GraphPathRequest(BaseModel):
    source: str
    target: str


class GraphBuildRequest(BaseModel):
    doc_id: str | None = None
    rebuild: bool = False


@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    """Run the agentic RAG pipeline (Plan-Route-Act-Verify-Stop)."""
    if request.stream:
        return _stream_agent(request)

    start = time.time()
    result = run_agent(query=request.query, history=request.history or None)
    latency = (time.time() - start) * 1000

    return AgentResponse(
        answer=result["answer"],
        iterations=result["iterations"],
        tool_trace=result["tool_trace"],
        latency_ms=round(latency, 1),
    )


def _stream_agent(request: AgentRequest):
    """Stream agent execution steps as Server-Sent Events."""

    def event_stream():
        for step in run_agent_stream(query=request.query, history=request.history or None):
            yield f"data: {json.dumps(step)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- GraphRAG endpoints ---

@router.get("/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    kg = KnowledgeGraph()
    return kg.stats()


@router.post("/graph/query")
async def graph_query(request: GraphQueryRequest):
    """Query the knowledge graph for an entity and its neighborhood."""
    kg = KnowledgeGraph()
    result = kg.query_neighbors(request.entity, max_hops=request.max_hops)
    return result


@router.post("/graph/path")
async def graph_find_path(request: GraphPathRequest):
    """Find the shortest path between two entities in the knowledge graph."""
    kg = KnowledgeGraph()
    path = kg.find_path(request.source, request.target)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between '{request.source}' and '{request.target}'",
        )
    return {"source": request.source, "target": request.target, "path": path}


@router.get("/graph/communities")
async def graph_communities():
    """Get community detection results with optional LLM summaries."""
    kg = KnowledgeGraph()
    communities = kg.detect_communities()
    return {
        "total_communities": len(communities),
        "communities": [
            {"id": i, "size": len(c), "entities": c[:20]}
            for i, c in enumerate(communities[:10])
        ],
    }


@router.post("/graph/communities/summarize")
async def graph_community_summaries():
    """Generate LLM summaries for top communities."""
    kg = KnowledgeGraph()
    summaries = kg.get_community_summaries(max_communities=5)
    return {"summaries": summaries}


@router.post("/graph/search")
async def graph_search(query: str):
    """Search the knowledge graph for entities matching a query."""
    kg = KnowledgeGraph()
    results = kg.graph_search(query, top_k=10)
    return {"results": results}


@router.post("/graph/build")
async def build_graph(request: GraphBuildRequest):
    """Build/extend the knowledge graph from ingested documents.

    Reads chunks from Qdrant and extracts entities/relationships via LLM.
    """
    from backend.storage.database import Database
    from backend.storage.vector import VectorStore

    db = Database()
    vs = VectorStore()
    kg = KnowledgeGraph()

    if request.rebuild:
        kg.graph.clear()

    # Get documents to process
    docs = db.list_documents()
    if request.doc_id:
        docs = [d for d in docs if d["doc_id"] == request.doc_id]

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found")

    total_entities = 0
    total_relationships = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        # Retrieve chunks from Qdrant
        from backend.ingestion.embedder import embed_query

        # Use a generic query to get all chunks for this doc
        results = vs.search(
            query_vector=embed_query("document content"),
            top_k=100,
            doc_id_filter=doc_id,
        )

        for chunk in results:
            content = chunk.get("content", "")
            chunk_id = chunk.get("_chunk_id", "")
            if len(content) < 50:
                continue

            counts = kg.extract_and_add(content, doc_id, chunk_id)
            total_entities += counts["entities"]
            total_relationships += counts["relationships"]

    kg.save()

    return {
        "status": "success",
        "documents_processed": len(docs),
        "entities_extracted": total_entities,
        "relationships_extracted": total_relationships,
        "graph_stats": kg.stats(),
    }
