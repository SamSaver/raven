from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int | None = None
    section_title: str | None = None
    content_preview: str  # first ~200 chars
    relevance_score: float


class RetrievalResult(BaseModel):
    chunks: list[SourceReference]
    query_type: str | None = None
    transformed_query: str | None = None  # if HyDE/expansion was applied


class ChatResponse(BaseModel):
    answer: str
    citations: list[SourceReference] = Field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    conversation_id: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_connected: bool = False
    qdrant_connected: bool = False
    available_models: list[str] = Field(default_factory=list)
