from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    MULTI_HOP = "multi_hop"


class RetrievalConfig(BaseModel):
    top_k: int = 10
    similarity_threshold: float = 0.5
    hybrid_weight: float = 0.7  # 0=pure BM25, 1=pure semantic
    reranker_enabled: bool = True
    max_context_tokens: int = 4096


class QueryRequest(BaseModel):
    query: str
    collection: str | None = None
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    model: str | None = None  # override default Ollama model
    stream: bool = False


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)  # [{"role": ..., "content": ...}]
    collection: str | None = None
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    model: str | None = None
    stream: bool = True
