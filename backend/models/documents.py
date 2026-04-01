import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    file_size: int
    page_count: int | None = None
    upload_time: datetime = Field(default_factory=datetime.utcnow)


class ChunkMetadata(BaseModel):
    doc_id: str
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    chunk_type: ChunkType = ChunkType.TEXT
    page_number: int | None = None
    section_title: str | None = None
    token_count: int = 0
    embedding_model: str = ""
    custom_tags: list[str] = Field(default_factory=list)
    extraction_confidence: float = 1.0


class Chunk(BaseModel):
    content: str
    metadata: ChunkMetadata


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str = "success"


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    upload_time: datetime
