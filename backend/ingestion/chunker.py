import uuid
from enum import Enum

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.ingestion.parser import ParsedDocument
from backend.models.documents import Chunk, ChunkMetadata, ChunkType

logger = structlog.get_logger()


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


def chunk_recursive(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def chunk_fixed(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def chunk_semantic(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    max_chunk_size: int = 1500,
    similarity_threshold: float = 0.75,
) -> list[str]:
    """Semantic chunking: split by sentences, merge by embedding similarity."""
    import numpy as np

    from backend.ingestion.embedder import get_model

    # Split into sentences
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    model = get_model(model_name)  # Uses cached model — no re-download
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = float(np.dot(current_embedding, embeddings[i]) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i]) + 1e-8
        ))

        merged_text = " ".join(current_chunk_sentences + [sentences[i]])
        if sim >= similarity_threshold and len(merged_text) <= max_chunk_size:
            current_chunk_sentences.append(sentences[i])
            # Update running average embedding
            current_embedding = np.mean(
                embeddings[i - len(current_chunk_sentences) + 1 : i + 1], axis=0
            )
        else:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def chunk_hierarchical(
    text: str,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 500,
    child_overlap: int = 50,
) -> list[dict]:
    """Hierarchical (parent-child) chunking.

    Creates large parent chunks and smaller child chunks within each parent.
    Child chunks are stored for retrieval (more precise matching).
    The parent content is preserved in metadata so it can be used as richer
    context during generation.

    Returns list of dicts with keys: content (child text), parent_content (parent text).
    """
    # Create parent chunks first (large, no overlap)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " "],
    )
    parent_chunks = parent_splitter.split_text(text)

    # Split each parent into children
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    results = []
    for parent_text in parent_chunks:
        children = child_splitter.split_text(parent_text)
        for child_text in children:
            results.append({
                "content": child_text,
                "parent_content": parent_text,
            })

    return results


def _split_sentences(text: str) -> list[str]:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    parsed_doc: ParsedDocument,
    doc_id: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[Chunk]:
    """Chunk a parsed document into Chunk objects with metadata."""
    logger.info("chunker.start", strategy=strategy, doc=parsed_doc.metadata.get("source"))

    # Select chunking function
    if strategy == ChunkingStrategy.HIERARCHICAL:
        hierarchical_results = chunk_hierarchical(
            parsed_doc.content,
            parent_chunk_size=chunk_size * 2,
            child_chunk_size=chunk_size,
            child_overlap=chunk_overlap,
        )

        chunks = []
        for item in hierarchical_results:
            text = item["content"]
            page_num = _estimate_page(text, parsed_doc.pages) if parsed_doc.pages else None
            chunk = Chunk(
                content=text,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    source=parsed_doc.metadata.get("source", "unknown"),
                    chunk_type=ChunkType.TEXT,
                    page_number=page_num,
                    section_title=item["parent_content"][:200],  # Store parent snippet for context
                    token_count=len(text.split()),
                    embedding_model=embedding_model,
                ),
            )
            chunks.append(chunk)

        # Add table chunks and return
        for table in parsed_doc.tables:
            chunk = Chunk(
                content=table["markdown"],
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    source=parsed_doc.metadata.get("source", "unknown"),
                    chunk_type=ChunkType.TABLE,
                    page_number=table.get("page"),
                    token_count=len(table["markdown"].split()),
                    embedding_model=embedding_model,
                ),
            )
            chunks.append(chunk)

        logger.info("chunker.done", strategy="hierarchical", chunk_count=len(chunks))
        return chunks

    if strategy == ChunkingStrategy.SEMANTIC:
        raw_chunks = chunk_semantic(parsed_doc.content, model_name=embedding_model)
    elif strategy == ChunkingStrategy.FIXED:
        raw_chunks = chunk_fixed(parsed_doc.content, chunk_size, chunk_overlap)
    else:
        raw_chunks = chunk_recursive(parsed_doc.content, chunk_size, chunk_overlap)

    # Build Chunk objects with metadata
    chunks = []
    for i, text in enumerate(raw_chunks):
        page_num = _estimate_page(text, parsed_doc.pages) if parsed_doc.pages else None
        chunk = Chunk(
            content=text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4()),
                source=parsed_doc.metadata.get("source", "unknown"),
                chunk_type=ChunkType.TEXT,
                page_number=page_num,
                token_count=len(text.split()),
                embedding_model=embedding_model,
            ),
        )
        chunks.append(chunk)

    # Add table chunks
    for table in parsed_doc.tables:
        chunk = Chunk(
            content=table["markdown"],
            metadata=ChunkMetadata(
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4()),
                source=parsed_doc.metadata.get("source", "unknown"),
                chunk_type=ChunkType.TABLE,
                page_number=table.get("page"),
                token_count=len(table["markdown"].split()),
                embedding_model=embedding_model,
            ),
        )
        chunks.append(chunk)

    logger.info("chunker.done", chunk_count=len(chunks))
    return chunks


def _estimate_page(chunk_text: str, pages: list[str]) -> int | None:
    """Estimate which page a chunk belongs to by overlap."""
    if not pages:
        return None
    best_page = 1
    best_overlap = 0
    chunk_words = set(chunk_text.lower().split()[:50])
    for i, page in enumerate(pages):
        page_words = set(page.lower().split())
        overlap = len(chunk_words & page_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = i + 1
    return best_page
