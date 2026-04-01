import structlog
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.models.documents import Chunk

logger = structlog.get_logger()

# Module-level model cache
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or settings.embedding_model
    if name not in _model_cache:
        logger.info("embedder.loading_model", model=name)
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
) -> list[list[float]]:
    model = get_model(model_name)
    # Process in batches to avoid OOM on large document sets
    embeddings = model.encode(
        texts,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
        batch_size=batch_size,
    )
    return embeddings.tolist()


def embed_query(query: str, model_name: str | None = None) -> list[float]:
    return embed_texts([query], model_name)[0]


def embed_chunks(chunks: list[Chunk], model_name: str | None = None) -> list[list[float]]:
    texts = [c.content for c in chunks]
    return embed_texts(texts, model_name)


def get_embedding_dimension(model_name: str | None = None) -> int:
    model = get_model(model_name)
    return model.get_sentence_embedding_dimension()
