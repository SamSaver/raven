from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # ChromaDB (embedded, no Docker needed)
    chroma_collection: str = "raven_documents"
    chroma_persist_dir: Path = Path("./data/chroma")

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 dim; override for bge-large (1024)

    # Retrieval
    default_top_k: int = 10
    default_similarity_threshold: float = 0.5
    default_hybrid_weight: float = 0.7  # 0=pure BM25, 1=pure semantic
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Storage
    sqlite_db_path: Path = Path("./data/raven.db")
    cache_dir: Path = Path("./data/cache")
    upload_dir: Path = Path("./data/uploads")

    def ensure_directories(self) -> None:
        for d in [self.sqlite_db_path.parent, self.cache_dir, self.upload_dir, self.chroma_persist_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
