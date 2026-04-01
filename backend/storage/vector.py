"""Vector store backed by ChromaDB (embedded, no Docker required).

Provides the same interface as the previous Qdrant implementation:
  ensure_collection(), upsert_chunks(), search(), delete_by_doc_id(), get_collection_info()
"""

from pathlib import Path

import chromadb
import structlog

from backend.config import settings

logger = structlog.get_logger()


class VectorStore:
    def __init__(self, collection: str | None = None, persist_dir: str | None = None):
        self.collection_name = collection or settings.chroma_collection
        persist = persist_dir or str(settings.chroma_persist_dir)

        # Persistent client — data survives restarts, stored on disk
        self.client = chromadb.PersistentClient(path=persist)
        self._collection = None

    def _get_collection(self):
        """Lazy-get the collection, creating it if needed."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ensure_collection(self, dimension: int | None = None) -> None:
        """Ensure the collection exists (ChromaDB creates on first use)."""
        self._get_collection()
        logger.info("chroma.collection_ready", name=self.collection_name)

    def upsert_chunks(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        collection = self._get_collection()

        # ChromaDB uses string IDs natively
        chroma_ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk_id, vec, payload in zip(ids, vectors, payloads):
            chroma_ids.append(chunk_id)
            embeddings.append(vec)

            # ChromaDB stores the document text separately from metadata
            doc_text = payload.pop("content", "")
            documents.append(doc_text)

            # Flatten metadata — ChromaDB only supports str/int/float/bool values
            meta = {"_chunk_id": chunk_id}
            for k, v in payload.items():
                if v is None:
                    meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        collection.upsert(
            ids=chroma_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("chroma.upserted", count=len(chroma_ids), collection=self.collection_name)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        doc_id_filter: str | None = None,
    ) -> list[dict]:
        collection = self._get_collection()

        where_filter = None
        if doc_id_filter:
            where_filter = {"doc_id": doc_id_filter}

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        except Exception as e:
            # Collection might be empty
            logger.warning("chroma.search_failed", error=str(e))
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances (lower = more similar for cosine)
            # Convert to similarity score: score = 1 - distance
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1.0 - distance

            if score_threshold is not None and score < score_threshold:
                continue

            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            content = results["documents"][0][i] if results["documents"] else ""

            hit = {
                "score": score,
                "content": content,
                **meta,
            }
            hits.append(hit)

        return hits

    def delete_by_doc_id(self, doc_id: str) -> None:
        collection = self._get_collection()
        try:
            # Get IDs matching this doc_id
            results = collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info("chroma.deleted", doc_id=doc_id, count=len(results["ids"]))
        except Exception as e:
            logger.warning("chroma.delete_failed", doc_id=doc_id, error=str(e))

    def get_collection_info(self) -> dict:
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "name": self.collection_name,
                "points_count": count,
                "status": "ready",
            }
        except Exception as e:
            logger.warning("chroma.collection_info_failed", error=str(e))
            return {"name": self.collection_name, "points_count": 0, "status": f"error: {e}"}
