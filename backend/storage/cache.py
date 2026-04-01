import hashlib

import structlog
from diskcache import Cache

from backend.config import settings

logger = structlog.get_logger()


class ResponseCache:
    def __init__(self, cache_dir: str | None = None):
        self.cache = Cache(str(cache_dir or settings.cache_dir))

    @staticmethod
    def _make_key(query: str, config_hash: str) -> str:
        raw = f"{query}:{config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, config_hash: str = "") -> dict | None:
        key = self._make_key(query, config_hash)
        result = self.cache.get(key)
        if result is not None:
            logger.debug("cache.hit", query=query[:50])
        return result

    def set(self, query: str, config_hash: str, value: dict, ttl: int = 3600) -> None:
        key = self._make_key(query, config_hash)
        self.cache.set(key, value, expire=ttl)

    def clear(self) -> None:
        self.cache.clear()

    def stats(self) -> dict:
        return {"size": len(self.cache), "volume": self.cache.volume()}
