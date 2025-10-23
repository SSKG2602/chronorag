from __future__ import annotations

import json
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import redis
except Exception:  # pragma: no cover
    redis = None


class CacheClient:
    def __init__(self, url: str | None):
        self.fallback: dict[str, Any] = {}
        self.client = None
        if url and redis is not None:
            try:
                self.client = redis.from_url(url)
            except Exception:
                self.client = None

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        payload = json.dumps(value)
        if self.client:
            try:
                self.client.set(key, payload, ex=ex)
                return
            except Exception:  # pragma: no cover
                pass
        self.fallback[key] = payload

    def get(self, key: str) -> Any | None:
        if self.client:
            try:
                value = self.client.get(key)
                if value is not None:
                    return json.loads(value)
            except Exception:  # pragma: no cover
                pass
        payload = self.fallback.get(key)
        return json.loads(payload) if payload else None

    def clear(self) -> None:
        if self.client:
            try:
                self.client.flushdb()
            except Exception:  # pragma: no cover
                pass
        self.fallback.clear()
