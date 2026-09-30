from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Protocol


class JobStore(Protocol):
    async def get(self, analysis_id: str) -> dict[str, Any] | None: ...
    async def set(self, job: dict[str, Any]) -> None: ...


class InMemoryJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    async def get(self, analysis_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(analysis_id)
        return deepcopy(job) if job else None

    async def set(self, job: dict[str, Any]) -> None:
        self.jobs[str(job["analysisId"])] = deepcopy(job)


class RedisJobStore:
    def __init__(self, url: str, *, ttl_seconds: int = 86400) -> None:
        try:
            from redis.asyncio import from_url
        except ImportError as exc:
            raise RuntimeError("Redis job storage requires the redis package") from exc
        self.redis = from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    async def get(self, analysis_id: str) -> dict[str, Any] | None:
        value = await self.redis.get(f"palda:analysis:{analysis_id}")
        return json.loads(value) if value else None

    async def set(self, job: dict[str, Any]) -> None:
        await self.redis.setex(
            f"palda:analysis:{job['analysisId']}",
            self.ttl_seconds,
            json.dumps(job, ensure_ascii=False),
        )

    async def close(self) -> None:
        await self.redis.aclose()


def store_from_env() -> JobStore:
    url = os.environ.get("PALDA_REDIS_URL", "").strip()
    if not url:
        return InMemoryJobStore()
    return RedisJobStore(url, ttl_seconds=int(os.environ.get("PALDA_JOB_TTL_SECONDS", "86400")))
