import asyncio

import httpx
import pytest

from palda_runtime.http_benchmark import measure


def test_rejected_submission_is_not_reported_as_completed():
    async def exercise():
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        )) as client:
            with pytest.raises(RuntimeError, match="HTTP 503"):
                await measure(client, {})
    asyncio.run(exercise())


def test_failed_job_is_not_reported_as_latency_success():
    def respond(request):
        if request.method == "POST":
            return httpx.Response(202, json={"analysisId": "test"})
        return httpx.Response(200, json={"status": "failed"})

    async def exercise():
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(RuntimeError, match="failed"):
                await measure(client, {})
    asyncio.run(exercise())


def test_pending_job_has_a_bounded_timeout():
    def respond(request):
        if request.method == "POST":
            return httpx.Response(202, json={"analysisId": "test"})
        return httpx.Response(200, json={"status": "queued"}, headers={"Retry-After": "2"})

    async def exercise():
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(TimeoutError):
                await measure(client, {}, timeout=0.01)
    asyncio.run(exercise())
