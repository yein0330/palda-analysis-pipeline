import asyncio

import httpx

from palda_runtime.api import create_app
from palda_runtime.store import InMemoryJobStore


class FakeAnalyzer:
    async def analyze(self, request: dict, on_stage=None) -> dict:
        if on_stage:
            on_stage("classifying", 10)
            on_stage("aggregating", 65)
            on_stage("generating", 75)
            on_stage("validating", 95)
        return {"analysisId": request["analysisId"], "sellingPoints": []}


def test_async_job_contract() -> None:
    asyncio.run(_exercise_api())


async def _exercise_api() -> None:
    transport = httpx.ASGITransport(app=create_app(FakeAnalyzer()))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).json() == {"status": "ready"}
        response = await client.post("/api/product-analyses", json={
            "product": {"productId": "p1", "categoryPath": "가전 > 이어폰"},
            "reviewSample": {"items": [{"id": "r1", "text": "연결이 편해요"}]},
            "qnaSample": {"items": [{"id": "q1", "text": "언제 배송되나요?"}]},
        })
        assert response.status_code == 202
        assert response.headers["retry-after"] == "2"
        assert response.headers["cache-control"] == "no-store"
        analysis_id = response.json()["analysisId"]
        for _ in range(20):
            job = (await client.get(f"/api/product-analyses/{analysis_id}")).json()
            if job["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        assert job["status"] == "completed"
        assert set(job["stages"].values()) == {"completed"}
        assert job["result"]["analysisId"] == analysis_id


def test_missing_model_is_not_ready() -> None:
    async def exercise() -> None:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")
            assert response.status_code == 503

    asyncio.run(exercise())


def test_completed_job_can_be_read_by_another_api_instance() -> None:
    async def exercise() -> None:
        store = InMemoryJobStore()
        first = httpx.ASGITransport(app=create_app(FakeAnalyzer(), store))
        async with httpx.AsyncClient(transport=first, base_url="http://test") as client:
            created = (await client.post("/api/product-analyses", json={
                "product": {"productId": "p1", "categoryPath": "가전 > 이어폰"},
                "reviewSample": {"items": []},
                "qnaSample": {"items": []},
            })).json()
            for _ in range(20):
                status = (await client.get(f"/api/product-analyses/{created['analysisId']}")).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
        second = httpx.ASGITransport(app=create_app(FakeAnalyzer(), store))
        async with httpx.AsyncClient(transport=second, base_url="http://test") as client:
            restored = (await client.get(f"/api/product-analyses/{created['analysisId']}")).json()
        assert restored["status"] == "completed"

    asyncio.run(exercise())
