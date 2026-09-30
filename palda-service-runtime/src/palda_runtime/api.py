from __future__ import annotations

import asyncio
import copy
import inspect
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Response

from .runtime import analyzer_from_env
from .schemas import AnalysisRequest, JobCreated, JobStatus
from .store import InMemoryJobStore, JobStore, store_from_env


class Analyzer(Protocol):
    async def analyze(
        self, request: dict[str, Any], on_stage: Callable[[str, int], None] | None = None
    ) -> dict[str, Any]: ...


class MissingAnalyzer:
    @property
    def ready(self) -> bool:
        return False

    async def analyze(
        self, request: dict[str, Any], on_stage: Callable[[str, int], None] | None = None
    ) -> dict[str, Any]:
        raise RuntimeError("Model analyzer is not configured")


def create_app(analyzer: Analyzer | None = None, store: JobStore | None = None) -> FastAPI:
    resolved_analyzer = analyzer or MissingAnalyzer()
    resolved_store = store or InMemoryJobStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(resolved_analyzer, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result
        close_store = getattr(resolved_store, "close", None)
        if close_store:
            result = close_store()
            if inspect.isawaitable(result):
                await result

    app = FastAPI(title="PALDA Analysis API", version="0.1.0", lifespan=lifespan)
    app.state.analyzer = resolved_analyzer
    app.state.store = resolved_store
    app.state.tasks: set[asyncio.Task[Any]] = set()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        readiness = getattr(app.state.analyzer, "ready", True)
        if callable(readiness):
            readiness = readiness()
        if not readiness:
            raise HTTPException(status_code=503, detail="model worker is not ready")
        return {"status": "ready"}

    @app.post("/api/product-analyses", response_model=JobCreated, status_code=202)
    async def create_analysis(request: AnalysisRequest, response: Response) -> JobCreated:
        response.headers["Retry-After"] = "2"
        response.headers["Cache-Control"] = "no-store"
        analysis_id = f"analysis_{uuid.uuid4().hex}"
        payload = request.model_dump()
        payload["analysisId"] = analysis_id
        payload["taxonomyVersion"] = os.environ.get("PALDA_TAXONOMY_VERSION", "theme-v1.0.0")
        job = {
            "analysisId": analysis_id, "status": "queued", "progress": 0,
            "stages": {
                "classification": "pending", "aggregation": "pending",
                "reportGeneration": "pending", "validation": "pending",
            },
        }
        await app.state.store.set(job)
        task = asyncio.create_task(_run_job(app, analysis_id, payload))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return JobCreated(analysisId=analysis_id, status="queued", progress=0)

    @app.get("/api/product-analyses/{analysis_id}", response_model=JobStatus)
    async def get_analysis(analysis_id: str, response: Response) -> JobStatus:
        job = await app.state.store.get(analysis_id)
        if not job:
            raise HTTPException(status_code=404, detail="analysis not found")
        response.headers["Cache-Control"] = "no-store"
        if job["status"] not in {"completed", "failed"}:
            response.headers["Retry-After"] = "2"
        return JobStatus.model_validate(job)

    return app


async def _run_job(app: FastAPI, analysis_id: str, payload: dict[str, Any]) -> None:
    job = await app.state.store.get(analysis_id)
    if job is None:
        return
    stage_writes: list[asyncio.Task[Any]] = []
    try:
        def on_stage(stage: str, progress: int) -> None:
            stage_keys = {
                "classifying": "classification", "aggregating": "aggregation",
                "generating": "reportGeneration", "validating": "validation",
            }
            current = stage_keys[stage]
            order = list(job["stages"])
            for key in order:
                if order.index(key) < order.index(current):
                    job["stages"][key] = "completed"
            job["stages"][current] = "running"
            job.update(status=stage, progress=progress)
            task = asyncio.create_task(app.state.store.set(copy.deepcopy(job)))
            stage_writes.append(task)
            app.state.tasks.add(task)
            task.add_done_callback(app.state.tasks.discard)

        result = await app.state.analyzer.analyze(payload, on_stage=on_stage)
        for stage in job["stages"]:
            job["stages"][stage] = "completed"
        job.update(status="completed", progress=100, result=result)
    except Exception as exc:
        job.update(status="failed", progress=100, error=f"{type(exc).__name__}: {exc}"[:1000])
    if stage_writes:
        await asyncio.gather(*stage_writes)
    await app.state.store.set(job)


def app_from_env() -> FastAPI:
    if os.environ.get("PALDA_MODEL_SUBPROCESS", "1") == "1":
        from .worker import ModelProcessAnalyzer

        return create_app(ModelProcessAnalyzer(), store_from_env())
    return create_app(analyzer_from_env(), store_from_env())


app = create_app()


def main() -> None:
    configured_app = app_from_env()
    uvicorn.run(
        configured_app,
        host=os.environ.get("PALDA_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("PALDA_API_PORT", "8080")),
    )
