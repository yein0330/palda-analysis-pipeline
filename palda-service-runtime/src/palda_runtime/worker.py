from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import traceback
import uuid
from typing import Any, Callable


class ModelProcessAnalyzer:
    """Own CUDA models in one spawned process while the API event loop stays responsive."""

    def __init__(self, *, startup_timeout: int = 600) -> None:
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_model_worker, args=(self._requests, self._responses),
            name="palda-model-worker", daemon=True,
        )
        self._process.start()
        try:
            ready = self._responses.get(timeout=startup_timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("PALDA model worker startup timed out") from exc
        if ready.get("kind") != "ready":
            self.close()
            raise RuntimeError(str(ready.get("error") or "PALDA model worker failed to start"))
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._process.is_alive()

    async def analyze(
        self,
        request: dict[str, Any],
        on_stage: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            request_id = uuid.uuid4().hex
            self._requests.put({"kind": "analyze", "id": request_id, "request": request})
            while True:
                message = await asyncio.to_thread(self._responses.get)
                if message.get("id") != request_id:
                    raise RuntimeError("Model worker response correlation failed")
                if message["kind"] == "stage":
                    if on_stage:
                        on_stage(str(message["stage"]), int(message["progress"]))
                elif message["kind"] == "result":
                    return dict(message["result"])
                elif message["kind"] == "error":
                    raise RuntimeError(str(message["error"]))

    def close(self) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.is_alive():
            self._requests.put({"kind": "stop"})
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)


def _model_worker(requests: Any, responses: Any) -> None:
    try:
        from .runtime import analyzer_from_env

        analyzer = analyzer_from_env()
        responses.put({"kind": "ready"})
    except BaseException as exc:
        responses.put({
            "kind": "startup_error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        })
        return
    while True:
        message = requests.get()
        if message.get("kind") == "stop":
            return
        request_id = str(message["id"])

        def on_stage(stage: str, progress: int) -> None:
            responses.put({
                "kind": "stage", "id": request_id,
                "stage": stage, "progress": progress,
            })

        try:
            result = asyncio.run(analyzer.analyze(message["request"], on_stage=on_stage))
            responses.put({"kind": "result", "id": request_id, "result": result})
        except BaseException as exc:
            responses.put({
                "kind": "error", "id": request_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
