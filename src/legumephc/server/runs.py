"""In-process registry of exact Studio worker child processes.

The server never solves in-process: every run is the same
``python -m legumephc.studio.worker`` child that the public core API drives.
This module only relays its JSON-line events to HTTP clients.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any
import uuid

from ..studio.project import record_reference
from ..studio.worker import WorkerProcess, start_worker


POLL_SECONDS = 0.15


@dataclass
class Run:
    id: str
    request: dict[str, Any]
    project_path: Path
    handle: WorkerProcess
    started: float = field(default_factory=time.monotonic)
    events: list[dict[str, Any]] = field(default_factory=list)
    state: str = "running"
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def add(self, event: dict[str, Any]) -> None:
        self.events.append({**event, "elapsed_s": round(time.monotonic() - self.started, 3)})
        self.changed.set()


class RunRegistry:
    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.runs: dict[str, Run] = {}

    def active(self) -> Run | None:
        return next((run for run in self.runs.values() if run.state == "running"), None)

    def start(self, request: dict[str, Any], project_path: Path) -> Run:
        if self.active() is not None:
            raise RuntimeError("a calculation is already running")
        handle = start_worker(request, cwd=self.cwd)
        run = Run(uuid.uuid4().hex[:12], deepcopy(request), project_path, handle)
        self.runs[run.id] = run
        asyncio.get_running_loop().create_task(self._pump(run))
        return run

    def cancel(self, run_id: str) -> Run:
        run = self.runs[run_id]
        if run.state == "running":
            run.state = "cancelled"
            run.handle.cancel()
            run.add({"event": "cancelled", "message": "Cancelled; exact worker terminated"})
        return run

    async def _pump(self, run: Run) -> None:
        while True:
            for event in run.handle.read_events():
                if event.get("event") == "completed":
                    event = {**event, "result_entry": self._result_entry(run, event)}
                run.add(event)
            if run.handle.poll() is not None:
                await asyncio.sleep(POLL_SECONDS)
                for event in run.handle.read_events():
                    if event.get("event") == "completed":
                        event = {**event, "result_entry": self._result_entry(run, event)}
                    run.add(event)
                break
            await asyncio.sleep(POLL_SECONDS)
        if run.state == "cancelled":
            return
        _stdout, stderr = await asyncio.to_thread(run.handle.communicate)
        ok = any(event.get("event") == "completed" for event in run.events)
        run.state = "succeeded" if ok else "failed"
        if not ok and not any(event.get("event") == "failed" for event in run.events):
            run.add({"event": "failed", "error": (stderr or "worker exited without a result").strip()[-2000:]})
        run.add({"event": "finished", "state": run.state})

    @staticmethod
    def _result_entry(run: Run, event: dict[str, Any]) -> dict[str, Any] | None:
        """Build the project result entry the Qt Studio used to append."""

        try:
            reference = record_reference(Path(event["record_path"]), run.project_path.parent)
        except (KeyError, ValueError) as exc:
            return {"error": str(exc)}
        request = run.request
        return {
            "id": None,  # assigned by the client against its current project
            "calculation_id": request.get("calculation_id", "calc-1"),
            "record_reference": reference,
            "model_snapshot": request.get("model_snapshot", request.get("case")),
            "calculation_snapshot": request.get("calculation_snapshot", request.get("calculation")),
            "display_state": {"pins": []},
        }
