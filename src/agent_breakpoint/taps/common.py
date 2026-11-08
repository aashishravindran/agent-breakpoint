"""Shared plumbing for taps: record an operation and collect the faults that apply to it."""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Any

from agent_breakpoint.controller import Controller
from agent_breakpoint.faults.base import Action, BoundaryKind, Kill
from agent_breakpoint.trace import Event


class OpScope:
    """``with OpScope(...) as scope:`` around one tap operation.

    ``scope.actions`` holds the request-side actions to apply; the event is written on exit.
    A ``Kill`` action kills the process immediately (crash faults), after flushing the trace.
    """

    def __init__(self, ctl: Controller, name: str, op: str, kind: BoundaryKind = BoundaryKind.STATE,
                 tags: tuple[str, ...] = (), detail: dict[str, Any] | None = None,
                 request: Any = None) -> None:
        self.ctl = ctl
        self.boundary = f"{'state' if kind == BoundaryKind.STATE else 'orchestration'}:{name}.{op}"
        self.kind = kind
        self.op = op
        self.idx = ctl.next_call(self.boundary)
        self.event = Event(boundary=self.boundary, kind=kind.value, call_index=self.idx, op=op,
                           tags=list(tags), detail=dict(detail or {}))
        self.decisions = ctl.decide(self.boundary, kind, self.idx, op=op, tags=tags)
        self.event.faults = [d.fault.id for d in self.decisions]
        self.actions: list[Action] = []
        self._started = time.time()
        for d in self.decisions:
            if d.fault.before is None:
                continue
            out = d.fault.before(d.ctx, request if request is not None else {"op": op})
            if isinstance(out, Kill):
                self.kill(d.ctx.meta.pop("signal", signal.SIGKILL))
            if isinstance(out, Action):
                self.actions.append(out)

    def after(self, request: Any, result: Any) -> Any:
        for d in self.decisions:
            if d.fault.after is not None:
                result = d.fault.after(d.ctx, request, result)
        return result

    def first(self, cls: type) -> Any:
        return next((a for a in self.actions if isinstance(a, cls)), None)

    def canary(self) -> str | None:
        for d in self.decisions:
            if d.ctx.meta.get("canary"):
                return d.ctx.meta.pop("canary")
        return None

    def kill(self, sig: int) -> None:
        self.event.status = "killed"
        self.ctl.record(self.event)
        sys.stdout.flush()
        sys.stderr.flush()
        os.kill(os.getpid(), sig)
        time.sleep(5)  # SIGINT may be handled; give the handler a moment

    def __enter__(self) -> OpScope:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and self.event.status == "ok":
            self.event.status = "error"
            self.event.detail["exception"] = repr(exc)
        c = self.canary()
        if c:
            self.event.canary = c
        self.event.latency_ms = (time.time() - self._started) * 1000
        self.ctl.record(self.event)
