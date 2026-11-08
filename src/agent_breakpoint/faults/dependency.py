"""Tool-dependency faults: the policy engine, JWKS endpoint, database or upstream API a tool
server relies on. Applied by the dependency proxy (HTTP or raw TCP)."""

from __future__ import annotations

from typing import Any

from agent_breakpoint.faults.base import (
    BoundaryKind,
    Delay,
    Drop,
    Fail,
    Fault,
    FaultClass,
    FaultContext,
    Hang,
    Pack,
)
from agent_breakpoint.faults.http import HttpResponse


def _latency(ctx: FaultContext, req: Any) -> Delay:
    return Delay(seconds=float(ctx.params.get("seconds", 5.0)))


def _timeout(ctx: FaultContext, req: Any) -> Hang:
    return Hang(seconds=float(ctx.params.get("seconds", 3600)))


def _server_error(ctx: FaultContext, req: Any) -> Fail:
    return Fail(status=int(ctx.params.get("status", 503)),
                body={"error": "service unavailable (injected by agent-breakpoint)"},
                headers={"content-type": "application/json"})


def _reset(ctx: FaultContext, req: Any) -> Drop:
    return Drop()


def _empty(ctx: FaultContext, req: Any, resp: HttpResponse) -> HttpResponse:
    """A 200 with an empty body: an authorizer returning nothing, a JWKS with no keys."""
    data = resp.json()
    if isinstance(data, dict):
        return resp.with_json({k: ([] if isinstance(v, list) else {} if isinstance(v, dict) else None)
                               for k, v in data.items()})
    if isinstance(data, list):
        return resp.with_json([])
    return resp.with_body(b"")


def _stale(ctx: FaultContext, req: Any, resp: HttpResponse) -> HttpResponse | None:
    """Serve the response this endpoint gave the first time it was called."""
    first = ctx.meta.get("first_response")
    return first if isinstance(first, HttpResponse) else None


PACK = Pack(
    name="dependency",
    description="Faults behind the tool server; fail-closed behaviour",
    faults=[
        Fault("dependency.latency", BoundaryKind.DEPENDENCY, FaultClass.LATENCY,
              "5 s added to the dependency call", "Timeouts", before=_latency),
        Fault("dependency.timeout", BoundaryKind.DEPENDENCY, FaultClass.OMISSION,
              "Dependency never answers", "Fail-closed", before=_timeout),
        Fault("dependency.server_error", BoundaryKind.DEPENDENCY, FaultClass.CRASH,
              "Dependency returns 503", "Fail-closed", before=_server_error, ops=("http",)),
        Fault("dependency.reset", BoundaryKind.DEPENDENCY, FaultClass.CRASH,
              "Connection reset", "Fail-closed", before=_reset),
        Fault("dependency.empty", BoundaryKind.DEPENDENCY, FaultClass.WRONG_VALUE,
              "200 with an empty payload", "Fail-closed", after=_empty, ops=("http",)),
        Fault("dependency.stale", BoundaryKind.DEPENDENCY, FaultClass.WRONG_VALUE,
              "First-ever response replayed", "Cache and revocation handling", after=_stale,
              ops=("http",), requires=("repeat",)),
    ],
)
