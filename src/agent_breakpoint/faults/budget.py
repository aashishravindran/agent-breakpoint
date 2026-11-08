"""Budget faults: a token, cost or wall-clock cap reached mid-task. Graceful stop vs runaway."""

from __future__ import annotations

import signal

from agent_breakpoint.faults.base import BoundaryKind, Fault, FaultClass, FaultContext, Kill, Pack
from agent_breakpoint.faults.http import HttpRequest
from agent_breakpoint.faults.model import error_response


def _quota(ctx: FaultContext, req: HttpRequest):
    fail = error_response(ctx.provider, 429, "insufficient_quota",
                          "You exceeded your current quota (injected by agent-breakpoint)")
    fail.headers.pop("retry-after", None)
    return fail


def _interrupt(ctx: FaultContext, req: HttpRequest) -> Kill:
    # SIGINT, not SIGKILL: the agent gets a chance to stop gracefully and report.
    ctx.meta["signal"] = signal.SIGINT
    return Kill()


PACK = Pack(
    name="budget",
    description="Caps reached mid-task",
    faults=[
        Fault("budget.quota_exhausted", BoundaryKind.BUDGET, FaultClass.EXHAUSTION,
              "Every model call from here on fails with insufficient_quota",
              "Graceful stop vs runaway retries", before=_quota,
              default_params={"sticky": True}),
        Fault("budget.wall_clock", BoundaryKind.BUDGET, FaultClass.EXHAUSTION,
              "Agent receives SIGINT mid-task (deadline reached)", "Graceful stop",
              before=_interrupt),
    ],
)
