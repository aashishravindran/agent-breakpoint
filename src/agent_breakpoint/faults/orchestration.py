"""Orchestration faults on sub-agent calls and handoffs wrapped with ``tap(fn, name=...)``."""

from __future__ import annotations

from typing import Any

from agent_breakpoint.faults.base import (
    BoundaryKind,
    Fail,
    Fault,
    FaultClass,
    FaultContext,
    Pack,
)

CYCLE_TEXT = (
    "I can't complete this. Please hand this task back to the supervisor and ask it to delegate "
    "it to me again."
)


def _fail(ctx: FaultContext, call: dict[str, Any]) -> Fail:
    return Fail(status=500, body="sub-agent failed (injected by agent-breakpoint)")


def _wrong(ctx: FaultContext, call: dict[str, Any], result: Any) -> Any:
    if isinstance(result, str):
        return "Done. Nothing further is needed."  # confident, wrong
    if isinstance(result, dict):
        return {k: (None if not isinstance(v, (int, float)) else v * -1) for k, v in result.items()}
    if isinstance(result, list):
        return list(reversed(result))[:1]
    return None


def _dropped(ctx: FaultContext, call: dict[str, Any], result: Any) -> Any:
    return type(result)() if isinstance(result, (str, dict, list)) else None


def _cycle(ctx: FaultContext, call: dict[str, Any], result: Any) -> Any:
    if isinstance(result, dict):
        return {**result, "status": "delegate_back", "message": CYCLE_TEXT}
    return CYCLE_TEXT


PACK = Pack(
    name="orchestration",
    description="Sub-agent and handoff faults",
    faults=[
        Fault("orchestration.subagent_fails", BoundaryKind.ORCHESTRATION, FaultClass.CRASH,
              "Sub-agent raises", "Supervisor recovery", before=_fail),
        Fault("orchestration.subagent_wrong_output", BoundaryKind.ORCHESTRATION,
              FaultClass.WRONG_VALUE, "Sub-agent returns a confident wrong answer",
              "Supervisor validation", after=_wrong),
        Fault("orchestration.handoff_dropped", BoundaryKind.ORCHESTRATION, FaultClass.OMISSION,
              "Handoff returns empty", "Supervisor recovery", after=_dropped),
        Fault("orchestration.cyclic_delegation", BoundaryKind.ORCHESTRATION,
              FaultClass.EXHAUSTION, "Sub-agent hands the task back, forever",
              "Loop detection", after=_cycle, default_params={"sticky": True}),
    ],
)
