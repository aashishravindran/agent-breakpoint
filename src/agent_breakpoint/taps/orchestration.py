"""Orchestration taps: ``tap(subagent_fn, name="researcher")`` wraps a sub-agent call or a handoff."""

from __future__ import annotations

import functools
import inspect
from typing import Any

from agent_breakpoint.controller import Controller
from agent_breakpoint.faults.base import BoundaryKind, Fail
from agent_breakpoint.taps.common import OpScope


class SubagentFailure(RuntimeError):
    pass


def tap_callable(fn: Any, ctl: Controller, name: str) -> Any:
    def scope(args: tuple, kwargs: dict) -> OpScope:
        return OpScope(ctl, name, "call", kind=BoundaryKind.ORCHESTRATION,
                       request={"args": args, "kwargs": kwargs})

    def check(s: OpScope) -> None:
        fail = s.first(Fail)
        if fail is not None:
            s.event.status = "error"
            raise SubagentFailure(str(fail.body))

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            with scope(args, kwargs) as s:
                check(s)
                result = await fn(*args, **kwargs)
                return s.after({"args": args, "kwargs": kwargs}, result)
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with scope(args, kwargs) as s:
            check(s)
            result = fn(*args, **kwargs)
            return s.after({"args": args, "kwargs": kwargs}, result)
    return wrapper
