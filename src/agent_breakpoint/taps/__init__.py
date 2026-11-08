"""In-process taps for boundaries with no network edge.

``tap(obj)`` returns ``obj`` unchanged outside the harness. Under the harness it returns a
wrapper that records reads and writes and applies state or orchestration faults:

* LangGraph checkpointers (``BaseCheckpointSaver``) and stores (``BaseStore``)
* mutable mappings: key-value memory, a shared blackboard
* any object with ``read/write/delete/list`` methods (the adapter contract)
* callables: sub-agent calls and handoffs (orchestration faults)
* a directory path: file-based memory the harness may corrupt between sessions
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, TypeVar

from agent_breakpoint.controller import ENV_PLAN, Controller

T = TypeVar("T")

_CONTROLLER: Controller | None = None
_LOADED = False


def agent_controller() -> Controller | None:
    """The controller for this agent process, built once from the environment."""
    global _CONTROLLER, _LOADED
    if not _LOADED:
        _CONTROLLER = Controller.from_env(component="agent") if os.environ.get(ENV_PLAN) else None
        _LOADED = True
    return _CONTROLLER


def reset_controller() -> None:
    """For tests that change the environment within one process."""
    global _CONTROLLER, _LOADED
    _CONTROLLER, _LOADED = None, False


def tap(obj: T, name: str | None = None, *, shared: bool = False) -> T:
    ctl = agent_controller()
    if ctl is None:
        return obj
    return _wrap(obj, ctl, name, shared)


def _wrap(obj: Any, ctl: Controller, name: str | None, shared: bool) -> Any:
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver  # type: ignore[import-not-found]

        if isinstance(obj, BaseCheckpointSaver):
            from agent_breakpoint.taps.langgraph import TappedCheckpointer

            return TappedCheckpointer(obj, ctl, name or "checkpointer")
    except ImportError:
        pass
    try:
        from langgraph.store.base import BaseStore  # type: ignore[import-not-found]

        if isinstance(obj, BaseStore):
            from agent_breakpoint.taps.langgraph import TappedStore

            return TappedStore(obj, ctl, name or "store")
    except ImportError:
        pass
    if isinstance(obj, (str, Path)):
        from agent_breakpoint.taps.fs import register_directory

        return register_directory(obj, ctl, name or Path(obj).name)
    if isinstance(obj, MutableMapping):
        from agent_breakpoint.taps.kv import TappedMapping

        return TappedMapping(obj, ctl, name or "kv", shared=shared)
    if all(hasattr(obj, m) for m in ("read", "write", "delete", "list")):
        from agent_breakpoint.taps.kv import TappedAdapter

        return TappedAdapter(obj, ctl, name or type(obj).__name__.lower(), shared=shared)
    if callable(obj):
        from agent_breakpoint.taps.orchestration import tap_callable

        return tap_callable(obj, ctl, name or getattr(obj, "__name__", "subagent"))
    raise TypeError(f"tap() does not know how to wrap {type(obj).__name__}")


__all__ = ["agent_controller", "reset_controller", "tap"]
