"""Fault definitions, actions and the pack registry.

A fault is a deterministic function of (seed, boundary, call index): the same plan always
interferes with the same calls in the same way, so every injected run replays exactly.

Six fault classes from distributed systems apply at each boundary kind. Packs are plugins
(entry point group ``agent_breakpoint.packs``) so the catalog grows without touching the core.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import entry_points
from typing import Any


class FaultClass(str, Enum):
    CRASH = "crash"
    OMISSION = "omission"
    LATENCY = "latency"
    WRONG_VALUE = "wrong_value"
    ADVERSARIAL = "adversarial"
    EXHAUSTION = "exhaustion"


class BoundaryKind(str, Enum):
    MODEL = "model"
    CONTEXT = "context"  # same proxy as MODEL; faults rewrite the request instead of the response
    TOOL = "tool"
    DEPENDENCY = "dependency"
    STATE = "state"
    ORCHESTRATION = "orchestration"
    BUDGET = "budget"


def rng_for(seed: int, boundary: str, call_index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{boundary}|{call_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# --- Actions a fault can take at a boundary -------------------------------------------------


@dataclass
class Action:
    """Base class. Proxies and taps interpret the subclasses below."""


@dataclass
class Delay(Action):
    seconds: float


@dataclass
class Hang(Action):
    """Never answer; the caller's own timeout decides what happens."""

    seconds: float = 3600.0


@dataclass
class Fail(Action):
    status: int = 500
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Reply(Action):
    """Answer in place of the real server, which never sees the call (tools: a JSON-RPC result)."""

    result: Any = None


@dataclass
class Drop(Action):
    """Close the connection without a response."""


@dataclass
class Kill(Action):
    """Kill the agent process (crash fault). The harness restarts it for resume tests."""


# --- Context passed to fault hooks ----------------------------------------------------------


@dataclass
class FaultContext:
    boundary: str  # e.g. "model:llm", "tool:bank.refund", "state:checkpointer"
    kind: BoundaryKind
    call_index: int
    rng: random.Random
    params: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None  # model boundaries
    op: str | None = None  # state boundaries: read/write/list/delete
    meta: dict[str, Any] = field(default_factory=dict)


Hook = Callable[..., Any]


@dataclass
class Fault:
    """A named fault.

    ``before(ctx, request)`` may return an ``Action`` to short-circuit the call, or a rewritten
    request (dict) to forward instead, or ``None`` to forward unchanged.

    ``after(ctx, request, response)`` may return an ``Action`` or a rewritten response, or
    ``None`` to keep the response.

    What "request" and "response" are depends on the boundary kind: parsed JSON bodies for model
    and HTTP dependency boundaries, JSON-RPC messages for tools, tap operations for state.
    """

    id: str
    kind: BoundaryKind
    fault_class: FaultClass
    description: str
    tests: str = ""
    before: Hook | None = None
    after: Hook | None = None
    # Restrict to providers (model) or ops (state); empty = all.
    providers: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    # Only meaningful on boundaries the baseline marked as destructive / streaming etc.
    requires: tuple[str, ...] = ()
    default_params: dict[str, Any] = field(default_factory=dict)

    def applies(self, boundary_kind: BoundaryKind, provider: str | None = None,
                op: str | None = None, tags: Iterable[str] | None = None) -> bool:
        """``tags=None`` skips the ``requires`` check: the planner already chose a call that
        satisfies it, and some tags (a tool call in the response) are only known afterwards."""
        model_side = (BoundaryKind.CONTEXT, BoundaryKind.BUDGET)
        if boundary_kind != self.kind and not (
            self.kind in model_side and boundary_kind == BoundaryKind.MODEL
        ):
            return False
        if self.providers and provider and provider not in self.providers:
            return False
        if self.ops and op and op not in self.ops:
            return False
        if tags is None:
            return True
        tagset = set(tags)
        return all(r in tagset for r in self.requires)


@dataclass
class Pack:
    name: str
    faults: list[Fault]
    description: str = ""


class Registry:
    def __init__(self) -> None:
        self.packs: dict[str, Pack] = {}
        self.faults: dict[str, Fault] = {}

    def add(self, pack: Pack) -> None:
        self.packs[pack.name] = pack
        for f in pack.faults:
            if f.id in self.faults and self.faults[f.id] is not f:
                raise ValueError(f"duplicate fault id {f.id!r} in pack {pack.name!r}")
            self.faults[f.id] = f

    def get(self, fault_id: str) -> Fault:
        try:
            return self.faults[fault_id]
        except KeyError:
            raise KeyError(f"unknown fault {fault_id!r}") from None

    def for_kind(self, kind: BoundaryKind) -> list[Fault]:
        return [f for f in self.faults.values() if f.kind == kind]


_REGISTRY: Registry | None = None


def registry(enabled: list[str] | str = "all") -> Registry:
    """Load packs from entry points. Cached for the process when ``enabled == "all"``."""
    global _REGISTRY
    if enabled == "all" and _REGISTRY is not None:
        return _REGISTRY
    reg = Registry()
    for ep in entry_points(group="agent_breakpoint.packs"):
        if enabled != "all" and ep.name not in enabled:
            continue
        pack = ep.load()
        reg.add(pack)
    if not reg.packs:  # running from a source checkout without installed metadata
        from agent_breakpoint.faults import builtin_packs

        for pack in builtin_packs():
            if enabled == "all" or pack.name in enabled:
                reg.add(pack)
    if enabled == "all":
        _REGISTRY = reg
    return reg
