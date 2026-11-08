"""The fault controller: asked by every proxy and tap, on every call, whether to interfere.

The plan travels to child processes (the agent, stdio MCP wrappers) in an environment
variable, and each process keeps its own per-boundary call counters. Decisions are a pure
function of (plan, boundary, call index), so the controller needs no IPC.
"""

from __future__ import annotations

import fnmatch
import json
import os
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_breakpoint.faults.base import (
    BoundaryKind,
    Fault,
    FaultContext,
    Registry,
    registry,
    rng_for,
)
from agent_breakpoint.trace import Event, TraceWriter

ENV_PLAN = "AGENT_BREAKPOINT_PLAN"
ENV_TRIAL_DIR = "AGENT_BREAKPOINT_TRIAL_DIR"
ENV_SANDBOX = "AGENT_BREAKPOINT_SANDBOX"
ENV_RESTART = "AGENT_BREAKPOINT_RESTART"  # set on a resumed process: crash faults don't refire
ENV_SESSION = "AGENT_BREAKPOINT_SESSION"
ENV_CANARIES = "AGENT_BREAKPOINT_CANARIES"  # JSON list of strings the MCP proxy watches for


@dataclass
class FaultSpec:
    boundary: str  # exact id or glob, e.g. "tool:bank.refund", "model:*"
    fault: str
    at_call: int | None = 1  # 1-based call index on that boundary; None = every call
    sticky: bool = False  # also every later call
    params: dict[str, Any] = field(default_factory=dict)
    session: int | None = None  # episodes: only in this session (1-based); None = any

    @property
    def key(self) -> str:
        at = "*" if self.at_call is None else f"{self.at_call}{'+' if self.sticky else ''}"
        sess = f"@s{self.session}" if self.session else ""
        return f"{self.boundary}|{self.fault}|{at}{sess}"

    def matches(self, boundary: str, call_index: int, session: int = 1) -> bool:
        if not fnmatch.fnmatchcase(boundary, self.boundary):
            return False
        if self.session is not None and self.session != session:
            return False
        if self.at_call is None:
            return True
        return call_index >= self.at_call if self.sticky else call_index == self.at_call

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FaultSpec:
        return cls(**data)


@dataclass
class Plan:
    seed: int = 0
    faults: list[FaultSpec] = field(default_factory=list)
    mode: str = "inject"  # record | inject
    mock_tools: bool = False
    destructive_tools: list[str] = field(default_factory=list)
    # Fork-at-fault replay: serve cached model responses while the request matches the baseline
    # and the call is before the first fault, then go live.
    replay_cache: str | None = None
    # boundary -> last call index that may be served from the cache (the faulted call's prefix)
    replay_until: dict[str, int] = field(default_factory=dict)
    packs: list[str] | str = "all"

    def to_env(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Plan:
        data = dict(data)
        data["faults"] = [FaultSpec.from_json(f) for f in data.get("faults", [])]
        return cls(**data)

    @classmethod
    def from_env(cls) -> Plan | None:
        raw = os.environ.get(ENV_PLAN)
        return cls.from_json(json.loads(raw)) if raw else None


@dataclass
class Decision:
    fault: Fault
    spec: FaultSpec
    ctx: FaultContext


class Controller:
    def __init__(self, plan: Plan, trial_dir: Path | None, component: str,
                 reg: Registry | None = None) -> None:
        self.plan = plan
        self.trial_dir = trial_dir
        self.registry = reg or registry(plan.packs)
        self.writer = TraceWriter(trial_dir, component) if trial_dir else None
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self.restarted = os.environ.get(ENV_RESTART) == "1"
        self.session = int(os.environ.get(ENV_SESSION, "1"))
        self.canaries: list[str] = json.loads(os.environ.get(ENV_CANARIES, "[]"))
        self.meta: dict[str, dict[str, Any]] = defaultdict(dict)  # per-boundary scratch

    @classmethod
    def from_env(cls, component: str) -> Controller | None:
        plan = Plan.from_env()
        if plan is None:
            return None
        trial = os.environ.get(ENV_TRIAL_DIR)
        return cls(plan, Path(trial) if trial else None, component)

    def next_call(self, boundary: str) -> int:
        with self._lock:
            self._counts[boundary] += 1
            return self._counts[boundary]

    def decide(self, boundary: str, kind: BoundaryKind, call_index: int, *,
               provider: str | None = None, op: str | None = None,
               tags: tuple[str, ...] = ()) -> list[Decision]:
        if self.plan.mode == "record":
            return []
        out: list[Decision] = []
        for spec in self.plan.faults:
            if not spec.matches(boundary, call_index, self.session):
                continue
            fault = self.registry.get(spec.fault)
            if not fault.applies(kind, provider=provider, op=op):
                continue
            if self.restarted and fault.id == "state.crash_before_checkpoint":
                continue
            params = {**{k: v for k, v in fault.default_params.items() if k not in ("sticky", "sweep")},
                      **spec.params}
            ctx = FaultContext(
                boundary=boundary, kind=kind, call_index=call_index,
                rng=rng_for(self.plan.seed, boundary, call_index), params=params,
                provider=provider, op=op, meta=self.meta[boundary],
            )
            out.append(Decision(fault, spec, ctx))
        return out

    def first_fault_call(self, boundary: str) -> int | None:
        """The earliest call index on ``boundary`` that any fault targets (for replay)."""
        calls = [s.at_call or 1 for s in self.plan.faults
                 if fnmatch.fnmatchcase(boundary, s.boundary)]
        return min(calls) if calls else None

    def record(self, event: Event) -> None:
        event.session = self.session
        if self.writer:
            self.writer.write(event)
