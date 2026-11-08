"""State oracle: the sandbox's end state checked against the scenario's invariants.

An invariant is ``module:function(sandbox, outcome)`` returning a bool, an ``InvariantResult``,
or a ``(bool, message)`` tuple. All pass = ok; some pass = partial; none = failed.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_breakpoint.contract import Outcome, Sandbox


@dataclass
class InvariantResult:
    ok: bool
    message: str = ""
    name: str = ""


@dataclass
class StateVerdict:
    status: Literal["ok", "partial", "failed", "unknown"]
    results: list[InvariantResult] = field(default_factory=list)

    @property
    def failures(self) -> list[InvariantResult]:
        return [r for r in self.results if not r.ok]


def load_callable(ref: str, python_path: list[str] | None = None) -> Any:
    for p in python_path or []:
        if p not in sys.path:
            sys.path.insert(0, p)
    module, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:function', got {ref!r}")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _coerce(value: Any, name: str) -> InvariantResult:
    if isinstance(value, InvariantResult):
        value.name = value.name or name
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return InvariantResult(bool(value[0]), str(value[1]), name)
    return InvariantResult(bool(value), "", name)


def check_invariants(refs: list[str], sandbox: Sandbox, outcome: Outcome,
                     python_path: list[str] | None = None) -> StateVerdict:
    if not refs:
        return StateVerdict("unknown")
    results = []
    for ref in refs:
        name = ref.rpartition(":")[2]
        try:
            results.append(_coerce(load_callable(ref, python_path)(sandbox, outcome), name))
        except Exception as exc:  # an invariant that can't read the state counts as failed
            results.append(InvariantResult(False, f"invariant raised {exc!r}", name))
    passed = sum(r.ok for r in results)
    status = "ok" if passed == len(results) else "failed" if passed == 0 else "partial"
    return StateVerdict(status, results)
