"""The integration contract: ``run(task, sandbox) -> Outcome``.

These types are a stable, semver-covered surface. Agent code receives a ``Task`` and a
``Sandbox`` and returns an ``Outcome``; nothing else about the agent changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Claim = Literal["success", "partial", "failure"]


@dataclass
class Task:
    """One unit of work for the agent under test."""

    prompt: str
    thread_id: str = "agent-breakpoint"
    session: int = 1
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Task:
        return cls(**data)


@dataclass
class Sandbox:
    """A fresh, harness-owned directory per trial.

    Agents keep durable state (databases, checkpoints, memory files) under ``root`` so that
    every trial starts from the same seeded state and the oracles can inspect the end state.
    """

    root: Path
    env: dict[str, str] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def checkpointer(self, name: str = "checkpoints.sqlite"):
        """A LangGraph SQLite checkpointer stored inside the sandbox."""
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]

        conn = sqlite3.connect(self.path(name), check_same_thread=False)
        return SqliteSaver(conn)

    def read_json(self, *parts: str, default: Any = None) -> Any:
        p = self.root.joinpath(*parts)
        if not p.exists():
            return default
        return json.loads(p.read_text())

    def write_json(self, data: Any, *parts: str) -> None:
        self.path(*parts).write_text(json.dumps(data, indent=2, sort_keys=True))

    def to_json(self) -> dict[str, Any]:
        return {"root": str(self.root), "env": self.env}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Sandbox:
        return cls(root=Path(data["root"]), env=dict(data.get("env", {})))

    @classmethod
    def current(cls) -> Sandbox | None:
        """The sandbox of the trial this process belongs to, if running under the harness."""
        root = os.environ.get("AGENT_BREAKPOINT_SANDBOX")
        return cls(root=Path(root)) if root else None


@dataclass
class Outcome:
    """What the agent says happened.

    ``claim`` is the agent's own structured success claim; the claim oracle compares it with
    the sandbox's actual end state. Free-text agents can leave it ``None`` and the harness
    infers a claim from ``answer`` (see ``oracles.claim.infer_claim``).
    """

    answer: str = ""
    claim: Claim | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Outcome:
        return cls(
            answer=data.get("answer", ""),
            claim=data.get("claim"),
            details=data.get("details", {}),
            error=data.get("error"),
        )
