"""Trace events. Every proxy and tap appends JSON lines to its own file in the trial directory;
the harness merges them by timestamp."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def stable_hash(obj: Any) -> str:
    if isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
    else:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class Event:
    boundary: str
    kind: str
    call_index: int
    component: str = ""
    session: int = 1
    ts: float = field(default_factory=time.time)
    op: str | None = None
    provider: str | None = None
    faults: list[str] = field(default_factory=list)
    status: str = "ok"  # ok | error | hang | drop | killed | replayed
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tags: list[str] = field(default_factory=list)
    req_hash: str | None = None
    destructive: bool = False
    tool: str | None = None
    canary: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}, "")
                or k in ("boundary", "kind", "call_index", "status", "session")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Event:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class TraceWriter:
    def __init__(self, directory: Path, component: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"trace-{component}-{os.getpid()}.jsonl"
        self.component = component
        self._lock = threading.Lock()

    def write(self, event: Event) -> None:
        event.component = event.component or self.component
        line = json.dumps(event.to_json(), default=str)
        with self._lock, self.path.open("a") as fh:
            fh.write(line + "\n")


def read_trace(directory: Path) -> list[Event]:
    events: list[Event] = []
    for p in sorted(directory.glob("trace-*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    events.append(Event.from_json(json.loads(line)))
                except ValueError:
                    continue  # a line torn by a killed process
    events.sort(key=lambda e: e.ts)
    return events
