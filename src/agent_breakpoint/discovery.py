"""Discovery: one traced baseline run becomes the failure-point graph, every model, tool,
dependency and state boundary the agent actually touched, with call counts. The fault space is
that graph × the faults each boundary kind supports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from agent_breakpoint.faults.base import BoundaryKind
from agent_breakpoint.trace import Event, stable_hash


@dataclass
class CallInfo:
    index: int
    session: int
    ts: float
    tags: list[str]
    req_hash: str | None


@dataclass
class FailurePoint:
    boundary: str
    kind: str
    calls: list[CallInfo] = field(default_factory=list)
    provider: str | None = None
    op: str | None = None
    destructive: bool = False
    schema_hash: str | None = None

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def tags(self) -> set[str]:
        return {t for c in self.calls for t in c.tags}

    def fingerprint(self, upto: int | None = None) -> str:
        calls = self.calls if upto is None else [c for c in self.calls if c.index <= upto]
        return stable_hash({"b": self.boundary, "schema": self.schema_hash,
                            "reqs": [c.req_hash for c in calls]})


@dataclass
class FailurePointGraph:
    points: dict[str, FailurePoint] = field(default_factory=dict)
    edges: dict[str, dict[str, int]] = field(default_factory=dict)  # boundary -> next boundary -> n
    sessions: int = 1
    events: list[Event] = field(default_factory=list)

    def busiest(self, n: int | None) -> list[FailurePoint]:
        pts = sorted(self.points.values(), key=lambda p: (-p.count, p.boundary))
        return pts if n is None else pts[:n]

    def model_calls_before(self, ts: float) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self.events:
            if e.kind == "model" and e.ts < ts:
                out[e.boundary] = max(out[e.boundary], e.call_index)
        return dict(out)

    def event_at(self, boundary: str, call_index: int, session: int | None = None) -> Event | None:
        for e in self.events:
            if e.boundary == boundary and e.call_index == call_index and (
                    session is None or e.session == session):
                return e
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "points": [{**{k: v for k, v in asdict(p).items() if k != "calls"},
                        "count": p.count, "tags": sorted(p.tags)} for p in self.busiest(None)],
            "edges": self.edges,
        }


def _kind(e: Event) -> str:
    return e.kind


def build_graph(events: list[Event]) -> FailurePointGraph:
    g = FailurePointGraph(events=sorted(events, key=lambda e: e.ts))
    schemas: dict[str, str] = {}
    destructive: set[str] = set()
    for e in g.events:
        if e.op == "tools/list":
            server = e.boundary.split(":", 1)[1]
            for tool, h in (e.detail.get("schemas") or {}).items():
                schemas[f"tool:{server}.{tool}"] = h
            destructive |= {f"tool:{server}.{t}" for t in e.detail.get("destructive", [])}
    prev: str | None = None
    for e in g.events:
        if e.call_index <= 0:  # registrations and tool listings are not calls
            continue
        p = g.points.get(e.boundary)
        if p is None:
            op = e.op if e.kind in (BoundaryKind.STATE.value, BoundaryKind.DEPENDENCY.value) else None
            if e.kind == BoundaryKind.STATE.value:
                op = e.boundary.rsplit(".", 1)[-1]
            p = g.points[e.boundary] = FailurePoint(e.boundary, _kind(e), provider=e.provider, op=op)
        p.calls.append(CallInfo(e.call_index, e.session, e.ts, list(e.tags), e.req_hash))
        p.destructive = p.destructive or e.destructive or e.boundary in destructive
        p.schema_hash = schemas.get(e.boundary, p.schema_hash)
        if prev is not None and prev != e.boundary:
            g.edges.setdefault(prev, {}).setdefault(e.boundary, 0)
            g.edges[prev][e.boundary] += 1
        prev = e.boundary
        g.sessions = max(g.sessions, e.session)
    # Directories registered with tap(path) are failure points for between-session faults.
    for e in g.events:
        if e.op == "register" and e.boundary not in g.points:
            g.points[e.boundary] = FailurePoint(e.boundary, "state", op="dir")
    return g
