"""Plan the cells of the tolerance matrix: failure point × fault, one fault per cell (quick and
full tiers), plus fault combinations in the full tier (see ``combos``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_breakpoint.config import Tier
from agent_breakpoint.controller import FaultSpec
from agent_breakpoint.discovery import FailurePoint, FailurePointGraph
from agent_breakpoint.faults.base import BoundaryKind, Fault, Registry
from agent_breakpoint.trace import stable_hash

# Faults the harness applies itself, between sessions of an episode, to tapped directories.
DIR_FAULTS = ("state.lost_write", "state.memory_poisoning", "state.torn_checkpoint",
              "state.schema_drift")


@dataclass
class Cell:
    faults: list[FaultSpec]
    replay_until: dict[str, int] = field(default_factory=dict)
    fingerprint: str = ""
    label: str = ""

    @property
    def key(self) -> str:
        return " + ".join(f.key for f in self.faults)

    @property
    def boundary(self) -> str:
        return self.faults[0].boundary

    @property
    def fault_ids(self) -> list[str]:
        return [f.fault for f in self.faults]

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "faults": [asdict(f) for f in self.faults],
                "replay_until": self.replay_until, "fingerprint": self.fingerprint,
                "label": self.label}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Cell:
        return cls([FaultSpec.from_json(f) for f in data["faults"]],
                   dict(data.get("replay_until", {})), data.get("fingerprint", ""),
                   data.get("label", ""))


def _kind(point: FailurePoint) -> BoundaryKind:
    return BoundaryKind(point.kind)


def eligible_calls(point: FailurePoint, fault: Fault) -> list[int]:
    out = []
    for c in point.calls:
        tags = set(c.tags)
        if point.destructive:
            tags.add("destructive")
        if all(r in tags for r in fault.requires):
            out.append(c.index)
    return out


def applicable(point: FailurePoint, reg: Registry) -> list[Fault]:
    kind = _kind(point)
    if kind == BoundaryKind.STATE and point.op == "dir":
        return [reg.faults[f] for f in DIR_FAULTS if f in reg.faults]
    faults = []
    for f in reg.faults.values():
        if not f.applies(kind, provider=point.provider, op=point.op):
            continue
        if kind == BoundaryKind.STATE and f.ops and point.op not in f.ops:
            continue
        faults.append(f)
    return sorted(faults, key=lambda f: f.id)


def make_cell(graph: FailurePointGraph, specs: list[FaultSpec], label: str = "") -> Cell:
    """Attach the fork-at-fault replay prefix and the incremental fingerprint."""
    replay: dict[str, int] = {}
    fps = []
    first_ts = None
    for s in specs:
        point = graph.points.get(s.boundary)
        if point is None:
            continue
        at = s.at_call or 1
        call = next((c for c in point.calls if c.index == at), None)
        if call is not None and (first_ts is None or call.ts < first_ts):
            first_ts = call.ts
        fps.append(point.fingerprint(upto=at))
    if first_ts is not None:
        for boundary, n in graph.model_calls_before(first_ts).items():
            replay[boundary] = n
    for s in specs:  # a fault on a model call itself: replay strictly before it
        if s.boundary.startswith("model:") and s.at_call:
            replay[s.boundary] = min(replay.get(s.boundary, s.at_call - 1), s.at_call - 1)
    fp = stable_hash({"faults": [asdict(s) for s in specs], "points": fps})
    return Cell(specs, replay, fp, label)


def plan_single(graph: FailurePointGraph, reg: Registry, tier: Tier,
                only_kinds: set[str] | None = None, keep=None) -> list[Cell]:
    cells: list[Cell] = []
    points = graph.busiest(tier.busiest)
    # Taps registered with tap(path) have no calls; keep them when sessions allow episodes.
    points += [p for p in graph.points.values() if p.op == "dir" and p not in points]
    for point in points:
        if only_kinds and point.kind not in only_kinds:
            continue
        for fault in applicable(point, reg):
            if point.op == "dir":
                if graph.sessions < 2:
                    continue
                spec = FaultSpec(point.boundary, fault.id, at_call=None, session=2)
                cells.append(make_cell(graph, [spec], f"{point.boundary} × {fault.id} (between sessions)"))
                continue
            calls = eligible_calls(point, fault)
            if not calls:
                continue
            sweep = int(fault.default_params.get("sweep", 0))
            if sweep:  # crash-style faults: where in the run they land is the whole question
                step = max(1, len(calls) // sweep)
                picks = sorted(set(calls[::step][:sweep]) | {calls[-1]})
            elif tier.runs <= 3 or len(calls) == 1:
                picks = [calls[0]]
            else:
                picks = sorted({calls[0], calls[-1]})
            for at in picks:
                sticky = bool(fault.default_params.get("sticky"))
                session = next((c.session for c in point.calls if c.index == at), None)
                spec = FaultSpec(point.boundary, fault.id, at_call=at, sticky=sticky,
                                 session=session if graph.sessions > 1 else None)
                cells.append(make_cell(graph, [spec], f"{point.boundary} × {fault.id} @{at}"))
    if keep is not None:
        cells = [c for c in cells if keep(c)]
    return cells[: tier.max_cells]
