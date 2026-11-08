"""Run a tier: baseline, discovery, planned cells, statistics, the tolerance matrix."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_breakpoint.combos import candidates, ddmin
from agent_breakpoint.config import Config, Scenario, Tier
from agent_breakpoint.discovery import FailurePointGraph, build_graph
from agent_breakpoint.grades import Grade
from agent_breakpoint.planner import Cell, make_cell, plan_single
from agent_breakpoint.runner import Harness, TrialResult
from agent_breakpoint.stats import CellStats, cell_stats
from agent_breakpoint.trace import stable_hash

Progress = Callable[[str], None]


@dataclass
class CellResult:
    cell: Cell
    trials: list[dict[str, Any]]
    stats: CellStats
    kind: str
    reused: bool = False
    minimal: list[str] | None = None  # ddmin result for failing combinations

    @property
    def grades(self) -> list[Grade]:
        return [Grade(t["grade"]) for t in self.trials]

    def to_json(self) -> dict[str, Any]:
        return {**self.cell.to_json(), "boundary": self.cell.boundary, "kind": self.kind,
                "fault_ids": self.cell.fault_ids, "stats": self.stats.to_json(),
                "trials": self.trials, "reused": self.reused, "minimal": self.minimal}


@dataclass
class Matrix:
    run_id: str
    scenario: str
    tier: str
    created: float
    baseline: dict[str, Any]
    graph: dict[str, Any]
    cells: list[CellResult] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""

    def failing(self, fail_on: list[Grade]) -> list[CellResult]:
        return [c for c in self.cells if any(g in fail_on for g in c.grades)]

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "scenario": self.scenario, "tier": self.tier,
                "created": self.created, "baseline": self.baseline, "graph": self.graph,
                "cells": [c.to_json() for c in self.cells], "skipped": self.skipped,
                "totals": self.totals, "gate": self.gate, "run_dir": self.run_dir}


def _config_hash(cfg: Config, scenario: Scenario) -> str:
    return stable_hash({"entrypoint": cfg.entrypoint, "scenario": scenario.model_dump(),
                        "seed": cfg.seed, "packs": cfg.packs, "safety": cfg.safety})


class ResultCache:
    """Incremental runs: a cell whose fingerprint (fault + the baseline calls leading to it, incl.
    tool schemas) is unchanged reuses its previous trials."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, ns: str, cell: Cell, runs: int) -> list[dict[str, Any]] | None:
        hit = self.data.get(ns, {}).get(cell.fingerprint)
        return hit if hit and len(hit) >= runs else None

    def put(self, ns: str, cell: Cell, trials: list[dict[str, Any]]) -> None:
        self.data.setdefault(ns, {})[cell.fingerprint] = trials

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data))


def run_tier(cfg: Config, tier_name: str = "quick", scenario_name: str | None = None, *,
             incremental: bool = True, only_kinds: set[str] | None = None,
             cells_filter: Callable[[Cell], bool] | None = None,
             progress: Progress = lambda s: None, harness: Harness | None = None) -> Matrix:
    tier = cfg.tier(tier_name)
    scenario = cfg.scenario(scenario_name)
    own = harness is None
    h = harness or Harness(cfg)
    if own:
        h.start()
    try:
        return _run(cfg, h, tier, tier_name, scenario, incremental, only_kinds, cells_filter, progress)
    finally:
        if own:
            h.stop()


def run_baseline(h: Harness, scenario: Scenario, runs: int, progress: Progress
                 ) -> tuple[list[TrialResult], FailurePointGraph]:
    progress(f"baseline 1/{runs} (recording)")
    first = h.run_trial(scenario, None, mode="record", label="baseline")
    graph = build_graph(first.events)
    trials = [first]
    for i in range(2, runs + 1):
        progress(f"baseline {i}/{runs}")
        trials.append(h.run_trial(scenario, None, label="baseline", replay=False, baseline=first))
    return trials, graph


def _run(cfg: Config, h: Harness, tier: Tier, tier_name: str, scenario: Scenario,
         incremental: bool, only_kinds: set[str] | None,
         cells_filter: Callable[[Cell], bool] | None, progress: Progress) -> Matrix:
    started = time.time()
    baseline, graph = run_baseline(h, scenario, max(1, min(tier.runs, 3)), progress)
    base_grades = [t.grade for t in baseline]
    spent = sum(t.usd for t in baseline)
    cap = tier.max_usd if tier.max_usd is not None else cfg.budgets.max_usd * 50
    cache = ResultCache(cfg.run_path / "results-cache.json")
    ns = _config_hash(cfg, scenario)

    cells = plan_single(graph, h.registry, tier, only_kinds, cells_filter)
    progress(f"{len(graph.points)} failure points, {len(cells)} cells planned")
    matrix = Matrix(h.run_id, scenario.name, tier_name, started,
                    baseline={"n": len(baseline), "grades": [g.value for g in base_grades],
                              "trials": [t.to_json() for t in baseline]},
                    graph=graph.to_json(), run_dir=str(h.run_dir))

    def run_cell(cell: Cell) -> CellResult | None:
        nonlocal spent
        kind = graph.points[cell.boundary].kind if cell.boundary in graph.points else "?"
        if incremental and (hit := cache.get(ns, cell, tier.runs)):
            grades = [Grade(t["grade"]) for t in hit[: tier.runs]]
            return CellResult(cell, hit[: tier.runs], cell_stats(grades, base_grades, tier.confidence),
                              kind, reused=True)
        trials = []
        for i in range(tier.runs):
            if spent > cap:
                matrix.skipped.append({"key": cell.key, "reason": f"tier dollar cap ${cap:.2f}"})
                return None
            progress(f"{cell.label or cell.key} run {i + 1}/{tier.runs}")
            tr = h.run_trial(scenario, cell, label="cell", baseline=baseline[0])
            spent += tr.usd
            trials.append(tr.to_json())
        grades = [Grade(t["grade"]) for t in trials]
        cache.put(ns, cell, trials)
        return CellResult(cell, trials, cell_stats(grades, base_grades, tier.confidence), kind)

    singles: list[tuple[Cell, Grade]] = []
    for cell in cells:
        res = run_cell(cell)
        if res is None:
            continue
        matrix.cells.append(res)
        singles.append((cell, res.stats.worst))

    if tier.combinations:
        combos = candidates(graph, singles, tier.max_combination_size, limit=tier.max_combinations)
        progress(f"{len(combos)} fault combinations")
        fail_on = cfg.gate.fail_on
        for cell in combos:
            res = run_cell(cell)
            if res is None:
                continue
            if any(g in fail_on for g in res.grades) and len(cell.faults) > 2:
                def fails(specs, _cell=cell) -> bool:
                    sub = make_cell(graph, specs)
                    tr = h.run_trial(scenario, sub, label="ddmin", baseline=baseline[0])
                    return tr.grade in fail_on
                res.minimal = [f.key for f in ddmin(cell.faults, fails)]
            elif any(g in fail_on for g in res.grades):
                res.minimal = [f.key for f in cell.faults]
            matrix.cells.append(res)

    cache.save()
    fail_on = cfg.gate.fail_on
    matrix.totals = {
        "usd": round(spent, 4), "cells": len(matrix.cells),
        "trials": sum(len(c.trials) for c in matrix.cells) + len(baseline),
        "reused": sum(c.reused for c in matrix.cells),
        "duration_s": round(time.time() - started, 1),
        "replay_hits": sum(t.get("replay_hits", 0) for c in matrix.cells for t in c.trials),
        "replay_misses": sum(t.get("replay_misses", 0) for c in matrix.cells for t in c.trials),
    }
    failing = matrix.failing(fail_on)
    nothing_ran = not matrix.cells and bool(matrix.skipped)
    matrix.gate = {"fail_on": [g.value for g in fail_on], "failed_cells": [c.cell.key for c in failing],
                   "passed": not failing and not nothing_ran}
    save_matrix(cfg, matrix)
    return matrix


def save_matrix(cfg: Config, matrix: Matrix) -> Path:
    run_dir = Path(matrix.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(matrix.to_json(), indent=2, default=str)
    (run_dir / "matrix.json").write_text(data)
    (cfg.run_path / "latest.json").write_text(data)
    return run_dir / "matrix.json"


def load_matrix_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
