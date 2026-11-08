"""Replayable failures: rerun one cell exactly, or export it as a pytest test, so a fix is
checked against the same fault."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_breakpoint.config import Config, load_config
from agent_breakpoint.grades import Grade
from agent_breakpoint.planner import Cell
from agent_breakpoint.runner import Harness, TrialResult


@dataclass
class CellRun:
    baseline: TrialResult
    trials: list[TrialResult]

    @property
    def grades(self) -> list[Grade]:
        return [t.grade for t in self.trials]

    @property
    def worst(self) -> Grade:
        return Grade.worst(self.grades)

    def reasons(self) -> list[str]:
        bad = next((t for t in self.trials if t.grade == self.worst), self.trials[0])
        return bad.result.reasons


def find_cell(matrix: dict[str, Any], key: str) -> dict[str, Any]:
    for c in matrix["cells"]:
        if c["key"] == key or c.get("label") == key:
            return c
    matches = [c for c in matrix["cells"] if key in c["key"] or key in (c.get("label") or "")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no cell matching {key!r}")
    raise KeyError(f"{key!r} is ambiguous: " + ", ".join(c["key"] for c in matches[:5]))


def latest_matrix(cfg: Config) -> dict[str, Any]:
    p = cfg.run_path / "latest.json"
    if not p.exists():
        raise FileNotFoundError("no previous run; run `agent-breakpoint quick` first")
    return json.loads(p.read_text())


def run_cell(cfg: Config, cell: Cell | dict[str, Any], *, scenario: str | None = None,
             runs: int = 1, harness: Harness | None = None) -> CellRun:
    """Record a fresh baseline (so replay reflects the current code), then run the cell."""
    if isinstance(cell, dict):
        cell = Cell.from_json(cell)
    sc = cfg.scenario(scenario)
    own = harness is None
    h = harness or Harness(cfg)
    if own:
        h.start()
    try:
        base = h.run_trial(sc, None, mode="record", label="baseline")
        trials = [h.run_trial(sc, cell, label="replay", baseline=base) for _ in range(runs)]
        return CellRun(base, trials)
    finally:
        if own:
            h.stop()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def export_pytest(config_path: str | Path, cell: dict[str, Any], scenario: str, dest: Path,
                  fail_on: list[str]) -> Path:
    name = _slug(cell.get("label") or cell["key"])
    payload = {k: cell[k] for k in ("key", "faults", "replay_until", "fingerprint", "label")}
    src = f'''"""Regression test exported by agent-breakpoint for cell:
    {cell.get("label") or cell["key"]}
Rerun just this fault with: agent-breakpoint replay {json.dumps(cell["key"])}
"""

import json

from agent_breakpoint.config import load_config
from agent_breakpoint.grades import Grade
from agent_breakpoint.testing import run_cell

CONFIG = {json.dumps(str(Path(config_path).resolve()))}
SCENARIO = {json.dumps(scenario)}
FAIL_ON = {{Grade(g) for g in {json.dumps(fail_on)}}}
CELL = json.loads({json.dumps(json.dumps(payload))})


def test_{name}():
    result = run_cell(load_config(CONFIG), CELL, scenario=SCENARIO)
    assert result.worst not in FAIL_ON, "\\n".join(result.reasons())
'''
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = dest if dest.suffix == ".py" else dest / f"test_abk_{name}.py"
    path.write_text(src)
    return path


__all__ = ["CellRun", "export_pytest", "find_cell", "latest_matrix", "load_config", "run_cell"]
