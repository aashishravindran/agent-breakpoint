"""``agent-breakpoint serve``: an MCP server so Claude Code or Kiro can run the quick tier and read
the matrix while writing the agent.

It is an operator's remote control, not a connection to the agent under test. Targets, budgets
and mutation settings come only from the config file on disk, read when the server starts; no
tool takes a target, URL, command or budget as an argument, so a tricked model can't point the
harness at anything else.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from agent_breakpoint.config import load_config
from agent_breakpoint.grades import Grade


def build_server(config_path: str):
    from mcp.server.mcpserver import MCPServer  # type: ignore[import-not-found]
    from mcp.types import ToolAnnotations  # type: ignore[import-not-found]

    cfg_path = str(Path(config_path).resolve())
    cfg = load_config(cfg_path)  # validated once; scenario names are the only selectors
    lock = threading.Lock()
    server = MCPServer("agent-breakpoint", instructions=(
        "Fault-injection test harness for the agent in this repo. run_quick breaks the agent's "
        "model, tools and state one fault at a time and grades each: tolerated, degraded, "
        "failed loudly, failed silently, unsafe. Fix silent and unsafe cells first."))

    def latest() -> dict[str, Any]:
        p = cfg.run_path / "latest.json"
        if not p.exists():
            raise ValueError("no run yet; call run_quick first")
        return json.loads(p.read_text())

    def scenario_name(name: str | None) -> str:
        return cfg.scenario(name).name  # raises on unknown names; never builds new targets

    def brief(c: dict[str, Any]) -> dict[str, Any]:
        worst = Grade(c["stats"]["worst"])
        bad = next((t for t in c["trials"] if Grade(t["grade"]) == worst), c["trials"][0])
        return {"cell": c["key"], "label": c.get("label"), "grade": worst.value,
                "distribution": c["stats"]["distribution"], "flaky": c["stats"]["flaky"],
                "why": [r for r in bad["reasons"] if not r.startswith(("state=", "claim="))],
                "answer": bad.get("answer"), "trial_dir": bad.get("trial_dir")}

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def list_scenarios() -> list[dict[str, Any]]:
        """Scenarios defined in agent-breakpoint.yaml."""
        return [{"name": s.name, "prompt": s.task.get("prompt"), "invariants": s.invariants,
                 "sessions": s.sessions} for s in cfg.scenarios]

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
    def run_quick(scenario: str | None = None) -> dict[str, Any]:
        """Run the quick tier (single faults, few runs, capped cost) and return failing cells."""
        from agent_breakpoint.report import summary_line
        from agent_breakpoint.suite import run_tier

        name = scenario_name(scenario)
        with lock:
            matrix = run_tier(load_config(cfg_path), "quick", name).to_json()
        fail = [Grade(g) for g in matrix["gate"]["fail_on"]]
        bad = [brief(c) for c in matrix["cells"] if Grade(c["stats"]["worst"]) in fail]
        return {"summary": summary_line(matrix), "failing": bad, "run_dir": matrix["run_dir"]}

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def get_matrix(only_failures: bool = True) -> dict[str, Any]:
        """The last tolerance matrix: cells with grades and reasons."""
        from agent_breakpoint.report import summary_line

        m = latest()
        cells = [brief(c) for c in m["cells"]
                 if not only_failures or Grade(c["stats"]["worst"]) not in (Grade.TOLERATED,)]
        return {"summary": summary_line(m), "cells": cells}

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def explain_cell(cell: str) -> dict[str, Any]:
        """Everything about one cell: the fault, every trial's grade and reasons, the trace."""
        from agent_breakpoint.testing import find_cell
        from agent_breakpoint.trace import read_trace

        c = find_cell(latest(), cell)
        trials = []
        for t in c["trials"]:
            events = read_trace(Path(t["trial_dir"])) if Path(t["trial_dir"]).exists() else []
            trials.append({**t, "trace": [
                {"boundary": e.boundary, "call": e.call_index, "status": e.status,
                 "faults": e.faults, "tool": e.tool} for e in events if e.call_index > 0]})
        return {"cell": c["key"], "faults": c["faults"], "stats": c["stats"], "trials": trials}

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
    def replay_cell(cell: str) -> dict[str, Any]:
        """Rerun one cell against the current code, to check a fix against the exact fault."""
        from agent_breakpoint.testing import find_cell, run_cell

        m = latest()
        c = find_cell(m, cell)
        with lock:
            res = run_cell(load_config(cfg_path), c, scenario=m["scenario"])
        return {"before": c["stats"]["worst"], "after": res.worst.value, "reasons": res.reasons()}

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def suggested_invariants() -> str:
        """Invariants suggested from the last recorded baseline's end-state diff, as Python, for
        a person to approve."""
        p = cfg.run_path / "suggested_invariants.py"
        return p.read_text() if p.exists() else "No suggestions yet; run `agent-breakpoint record`."

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def list_faults() -> list[dict[str, str]]:
        """The fault catalog."""
        from agent_breakpoint.faults import registry

        return [{"id": f.id, "class": f.fault_class.value, "description": f.description}
                for f in sorted(registry().faults.values(), key=lambda f: f.id)]

    return server


def serve(config_path: str) -> None:
    build_server(config_path).run("stdio")
