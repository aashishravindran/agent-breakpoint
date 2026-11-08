"""Render a tolerance matrix (the ``matrix.json`` dict) for people and for CI."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from xml.sax.saxutils import escape

from rich.console import Console
from rich.table import Table
from rich.text import Text

from agent_breakpoint.grades import Grade

STYLE = {
    Grade.TOLERATED: "green", Grade.DEGRADED: "cyan", Grade.FAILED_LOUDLY: "yellow",
    Grade.FAILED_SILENTLY: "bold red", Grade.UNSAFE: "bold white on red",
}


def _grades(cell: dict[str, Any]) -> list[Grade]:
    return [Grade(t["grade"]) for t in cell["trials"]]


def _fault_short(fid: str) -> str:
    return fid.split(".", 1)[1] if "." in fid else fid


def _cell_title(cell: dict[str, Any]) -> str:
    return cell.get("label") or cell["key"]


def _short_boundary(b: str) -> str:
    kind, _, rest = b.partition(":")
    return {"tool": "", "model": "model:", "dep": "dep:", "state": "state:"}.get(kind, kind + ":") + rest


def grid(matrix: dict[str, Any]) -> Table:
    """Fault × failure point, one symbol per cell: its worst grade over all runs."""
    cells: dict[tuple[str, str], Grade] = {}
    boundaries: list[str] = []
    for c in matrix["cells"]:
        if len(c["faults"]) != 1:
            continue
        key = (c["fault_ids"][0], c["boundary"])
        worst = Grade.worst(_grades(c))
        cells[key] = Grade.worst([cells[key], worst]) if key in cells else worst
        if c["boundary"] not in boundaries:
            boundaries.append(c["boundary"])
    order = {"model": 0, "tool": 1, "dep": 2, "state": 3, "orchestration": 4}
    boundaries.sort(key=lambda b: (order.get(b.split(":")[0], 9), b))
    faults = sorted({f for f, _ in cells})
    t = Table(title=f"Tolerance matrix: {matrix['scenario']} ({matrix['tier']})",
              header_style="bold")
    t.add_column("fault", no_wrap=True)
    for b in boundaries:
        t.add_column(_short_boundary(b), justify="center", overflow="fold")
    for f in faults:
        row = [Text(cells[(f, b)].symbol, style=STYLE[cells[(f, b)]]) if (f, b) in cells
               else Text("·", style="dim") for b in boundaries]
        t.add_row(f, *row)
    return t


def _why(trial: dict[str, Any]) -> str:
    reasons = [r for r in trial["reasons"] if not r.startswith(("state=", "claim="))]
    findings = [r for r in reasons if r.startswith("[")]
    why = (findings or reasons or [f"claim={trial.get('claim')}"])[0]
    rc = trial.get("root_cause") or {}
    if rc.get("diverged_at"):
        why += f" (behaviour diverged at {rc['diverged_at']})"
    return why


def details(matrix: dict[str, Any], only_bad: bool = True) -> Table:
    fail = {Grade(g) for g in matrix.get("gate", {}).get("fail_on", [])} or {
        Grade.FAILED_SILENTLY, Grade.UNSAFE}
    multi = any(c["stats"]["n"] > 1 for c in matrix["cells"])
    t = Table(title="Cells that fail the gate" if only_bad else "Cells", header_style="bold",
              show_lines=False, expand=False)
    t.add_column("grade", no_wrap=True)
    t.add_column("cell", overflow="fold", ratio=2)
    if multi:
        t.add_column("grades", no_wrap=True)
        t.add_column("success vs baseline", no_wrap=True)
    t.add_column("why", overflow="fold", ratio=3)
    base_n = matrix.get("baseline", {}).get("n", 0)
    base_ok = sum(g == "tolerated" for g in matrix.get("baseline", {}).get("grades", []))
    ordered = sorted(matrix["cells"], key=lambda c: (-Grade(c["stats"]["worst"]).severity, c["key"]))
    for c in ordered:
        worst = Grade(c["stats"]["worst"])
        st = c["stats"]
        if only_bad and worst not in fail and not st["flaky"]:
            continue
        bad = next((t for t in c["trials"] if Grade(t["grade"]) == worst), c["trials"][0])
        name = _cell_title(c) + (" [flaky]" if st["flaky"] else "")
        row = [Text(worst.label, style=STYLE[worst]), Text(name)]
        if multi:
            ok = st["distribution"].get("tolerated", 0)
            row += [" ".join(f"{Grade(k).symbol}{v}" for k, v in sorted(st["distribution"].items())),
                    f"{ok}/{st['n']} vs {base_ok}/{base_n}"]
        t.add_row(*row, Text(_why(bad)))
    return t


def summary_line(matrix: dict[str, Any]) -> str:
    counts: dict[Grade, int] = defaultdict(int)
    for c in matrix["cells"]:
        counts[Grade(c["stats"]["worst"])] += 1
    parts = [f"{counts[g]} {g.label.lower()}" for g in Grade if counts[g]]
    tot = matrix.get("totals", {})
    gate = matrix.get("gate", {})
    verdict = "PASS" if gate.get("passed") else "FAIL"
    return (f"{len(matrix['cells'])} cells: {', '.join(parts)}. "
            f"${tot.get('usd', 0):.4f}, {tot.get('trials', 0)} trials, "
            f"{tot.get('replay_hits', 0)} replayed model calls, {tot.get('duration_s', 0)}s. "
            f"Gate ({', '.join(gate.get('fail_on', []))}): {verdict}")


def print_matrix(matrix: dict[str, Any], console: Console | None = None, all_cells: bool = False) -> None:
    console = console or Console()
    console.print(grid(matrix))
    legend = "  ".join(f"[{STYLE[g]}]{g.symbol}[/] {g.label}" for g in Grade)
    console.print(legend)
    console.print(details(matrix, only_bad=not all_cells))
    if matrix.get("skipped"):
        console.print(f"[yellow]{len(matrix['skipped'])} cells skipped[/]: "
                      f"{matrix['skipped'][0]['reason']}")
    console.print(summary_line(matrix))


def markdown(matrix: dict[str, Any]) -> str:
    lines = [f"## agent-breakpoint: {matrix['scenario']} ({matrix['tier']})", "",
             summary_line(matrix), "",
             "| Grade | Cell | Runs | Why |", "| --- | --- | --- | --- |"]
    for c in sorted(matrix["cells"], key=lambda c: (-Grade(c["stats"]["worst"]).severity, c["key"])):
        worst = Grade(c["stats"]["worst"])
        dist = " ".join(f"{Grade(k).symbol}×{v}" for k, v in sorted(c["stats"]["distribution"].items()))
        bad = next((t for t in c["trials"] if Grade(t["grade"]) == worst), c["trials"][0])
        why = _why(bad)
        lines.append(f"| {worst.label} | `{_cell_title(c)}` | {dist} | {why.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def junit(matrix: dict[str, Any], fail_on: list[str] | None = None) -> str:
    fail = {Grade.parse(g) for g in (fail_on or matrix.get("gate", {}).get("fail_on", []))}
    cases = []
    failures = 0
    for c in matrix["cells"]:
        worst = Grade(c["stats"]["worst"])
        name = escape(_cell_title(c), {'"': "&quot;"})
        body = ""
        if worst in fail:
            failures += 1
            bad = next(t for t in c["trials"] if Grade(t["grade"]) == worst)
            msg = escape("; ".join(bad["reasons"]), {'"': "&quot;"})
            body = f'<failure message="{worst.label}" type="{worst.value}">{msg}</failure>'
        cases.append(f'  <testcase classname="agent_breakpoint.{escape(matrix["scenario"])}" '
                     f'name="{name}" time="{sum(t["duration_s"] for t in c["trials"]):.3f}">{body}</testcase>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<testsuite name="agent-breakpoint" tests="{len(cases)}" failures="{failures}">\n'
            + "\n".join(cases) + "\n</testsuite>\n")


_SARIF_LEVEL = {Grade.UNSAFE: "error", Grade.FAILED_SILENTLY: "error",
                Grade.FAILED_LOUDLY: "warning", Grade.DEGRADED: "note"}


def sarif(matrix: dict[str, Any], config_path: str = "agent-breakpoint.yaml") -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    results = []
    for c in matrix["cells"]:
        worst = Grade(c["stats"]["worst"])
        if worst == Grade.TOLERATED:
            continue
        rule_id = "ABK/" + "+".join(c["fault_ids"])
        rules.setdefault(rule_id, {"id": rule_id, "name": rule_id.replace("/", "_"),
                                   "shortDescription": {"text": f"Agent under {', '.join(c['fault_ids'])}"}})
        bad = next(t for t in c["trials"] if Grade(t["grade"]) == worst)
        results.append({
            "ruleId": rule_id, "level": _SARIF_LEVEL.get(worst, "note"),
            "message": {"text": f"{worst.label}: {_cell_title(c)}. "
                                + "; ".join(bad["reasons"])},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": config_path},
                                                "region": {"startLine": 1}}}],
            "partialFingerprints": {"cell": c["key"]},
            "properties": {"grade": worst.value, "distribution": c["stats"]["distribution"],
                           "replay": f"agent-breakpoint replay '{c['key']}'"},
        })
    return {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
            "runs": [{"tool": {"driver": {"name": "agent-breakpoint", "version": "0.1.0",
                                          "informationUri": "https://github.com/aashishravindran/agent-breakpoint",
                                          "rules": list(rules.values())}},
                      "results": results}]}


def diff(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """Cells whose worst grade changed between two matrices (e.g. main vs this branch)."""
    before = {c["key"]: Grade(c["stats"]["worst"]) for c in old["cells"]}
    out = []
    for c in new["cells"]:
        g = Grade(c["stats"]["worst"])
        b = before.get(c["key"])
        if b is None:
            out.append({"key": c["key"], "label": _cell_title(c), "before": None, "after": g.value,
                        "change": "new"})
        elif b != g:
            out.append({"key": c["key"], "label": _cell_title(c), "before": b.value, "after": g.value,
                        "change": "worse" if g.severity > b.severity else "better"})
    gone = set(before) - {c["key"] for c in new["cells"]}
    out += [{"key": k, "label": k, "before": before[k].value, "after": None, "change": "removed"}
            for k in sorted(gone)]
    return out


def to_json(matrix: dict[str, Any]) -> str:
    return json.dumps(matrix, indent=2)
