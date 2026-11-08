"""``agent-breakpoint`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from agent_breakpoint.config import DEFAULT_CONFIG, load_config
from agent_breakpoint.grades import Grade

console = Console(stderr=True)
out = Console()


def _progress(msg: str) -> None:
    console.print(f"[dim]· {msg}[/]")


def _write_outputs(args: argparse.Namespace, matrix: dict, cfg_path: str) -> None:
    from agent_breakpoint import report

    if getattr(args, "json", None):
        Path(args.json).write_text(report.to_json(matrix))
    if getattr(args, "junit", None):
        Path(args.junit).write_text(report.junit(matrix))
    if getattr(args, "sarif", None):
        Path(args.sarif).write_text(json.dumps(report.sarif(matrix, cfg_path), indent=2))
    if getattr(args, "markdown", None):
        Path(args.markdown).write_text(report.markdown(matrix))


def _gate(matrix: dict, fail_on: list[str] | None) -> int:
    fail = {Grade.parse(g) for g in (fail_on or matrix["gate"]["fail_on"])}
    bad = [c for c in matrix["cells"] if Grade(c["stats"]["worst"]) in fail]
    return 1 if bad else 0


def cmd_tier(args: argparse.Namespace) -> int:
    from agent_breakpoint import report
    from agent_breakpoint.suite import run_tier

    cfg = load_config(args.config)
    if args.fail_on:
        cfg.gate.fail_on = [Grade.parse(g) for g in args.fail_on.split(",")]
    kinds = set(args.only.split(",")) if args.only else None
    cells_filter = None
    if args.cells:
        import fnmatch

        pats = args.cells.split(",")
        cells_filter = lambda c: any(fnmatch.fnmatchcase(c.label or c.key, p)  # noqa: E731
                                     or fnmatch.fnmatchcase(c.key, p) for p in pats)
    matrix = run_tier(cfg, args.tier, args.scenario, incremental=not args.no_incremental,
                      only_kinds=kinds, cells_filter=cells_filter,
                      progress=_progress if not args.quiet else lambda s: None)
    data = matrix.to_json()
    report.print_matrix(data, out, all_cells=args.all)
    if args.diff:
        old = json.loads(Path(args.diff).read_text())
        _print_diff(report.diff(old, data))
    _write_outputs(args, data, args.config)
    return _gate(data, None)


def _print_diff(changes: list[dict]) -> None:
    if not changes:
        out.print("No grade changes against the comparison matrix.")
        return
    for c in changes:
        style = {"worse": "red", "better": "green"}.get(c["change"], "yellow")
        out.print(f"[{style}]{c['change']:8}[/] {c['label']}: {c['before']} → {c['after']}")


def cmd_report(args: argparse.Namespace) -> int:
    from agent_breakpoint import report

    path = Path(args.matrix) if args.matrix else load_config(args.config).run_path / "latest.json"
    data = json.loads(path.read_text())
    report.print_matrix(data, out, all_cells=args.all)
    _write_outputs(args, data, args.config)
    return _gate(data, args.fail_on.split(",") if args.fail_on else None)


def cmd_diff(args: argparse.Namespace) -> int:
    from agent_breakpoint import report

    old = json.loads(Path(args.old).read_text())
    new = json.loads(Path(args.new).read_text())
    changes = report.diff(old, new)
    _print_diff(changes)
    if args.markdown:
        lines = ["| Change | Cell | Before | After |", "| --- | --- | --- | --- |"]
        lines += [f"| {c['change']} | `{c['label']}` | {c['before']} | {c['after']} |" for c in changes]
        Path(args.markdown).write_text("\n".join(lines) + "\n")
    return 1 if any(c["change"] == "worse" for c in changes) else 0


def cmd_init(args: argparse.Namespace) -> int:
    from agent_breakpoint.onboarding import detect, write_init

    root = Path(args.dir).resolve()
    d = detect(root)
    out.print(f"Frameworks: {', '.join(d.frameworks) or 'none detected'}")
    out.print(f"Model providers: {', '.join(d.providers) or 'none detected (defaulting to openai)'}")
    out.print(f"MCP config: {d.mcp_config or 'none'}"
              + (f" ({', '.join(d.mcp_servers)})" if d.mcp_servers else ""))
    out.print(f"Checkpointer in code: {'yes' if d.checkpointer else 'no'}")
    for p in write_init(root, force=args.force):
        out.print(f"[green]wrote[/] {p.relative_to(root)}")
    out.print("Next: fill in agent_breakpoint_entry.py, then run `agent-breakpoint record`.")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from agent_breakpoint import report
    from agent_breakpoint.discovery import build_graph
    from agent_breakpoint.onboarding import suggest_invariants
    from agent_breakpoint.runner import Harness
    from agent_breakpoint.suite import run_tier

    cfg = load_config(args.config)
    sc = cfg.scenario(args.scenario)
    with Harness(cfg) as h:
        _progress("recording a baseline run")
        base = h.run_trial(sc, None, mode="record", label="baseline")
        out.print(f"Baseline: [bold]{base.grade.label}[/] — {base.outcome.answer[:200] or base.outcome.error}")
        graph = build_graph(base.events)
        out.print("Failure points:")
        for p in graph.busiest(None):
            flag = " (destructive)" if p.destructive else ""
            out.print(f"  {p.boundary:40} {p.count:3} calls{flag}")
        suggestions = suggest_invariants(base.trial_dir / "pre-session-snapshot",
                                         base.trial_dir / "sandbox")
        sugg_path = cfg.run_path / "suggested_invariants.py"
        sugg_path.write_text(suggestions)
        out.print(f"Suggested invariants from the end-state diff: {sugg_path}")
        if args.no_matrix:
            return 0
        _progress("first matrix: model and tool faults, graded by the claim oracle"
                  + ("" if sc.invariants else " (no invariants yet)"))
        matrix = run_tier(cfg, "quick", sc.name, only_kinds={"model", "tool"}, harness=h,
                          progress=_progress)
    data = matrix.to_json()
    report.print_matrix(data, out)
    return _gate(data, None)


def cmd_replay(args: argparse.Namespace) -> int:
    from agent_breakpoint.testing import find_cell, latest_matrix, run_cell

    cfg = load_config(args.config)
    matrix = json.loads(Path(args.matrix).read_text()) if args.matrix else latest_matrix(cfg)
    cell = find_cell(matrix, args.cell)
    _progress(f"replaying {cell.get('label') or cell['key']} ×{args.runs}")
    res = run_cell(cfg, cell, scenario=matrix["scenario"], runs=args.runs)
    for t in res.trials:
        out.print(f"[bold]{t.grade.label}[/]  {t.trial_dir}")
        for r in t.result.reasons:
            out.print(f"   {r}")
    before = Grade(cell["stats"]["worst"])
    out.print(f"Was {before.label}; now {res.worst.label}.")
    return 1 if res.worst in cfg.gate.fail_on else 0


def cmd_export(args: argparse.Namespace) -> int:
    from agent_breakpoint.testing import export_pytest, find_cell, latest_matrix

    cfg = load_config(args.config)
    matrix = latest_matrix(cfg)
    cell = find_cell(matrix, args.cell)
    path = export_pytest(args.config, cell, matrix["scenario"], Path(args.out),
                         [g.value for g in cfg.gate.fail_on])
    out.print(f"[green]wrote[/] {path}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data = json.loads((cfg.run_path / "latest.json").read_text())
    out.print_json(json.dumps(data["graph"]))
    return 0


def cmd_compat(args: argparse.Namespace) -> int:
    from agent_breakpoint.compat import run_compat

    cfg = load_config(args.config)
    res = run_compat(cfg, args.ref, args.scenario)
    if res.ok:
        out.print(f"[green]compatible[/]: checkpoints from {res.ref} load and resume with this tree")
        return 0
    out.print(f"[red]incompatible[/] with checkpoints from {res.ref}:")
    for p in res.problems:
        out.print(f"  - {p}")
    return 1


def cmd_bench(args: argparse.Namespace) -> int:
    from rich.table import Table

    from agent_breakpoint.bench import agentfault500

    if args.dataset != "agentfault-500":
        raise ValueError(f"unknown benchmark {args.dataset!r}; available: agentfault-500")
    data = Path(args.data) if args.data else None
    res = agentfault500.run(data, bootstrap=args.bootstrap)
    split = res["splits"][args.split]
    t = Table(title=f"AgentFault-500, {args.split} split: {split['trajectories']} trajectories, "
                    f"{split['groups']} baseline groups", header_style="bold")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_column("95% CI (grouped bootstrap)")
    for k, v in split["metrics"].items():
        ci = f"{v['ci95'][0]:.3f} to {v['ci95'][1]:.3f}" if v.get("ci95") else ""
        t.add_row(k, f"{v['value']:.3f}", ci)
    out.print(t)
    fam = Table(title="By fault family", header_style="bold")
    for col in ("family", "n", "auroc_trace", "rootcause_unpaired_top3", "rootcause_paired_exact"):
        fam.add_column(col, justify="right" if col != "family" else "left")
    for f, v in split["by_family"].items():
        fam.add_row(f, str(v["n"]), *(f"{v[k]:.2f}" for k in ("auroc_trace", "rootcause_unpaired_top3",
                                                             "rootcause_paired_exact")))
    out.print(fam)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from agent_breakpoint.mcp_server import serve

    serve(args.config)
    return 0


def cmd_faults(args: argparse.Namespace) -> int:
    from rich.table import Table

    from agent_breakpoint.faults import registry
    from agent_breakpoint.faults.taxonomy import family

    t = Table(title="Fault catalog", header_style="bold")
    for col in ("id", "class", "family", "description", "tests"):
        t.add_column(col)
    for f in sorted(registry().faults.values(), key=lambda f: f.id):
        t.add_row(f.id, f.fault_class.value, family(f.id), f.description, f.tests)
    out.print(t)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "mcp-stdio":  # internal: stdio MCP wrapper
        from agent_breakpoint.proxy.mcp_stdio_main import main as stdio_main

        return stdio_main(argv[1:])

    p = argparse.ArgumentParser(prog="agent-breakpoint",
                                description="Break an agent system on purpose; get a graded tolerance matrix.")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)

    def outputs(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json")
        sp.add_argument("--junit")
        sp.add_argument("--sarif")
        sp.add_argument("--markdown")
        sp.add_argument("--all", action="store_true", help="list every cell, not just failures")

    for tier in ("quick", "full"):
        sp = sub.add_parser(tier, help=f"run the {tier} tier")
        sp.set_defaults(func=cmd_tier, tier=tier)
        sp.add_argument("-s", "--scenario")
        sp.add_argument("--only", help="boundary kinds, e.g. model,tool")
        sp.add_argument("--cells", help="glob(s) on cell labels, e.g. '*stale_read*'")
        sp.add_argument("--no-incremental", action="store_true")
        sp.add_argument("--fail-on", help="grades that fail the run, e.g. failed_silently,unsafe")
        sp.add_argument("--diff", help="compare with a previous matrix.json")
        sp.add_argument("-q", "--quiet", action="store_true")
        outputs(sp)
    sp = sub.add_parser("run", help="run a named tier from the config")
    sp.add_argument("tier")
    sp.add_argument("-s", "--scenario")
    sp.add_argument("--only")
    sp.add_argument("--cells")
    sp.add_argument("--no-incremental", action="store_true")
    sp.add_argument("--fail-on")
    sp.add_argument("--diff")
    sp.add_argument("-q", "--quiet", action="store_true")
    outputs(sp)
    sp.set_defaults(func=cmd_tier)

    sp = sub.add_parser("init", help="detect the stack and write starter files")
    sp.add_argument("dir", nargs="?", default=".")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("record", help="record a baseline, discover failure points, first matrix")
    sp.add_argument("-s", "--scenario")
    sp.add_argument("--no-matrix", action="store_true")
    sp.set_defaults(func=cmd_record)

    sp = sub.add_parser("replay", help="rerun one cell exactly")
    sp.add_argument("cell")
    sp.add_argument("--runs", type=int, default=1)
    sp.add_argument("--matrix")
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("export", help="export a cell as a pytest test")
    sp.add_argument("cell")
    sp.add_argument("--out", default="tests")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("report", help="render a saved matrix")
    sp.add_argument("matrix", nargs="?")
    sp.add_argument("--fail-on")
    outputs(sp)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("diff", help="grade changes between two matrices")
    sp.add_argument("old")
    sp.add_argument("new")
    sp.add_argument("--markdown")
    sp.set_defaults(func=cmd_diff)

    sp = sub.add_parser("graph", help="print the failure-point graph of the last run")
    sp.set_defaults(func=cmd_graph)

    sp = sub.add_parser("compat", help="checkpoint compatibility test against a git ref")
    sp.add_argument("--ref")
    sp.add_argument("-s", "--scenario")
    sp.set_defaults(func=cmd_compat)

    sp = sub.add_parser("faults", help="list the fault catalog")
    sp.set_defaults(func=cmd_faults)

    sp = sub.add_parser("bench", help="benchmark the trace oracles on a labeled dataset")
    sp.add_argument("dataset", choices=["agentfault-500"])
    sp.add_argument("--data", help="directory with baseline.jsonl and injected.jsonl (default: download)")
    sp.add_argument("--split", default="held_out", choices=["held_out", "dev", "all"])
    sp.add_argument("--bootstrap", type=int, default=1000)
    sp.add_argument("--json")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("serve", help="MCP server for coding agents (config from disk only)")
    sp.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"[red]error:[/] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
