"""The four front doors: CLI, pytest plugin, MCP server (and the Action, which wraps the CLI)."""

import asyncio
import json
import os
import subprocess
import sys

pytest_plugins = ["pytester"]


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "agent_breakpoint.cli", *args], cwd=cwd,
                          capture_output=True, text=True, env={**os.environ, "COLUMNS": "160"})


def test_init_detects_stack_and_writes_files(tmp_path):
    (tmp_path / "requirements.txt").write_text("langgraph>=0.2\nanthropic\nmcp\n")
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"github": {"command": "gh-mcp"}}}))
    (tmp_path / "app.py").write_text("graph = b.compile(checkpointer=saver)\n")
    r = cli("init", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "LangGraph" in r.stdout and "github" in r.stdout
    cfg = (tmp_path / "agent-breakpoint.yaml").read_text()
    assert "mcp_config:" in cfg and "entrypoint: agent_breakpoint_entry:run" in cfg
    assert (tmp_path / "agent_breakpoint_entry.py").exists()
    assert ".agent-breakpoint/" in (tmp_path / ".gitignore").read_text()
    from agent_breakpoint.config import load_config

    load_config(tmp_path / "agent-breakpoint.yaml")  # valid


def test_record_suggests_invariants(bank_config):
    r = cli("-c", str(bank_config.root / "agent-breakpoint.yaml"), "record", "--no-matrix")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tool:bank.refund" in r.stdout and "(destructive)" in r.stdout
    sugg = (bank_config.run_path / "suggested_invariants.py").read_text()
    assert "def ledger_json_grew_by_1" in sugg
    ns: dict = {}
    exec(compile(sugg, "s", "exec"), ns)  # suggestions are valid Python


def test_faults_command_lists_catalog():
    r = cli("faults")
    assert r.returncode == 0 and "tool.injected_instructions" in r.stdout


def test_mcp_server_front_door(bank_config):
    from mcp.client import Client

    from agent_breakpoint.mcp_server import build_server

    server = build_server(str(bank_config.root / "agent-breakpoint.yaml"))

    async def go():
        async with Client(server) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            # no tool can be pointed at a different target
            for t in tools.values():
                props = set((t.input_schema or {}).get("properties", {}))
                assert not props & {"url", "command", "upstream", "config", "target", "budget"}
            scen = await c.call_tool("list_scenarios", {})
            run = await c.call_tool("run_quick", {})
            bad = await c.call_tool("run_quick", {"scenario": "http://evil"})
            return tools, scen, run, bad

    tools, scen, run, bad = asyncio.run(go())
    assert {"run_quick", "get_matrix", "explain_cell", "replay_cell"} <= set(tools)
    assert "refund" in scen.content[0].text
    data = json.loads(run.content[0].text)
    assert data["failing"] and "cells:" in data["summary"]
    assert bad.is_error


def test_pytest_plugin(pytester, bank_config):
    cfg = bank_config.root / "agent-breakpoint.yaml"
    pytester.makepyfile("""
        def test_gate(agent_breakpoint):
            m = agent_breakpoint.quick(only_kinds={"model"})
            assert agent_breakpoint.cells_with(m, "failed_silently")
            agent_breakpoint.assert_no(m, "unsafe")

        def test_gate_fails(agent_breakpoint):
            m = agent_breakpoint.quick(only_kinds={"model"})
            agent_breakpoint.assert_no(m, "failed_silently")
    """)
    res = pytester.runpytest_subprocess(f"--agent-breakpoint-config={cfg}", "-q")
    res.assert_outcomes(passed=1, failed=1)
    res.stdout.fnmatch_lines(["*context.dropped_system_prompt*"])
