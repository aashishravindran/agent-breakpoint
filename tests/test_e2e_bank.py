"""End to end: the toy bank agent, its fake model and its stdio MCP server, through every proxy."""

import json
import subprocess
import sys

import pytest

from agent_breakpoint.grades import Grade
from agent_breakpoint.suite import run_tier
from agent_breakpoint.testing import export_pytest, find_cell, run_cell

EXPECTED = {
    # cell label prefix -> worst grade
    "model:llm × model.rate_limit": Grade.TOLERATED,
    "model:llm × model.timeout": Grade.TOLERATED,
    "model:llm × model.malformed_tool_call": Grade.TOLERATED,
    "model:llm × context.dropped_system_prompt": Grade.FAILED_SILENTLY,
    "model:llm × context.oversized_prompt": Grade.FAILED_LOUDLY,
    "model:llm × budget.quota_exhausted": Grade.FAILED_LOUDLY,
    "tool:bank.get_order × tool.wrong_result": Grade.FAILED_SILENTLY,
    "tool:bank.get_order × tool.injected_instructions": Grade.UNSAFE,
    "tool:bank.refund × tool.error_in_200": Grade.FAILED_SILENTLY,
    "tool:bank.refund × tool.is_error": Grade.UNSAFE,
    "tool:bank.refund × tool.timeout": Grade.TOLERATED,
    "tool:bank.refund × tool.removed_mid_session": Grade.FAILED_LOUDLY,
}


@pytest.fixture(scope="module")
def bank_matrix(tmp_path_factory, fake_llm):
    from agent_breakpoint.config import load_config
    from tests.conftest import _copy_example

    cfg = load_config(_copy_example("toy_bank", tmp_path_factory.mktemp("bank")))
    return cfg, run_tier(cfg, "quick", incremental=False).to_json()


def _worst(matrix, prefix):
    cells = [c for c in matrix["cells"] if (c.get("label") or "").startswith(prefix)]
    assert cells, f"no cell {prefix!r}; have {[c['label'] for c in matrix['cells']]}"
    return Grade.worst([Grade(c["stats"]["worst"]) for c in cells])


def test_quick_tier_grades(bank_matrix):
    _, m = bank_matrix
    assert m["baseline"]["grades"] == ["tolerated"]
    got = {k: _worst(m, k) for k in EXPECTED}
    assert got == EXPECTED
    assert not m["gate"]["passed"]
    assert m["totals"]["replay_hits"] > 0 and m["totals"]["replay_misses"] == 0


def test_discovery_found_every_boundary(bank_matrix):
    _, m = bank_matrix
    points = {p["boundary"]: p for p in m["graph"]["points"]}
    assert set(points) == {"model:llm", "tool:bank.get_order", "tool:bank.refund",
                           "state:scratchpad.write"}
    assert points["tool:bank.refund"]["destructive"] and not points["tool:bank.get_order"]["destructive"]


def test_incremental_rerun_reuses_cells(bank_matrix):
    cfg, first = bank_matrix
    again = run_tier(cfg, "quick").to_json()
    assert again["totals"]["reused"] == len(again["cells"]) == len(first["cells"])


def test_replay_reproduces_the_exact_fault(bank_matrix):
    cfg, m = bank_matrix
    cell = find_cell(m, "tool:bank.refund × tool.is_error")
    res = run_cell(cfg, cell, scenario="refund")
    assert res.worst == Grade.UNSAFE
    assert any("executed 2 times" in r for r in res.reasons())


def test_export_pytest_writes_a_runnable_test(bank_matrix, tmp_path):
    cfg, m = bank_matrix
    cell = find_cell(m, "tool:bank.refund × tool.timeout")
    path = export_pytest(cfg.root / "agent-breakpoint.yaml", cell, "refund", tmp_path,
                         ["failed_silently", "unsafe"])
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(path), "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_reports_render(bank_matrix):
    from agent_breakpoint import report

    _, m = bank_matrix
    sarif = report.sarif(m)
    levels = {r["level"] for r in sarif["runs"][0]["results"]}
    assert levels <= {"error", "warning", "note"} and "error" in levels
    assert "<failure" in report.junit(m)
    assert "| Unsafe |" in report.markdown(m)
    worse = dict(m, cells=[dict(c, stats=dict(c["stats"], worst="unsafe")) for c in m["cells"]])
    assert any(d["change"] == "worse" for d in report.diff(m, worse))


def test_full_tier_runs_combinations(bank_config):
    from agent_breakpoint.config import Tier

    bank_config.tiers["full"] = Tier(runs=1, max_cells=60, busiest=None, combinations=True,
                                     max_combinations=4)
    m = run_tier(bank_config, "full", incremental=False, only_kinds={"model", "tool"},
                 cells_filter=lambda c: c.fault_ids[0] in ("model.rate_limit", "tool.timeout",
                                                           "tool.partial_result")).to_json()
    combos = [c for c in m["cells"] if len(c["faults"]) > 1]
    assert combos, "expected fault combinations in the full tier"
    assert all(len({f["boundary"] for f in c["faults"]}) == 2 for c in combos)


def test_mock_tools_serve_recorded_responses(bank_config):
    from agent_breakpoint.runner import Harness

    with Harness(bank_config) as h:
        sc = bank_config.scenarios[0]
        base = h.run_trial(sc, None, mode="record", label="baseline")
        bank_config.mock_tools = True
        h.cfg.mock_tools = True
        mocked = h.run_trial(sc, None, label="mocked", baseline=base)
    calls = [e for e in mocked.events if e.op == "tools/call"]
    assert calls and all(e.status == "mocked" for e in calls)
    ledger = json.loads((mocked.trial_dir / "sandbox" / "ledger.json").read_text())
    assert ledger == []  # the real refund never ran
