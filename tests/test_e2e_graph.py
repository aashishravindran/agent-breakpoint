"""State taps end to end: crashes between side effect and checkpoint, episodes, compat."""

import shutil
import subprocess

from agent_breakpoint.grades import Grade
from agent_breakpoint.suite import run_tier


def _by_fault(m, fault):
    return [c for c in m["cells"] if c["fault_ids"] == [fault]]


def test_crash_between_side_effect_and_checkpoint_double_pays(graph_config):
    m = run_tier(graph_config, "quick", "refund", incremental=False).to_json()
    crash = _by_fault(m, "state.crash_before_checkpoint")
    assert len(crash) >= 4  # swept across write points
    silent = [c for c in crash if c["stats"]["worst"] == Grade.FAILED_SILENTLY.value]
    assert silent and "2 payments" in silent[0]["trials"][0]["reasons"][-1]
    assert all(t["restarts"] == 1 for c in crash for t in c["trials"])
    assert _by_fault(m, "state.stale_read")


def test_idempotent_node_tolerates_every_state_fault(graph_config):
    m = run_tier(graph_config, "quick", "refund-idempotent", incremental=False).to_json()
    assert m["gate"]["passed"], [c["label"] for c in m["cells"] if c["stats"]["worst"] != "tolerated"]


def test_episode_memory_faults(graph_config):
    m = run_tier(graph_config, "quick", "episode", incremental=False).to_json()
    dir_cells = {c["fault_ids"][0]: Grade(c["stats"]["worst"]) for c in m["cells"]
                 if c["boundary"] == "state:memory.dir"}
    assert dir_cells["state.memory_poisoning"] == Grade.UNSAFE
    assert dir_cells["state.lost_write"] == Grade.DEGRADED
    assert dir_cells["state.schema_drift"] == Grade.DEGRADED


def test_checkpoint_compat_detects_renamed_field(graph_config, tmp_path):
    from agent_breakpoint.compat import run_compat
    from agent_breakpoint.config import load_config
    from tests.conftest import ROOT

    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / "toy_graph", repo / "examples" / "toy_graph")
    (repo / "examples" / "__init__.py").write_text("")
    cfgfile = repo / "examples" / "toy_graph" / "agent-breakpoint.yaml"
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=repo, check=True)

    ok = run_compat(load_config(cfgfile), "HEAD", "refund")
    assert ok.ok, ok.problems

    graph_py = repo / "examples" / "toy_graph" / "graph.py"
    graph_py.write_text(graph_py.read_text().replace("    paid: bool\n", "    settled: bool\n")
                        .replace('{"paid": True}', '{"settled": True}')
                        .replace('state.get("paid")', 'state.get("settled")'))
    bad = run_compat(load_config(cfgfile), "HEAD", "refund")
    assert not bad.ok and any("paid" in p for p in bad.problems)
