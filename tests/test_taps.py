
import pytest

from agent_breakpoint import tap
from agent_breakpoint.controller import ENV_PLAN, ENV_TRIAL_DIR, FaultSpec, Plan
from agent_breakpoint.taps import reset_controller
from agent_breakpoint.taps.orchestration import SubagentFailure
from agent_breakpoint.trace import read_trace


def harness_env(monkeypatch, tmp_path, faults):
    monkeypatch.setenv(ENV_PLAN, Plan(faults=faults).to_env())
    monkeypatch.setenv(ENV_TRIAL_DIR, str(tmp_path))
    reset_controller()


def test_tap_is_inert_outside_the_harness():
    d = {}
    assert tap(d) is d
    def fn(x):
        return x

    assert tap(fn) is fn


def test_mapping_stale_read_and_lost_update(monkeypatch, tmp_path):
    harness_env(monkeypatch, tmp_path, [
        FaultSpec("state:board.read", "state.stale_read", 1),
        FaultSpec("state:board.write", "state.lost_update", 3),
    ])
    board = tap({}, name="board", shared=True)
    board["k"] = 1
    board["k"] = 2
    assert board["k"] == 1  # stale
    assert board["k"] == 2
    board["k"] = 3  # overwritten by a concurrent stale writer
    assert board["k"] == 2
    kinds = {e.boundary for e in read_trace(tmp_path)}
    assert kinds == {"state:board.read", "state:board.write"}


def test_orchestration_faults(monkeypatch, tmp_path):
    harness_env(monkeypatch, tmp_path, [
        FaultSpec("orchestration:researcher.call", "orchestration.subagent_fails", 1),
        FaultSpec("orchestration:researcher.call", "orchestration.cyclic_delegation", 2, sticky=True),
    ])
    researcher = tap(lambda q: f"answer to {q}", name="researcher")
    with pytest.raises(SubagentFailure):
        researcher("a")
    assert "hand this task back" in researcher("b")
    assert "hand this task back" in researcher("c")


def test_langgraph_checkpointer_lost_write(monkeypatch, tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class S(TypedDict):
        n: int

    harness_env(monkeypatch, tmp_path, [FaultSpec("state:cp.write", "state.lost_write", None)])
    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    graph = g.compile(checkpointer=tap(InMemorySaver(), name="cp"))
    cfg = {"configurable": {"thread_id": "t"}}
    assert graph.invoke({"n": 1}, cfg)["n"] == 2
    assert graph.get_state(cfg).values == {}  # nothing was kept
    assert any(e.faults == ["state.lost_write"] for e in read_trace(tmp_path))


def test_checkpointer_list_gets_read_faults(monkeypatch, tmp_path):
    """Code that reads state through list() (walking history) must see stale and drifted reads."""
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.memory import InMemorySaver

    harness_env(monkeypatch, tmp_path, [FaultSpec("state:cp.list", "state.stale_read", 1),
                                        FaultSpec("state:cp.list", "state.schema_drift", 2)])
    saver = tap(InMemorySaver(), name="cp")
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    for n in (1, 2):
        cp = empty_checkpoint()
        cp["channel_values"] = {"n": n}
        cp["channel_versions"] = {"n": n}
        saver.put(cfg, cp, {}, {"n": n})
    assert next(iter(saver.list(cfg))).checkpoint["channel_values"] == {"n": 1}  # newest hidden
    assert next(iter(saver.list(cfg))).checkpoint["channel_values"] == {"n_old": 2}
    assert next(iter(saver.list(cfg))).checkpoint["channel_values"] == {"n": 2}
