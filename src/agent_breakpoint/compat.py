"""Checkpoint compatibility test, no LLM needed.

Make real checkpoints with the code on a git ref (default ``main``), then load them with the
working tree's code and resume one step. A renamed state field that would break in-flight
threads fails here in seconds.

Config::

    compat:
      ref: main
      check: myapp.compat:check     # check(sandbox, task) -> list[str] of problems

Checkpoints are produced by running the scenario's first session with the ref's code. Model
calls are served from the recorded cache (``record`` first), so the old run costs nothing when
prompts are unchanged; misses go live.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_breakpoint.config import Config
from agent_breakpoint.planner import Cell
from agent_breakpoint.runner import Harness


@dataclass
class CompatResult:
    ref: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    sandbox: str = ""


def langgraph_check(graph: Any, config: dict[str, Any]) -> list[str]:
    """Load a thread's latest checkpoint with the current graph and compare its channels with the
    current state schema. Returns problems; empty means compatible."""
    problems = []
    try:
        snap = graph.get_state(config)
    except Exception as exc:
        return [f"checkpoint failed to load: {exc!r}"]
    values = dict(snap.values or {})
    # get_state() drops channels the new code doesn't define; the raw checkpoint keeps them.
    saver = getattr(graph, "checkpointer", None)
    if saver is not None and hasattr(saver, "get_tuple"):
        tup = saver.get_tuple(config)
        if tup is not None:
            raw = tup.checkpoint.get("channel_values") or {}
            values.update({k: v for k, v in raw.items() if not k.startswith(("__", "branch:"))})
    channels = {k for k in getattr(graph, "channels", {}) if not k.startswith("__")}
    if channels:
        unknown = sorted(set(values) - channels)
        if unknown:
            problems.append(f"checkpoint has fields the new code doesn't know: {unknown}")
    schemas = getattr(getattr(graph, "builder", None), "schemas", None) or {}
    known: set[str] = set()
    required: set[str] = set()
    for s in schemas:
        known |= set(getattr(s, "__annotations__", {}))
        required |= set(getattr(s, "__required_keys__", ()))
    if known and not channels:
        unknown = sorted(set(values) - known)
        if unknown:
            problems.append(f"checkpoint has fields the new code doesn't know: {unknown}")
    missing = sorted(required - set(values)) if values else []
    if missing:
        problems.append(f"checkpoint lacks fields the new code requires: {missing}")
    for node in snap.next or ():
        if node not in getattr(graph, "nodes", {}):
            problems.append(f"checkpoint resumes at node {node!r}, which no longer exists")
    return problems


def _worktree(root: Path, ref: str) -> Path:
    top = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip())
    dest = Path(tempfile.mkdtemp(prefix="abk-compat-"))
    shutil.rmtree(dest)
    subprocess.check_call(["git", "worktree", "add", "--detach", str(dest), ref], cwd=top,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dest / root.resolve().relative_to(top)


def run_compat(cfg: Config, ref: str | None = None, scenario: str | None = None) -> CompatResult:
    ref = ref or cfg.compat.get("ref", "main")
    check = cfg.compat.get("check")
    if not check:
        raise ValueError("set compat.check (module:function(sandbox, task) -> list[str])")
    sc = cfg.scenario(scenario)
    old_root = _worktree(cfg.root, ref)
    try:
        old_cfg = cfg.model_copy(deep=True)
        old_cfg.root = old_root
        old_cfg.run_dir = str(cfg.run_path)  # share the model cache
        with Harness(old_cfg) as h:
            replay_all = Cell([], {f"model:{m.name}": 10**9 for m in cfg.models})
            trial = h.run_trial(sc, replay_all, label="compat-old")
        sandbox = trial.trial_dir / "sandbox"
        task = {"thread_id": f"abk-{sc.name}", **sc.task, "session": 2}
        spec = {"entrypoint": cfg.entrypoint, "call": check,
                "python_path": [str((cfg.root / p).resolve()) for p in cfg.python_path],
                "sandbox": {"root": str(sandbox), "env": {}}, "task": task}
        trial_dir = trial.trial_dir
        (trial_dir / "child.json").write_text(json.dumps(spec))
        with Harness(cfg) as h:
            env = h._child_env(trial_dir, _sandbox(sandbox), _plan(), 2, False, sc)
            rc, _ = h._launch("call", trial_dir, env, timeout=120)
        out = trial_dir / "call-result.json"
        if not out.exists():
            log = (trial_dir / "agent.log").read_text()[-2000:]
            return CompatResult(ref, False, [f"check crashed (exit {rc}): {log}"], str(sandbox))
        problems = json.loads(out.read_text())
        return CompatResult(ref, not problems, problems, str(sandbox))
    finally:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=cfg.root, text=True).strip()
        subprocess.call(["git", "worktree", "remove", "--force", str(old_root)], cwd=top,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _sandbox(root: Path):
    from agent_breakpoint.contract import Sandbox

    return Sandbox(root)


def _plan():
    from agent_breakpoint.controller import Plan

    return Plan(mode="record")
