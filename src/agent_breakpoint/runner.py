"""The harness: start proxies, run trials in fresh sandboxes, grade them.

A trial is one execution of a scenario (all its sessions) under one plan. The agent runs
unmodified in a child process; every edge it crosses goes through a proxy or tap that records
the call and asks the controller whether to interfere.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import agent_breakpoint
from agent_breakpoint.config import Config, Scenario
from agent_breakpoint.contract import Outcome, Sandbox, Task
from agent_breakpoint.controller import (
    ENV_CANARIES,
    ENV_PLAN,
    ENV_RESTART,
    ENV_SANDBOX,
    ENV_SESSION,
    ENV_TRIAL_DIR,
    Controller,
    Plan,
)
from agent_breakpoint.faults.base import registry
from agent_breakpoint.grades import Grade
from agent_breakpoint.oracles import (
    GradeResult,
    TrialView,
    check_invariants,
    grade_trial,
    infer_claim,
    load_safety_packs,
)
from agent_breakpoint.oracles.state import load_callable
from agent_breakpoint.planner import Cell
from agent_breakpoint.proxy.dependency import HttpDependencyProxy, TcpDependencyProxy
from agent_breakpoint.proxy.mcp import ENV_MCP_CACHE, HttpMcpProxy, wrap_stdio_command
from agent_breakpoint.proxy.model import ModelProxy
from agent_breakpoint.proxy.server import BackgroundServer
from agent_breakpoint.taps import fs as fs_tap
from agent_breakpoint.trace import Event, read_trace

SRC_ROOT = str(Path(agent_breakpoint.__file__).resolve().parent.parent)


@dataclass
class TrialResult:
    trial_id: str
    trial_dir: Path
    cell_key: str | None
    grade: Grade
    result: GradeResult
    outcome: Outcome
    events: list[Event]
    usd: float = 0.0
    tokens: int = 0
    duration_s: float = 0.0
    restarts: int = 0
    budget_blown: str | None = None
    replay_hits: int = 0
    replay_misses: int = 0
    root_cause: dict[str, Any] | None = None

    def view(self, settings: dict[str, Any]) -> TrialView:
        return TrialView(self.events, self.outcome, self.budget_blown, settings)

    def to_json(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id, "trial_dir": str(self.trial_dir), "cell": self.cell_key,
            "grade": self.grade.value, "claim": self.result.claim, "state": self.result.state,
            "reasons": self.result.reasons,
            "findings": [{"pack": f.pack, "message": f.message, "evidence": f.evidence}
                         for f in self.result.findings],
            "judge_score": self.result.judge_score,
            "answer": (self.outcome.answer or "")[:2000], "error": self.outcome.error,
            "usd": round(self.usd, 6), "tokens": self.tokens,
            "duration_s": round(self.duration_s, 3), "restarts": self.restarts,
            "budget_blown": self.budget_blown, "replay_hits": self.replay_hits,
            "replay_misses": self.replay_misses, "root_cause": self.root_cause,
        }


class Harness:
    def __init__(self, cfg: Config, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.run_dir = cfg.run_path / "runs" / self.run_id
        self.cache_dir = cfg.run_path / "cache"
        self.model_proxies: list[tuple[ModelProxy, BackgroundServer]] = []
        self.mcp_http: list[tuple[HttpMcpProxy, BackgroundServer]] = []
        self.deps_http: list[tuple[HttpDependencyProxy, BackgroundServer]] = []
        self.deps_tcp: list[TcpDependencyProxy] = []
        self.registry = registry(cfg.packs)
        self.safety = load_safety_packs(cfg.safety)
        self._child: subprocess.Popen | None = None
        self._child_lock = threading.Lock()
        self._killed_with: int | None = None
        self.base_env: dict[str, str] = {}
        self.destructive: list[str] = sorted({d for m in cfg.mcp for d in m.destructive})

    # --- lifecycle ------------------------------------------------------------------------------

    def __enter__(self) -> Harness:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        env: dict[str, str] = {}
        for mb in self.cfg.models:
            proxy = ModelProxy(mb, self.cfg.budgets, self.cfg.pricing)
            proxy.on_kill = self._kill_child
            server = BackgroundServer(proxy).start()
            self.model_proxies.append((proxy, server))
            for var in mb.env:
                env[var] = server.url + mb.env_suffix
        for m in self.cfg.mcp:
            if m.transport == "http":
                assert m.url, f"mcp server {m.name} needs url"
                hp = HttpMcpProxy(m.name, m.url, m.destructive)
                server = BackgroundServer(hp).start()
                self.mcp_http.append((hp, server))
                if m.env:
                    path = "/" + m.url.split("/", 3)[3] if m.url.count("/") >= 3 else ""
                    env[m.env] = server.url + path
            else:
                wrapped = wrap_stdio_command(m.name, m.command, m.destructive)
                if m.env:
                    env[m.env] = json.dumps(wrapped)
        for d in self.cfg.dependencies:
            if d.kind == "http":
                dp = HttpDependencyProxy(d)
                server = BackgroundServer(dp, port=d.listen_port or 0).start()
                self.deps_http.append((dp, server))
                if d.env:
                    env[d.env] = server.url
            else:
                tp = TcpDependencyProxy(d).start()
                self.deps_tcp.append(tp)
                if d.env:
                    env[d.env] = tp.url
        if self.cfg.mcp_config:
            rewritten = self._rewrite_mcp_config()
            if self.cfg.mcp_config.env:
                env[self.cfg.mcp_config.env] = str(rewritten)
        env.update(self.cfg.env)
        self.base_env = env

    def stop(self) -> None:
        for _, s in self.model_proxies + self.mcp_http + self.deps_http:  # type: ignore[operator]
            s.stop()
        for t in self.deps_tcp:
            t.stop()

    def _rewrite_mcp_config(self) -> Path:
        assert self.cfg.mcp_config is not None
        src = (self.cfg.root / self.cfg.mcp_config.path).resolve()
        data = json.loads(src.read_text())
        servers = data.get("mcpServers", {})
        known = {m.name: m for m in self.cfg.mcp}
        for name, spec in servers.items():
            destructive = known[name].destructive if name in known else []
            if spec.get("command"):
                cmd = [spec["command"], *spec.get("args", [])]
                wrapped = wrap_stdio_command(name, cmd, destructive)
                spec["command"], spec["args"] = wrapped[0], wrapped[1:]
            elif spec.get("url"):
                hp = HttpMcpProxy(name, spec["url"], destructive)
                server = BackgroundServer(hp).start()
                self.mcp_http.append((hp, server))
                path = "/" + spec["url"].split("/", 3)[3] if spec["url"].count("/") >= 3 else ""
                spec["url"] = server.url + path
        out = self.run_dir / "mcp.json"
        out.write_text(json.dumps(data, indent=2))
        return out

    # --- trials ---------------------------------------------------------------------------------

    def _kill_child(self, sig: int) -> None:
        with self._child_lock:
            if self._child and self._child.poll() is None:
                self._killed_with = sig
                try:
                    os.killpg(self._child.pid, sig)
                except (ProcessLookupError, PermissionError):
                    self._child.send_signal(sig)

    def _child_env(self, trial_dir: Path, sandbox: Sandbox, plan: Plan, session: int,
                   restart: bool, scenario: Scenario) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.base_env)
        pp = [str((self.cfg.root / p).resolve()) for p in self.cfg.python_path] + [SRC_ROOT]
        if env.get("PYTHONPATH"):
            pp.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pp)
        env[ENV_PLAN] = plan.to_env()
        env[ENV_TRIAL_DIR] = str(trial_dir)
        env[ENV_SANDBOX] = str(sandbox.root)
        env[ENV_SESSION] = str(session)
        env[ENV_MCP_CACHE] = str(self.cache_dir / f"mcp-{scenario.name}.jsonl")
        env[ENV_CANARIES] = json.dumps([c.value for c in self.cfg.canaries])
        env["PYTHONUNBUFFERED"] = "1"
        if restart:
            env[ENV_RESTART] = "1"
        return env

    def _python(self) -> str:
        return self.cfg.python_exe or sys.executable

    def _launch(self, mode: str, trial_dir: Path, env: dict[str, str], timeout: float) -> tuple[int, bool]:
        log = (trial_dir / "agent.log").open("ab")
        cmd = [self._python(), "-m", "agent_breakpoint._child", mode, str(trial_dir)]
        with self._child_lock:
            self._killed_with = None
            self._child = subprocess.Popen(cmd, cwd=self.cfg.work_dir, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        child = self._child
        timed_out = False
        try:
            rc = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_child(signal.SIGKILL)
            rc = child.wait()
        finally:
            log.close()
            with self._child_lock:
                self._child = None
        return rc, timed_out

    def run_trial(self, scenario: Scenario, cell: Cell | None, *, mode: str = "inject",
                  label: str = "trial", baseline: TrialResult | None = None,
                  replay: bool = True) -> TrialResult:
        trial_id = f"{label}-{uuid.uuid4().hex[:8]}"
        trial_dir = self.run_dir / "trials" / trial_id
        sandbox = Sandbox(trial_dir / "sandbox")
        sandbox.root.mkdir(parents=True, exist_ok=True)
        model_cache = self.cache_dir / f"model-{scenario.name}.jsonl"
        if mode == "record" and model_cache.exists():
            model_cache.unlink()
        mcp_cache = self.cache_dir / f"mcp-{scenario.name}.jsonl"
        if mode == "record" and mcp_cache.exists():
            mcp_cache.unlink()
        plan = Plan(
            seed=self.cfg.seed, faults=list(cell.faults) if cell else [], mode=mode,
            mock_tools=self.cfg.mock_tools, destructive_tools=self.destructive,
            replay_cache=str(model_cache) if (replay and mode != "record") else None,
            replay_until=dict(cell.replay_until) if (cell and replay) else {},
            packs=self.cfg.packs,
        )
        (trial_dir / "plan.json").write_text(plan.to_env())
        harness_ctl = Controller(plan, trial_dir, "harness", self.registry)
        for proxy, _ in self.model_proxies:
            proxy.begin_trial(harness_ctl, model_cache if mode == "record" else None)
        for hp, _ in self.mcp_http:
            hp.begin_trial(harness_ctl, mcp_cache)
        for dp, _ in self.deps_http:
            dp.begin_trial(harness_ctl)
        for tp in self.deps_tcp:
            tp.begin_trial(harness_ctl)

        started = time.time()
        sessions = max(1, scenario.sessions, 1 + len(scenario.later_tasks))
        child_spec = {
            "entrypoint": self.cfg.entrypoint, "seed": scenario.seed,
            "python_path": [str((self.cfg.root / p).resolve()) for p in self.cfg.python_path],
            "sandbox": sandbox.to_json(),
        }
        (trial_dir / "child.json").write_text(json.dumps({**child_spec, "task": scenario.task}))
        env = self._child_env(trial_dir, sandbox, plan, 1, False, scenario)
        if scenario.seed:
            self._launch("seed", trial_dir, env, timeout=120)
        pre_snapshot = trial_dir / "pre-session-snapshot"
        fs_tap.snapshot(sandbox.root, pre_snapshot)

        outcome = Outcome()
        restarts = 0
        timed_out_any = False
        remaining = self.cfg.budgets.wall_clock_s
        for session in range(1, sessions + 1):
            task = scenario.task if session == 1 else scenario.later_tasks[session - 2]
            task = {"thread_id": f"abk-{scenario.name}", **task, "session": session}
            (trial_dir / "child.json").write_text(json.dumps({**child_spec, "task": task}))
            harness_ctl.session = session
            if session > 1 and cell:
                self._between_sessions(cell, session, trial_dir, sandbox, pre_snapshot)
            restart = False
            while True:
                env = self._child_env(trial_dir, sandbox, plan, session, restart, scenario)
                t0 = time.time()
                rc, timed_out = self._launch("run", trial_dir, env, timeout=max(1.0, remaining))
                remaining -= time.time() - t0
                timed_out_any |= timed_out
                crashed = (rc == -signal.SIGKILL and not timed_out
                           and self._killed_with in (None, signal.SIGKILL)
                           and not (trial_dir / f"outcome-s{session}.json").exists())
                if crashed and restarts < 3 and remaining > 0:
                    restarts += 1
                    restart = True  # resume with the same thread id: exactly-once is on trial
                    continue
                break
            out_file = trial_dir / f"outcome-s{session}.json"
            if out_file.exists():
                outcome = Outcome.from_json(json.loads(out_file.read_text()))
            elif timed_out:
                outcome = Outcome(error=f"wall clock budget exceeded ({self.cfg.budgets.wall_clock_s}s)")
            else:
                outcome = Outcome(error=f"agent process exited with code {rc} and no outcome")
            if timed_out or remaining <= 0:
                break

        events = read_trace(trial_dir)
        blown = None
        usd = tokens = 0
        for proxy, _ in self.model_proxies:
            usd += proxy.usage.usd
            tokens += proxy.usage.tokens_in + proxy.usage.tokens_out
            blown = blown or proxy.usage.blown
        if timed_out_any:
            blown = blown or f"wall_clock_s={self.cfg.budgets.wall_clock_s}"
        result = self._grade(scenario, sandbox, outcome, events, blown, baseline, trial_dir)
        tr = TrialResult(
            trial_id, trial_dir, cell.key if cell else None, result.grade, result, outcome,
            events, usd=usd, tokens=tokens, duration_s=time.time() - started, restarts=restarts,
            budget_blown=blown,
            replay_hits=sum(e.status == "replayed" for e in events),
            replay_misses=sum(bool(e.detail.get("replay_miss")) for e in events),
        )
        if tr.grade != Grade.TOLERATED and cell is not None:
            from agent_breakpoint.rootcause import explain

            tr.root_cause = explain(events, baseline.events if baseline else None)
        (trial_dir / "result.json").write_text(json.dumps(tr.to_json(), indent=2))
        return tr

    def _invariants_in_agent(self, scenario: Scenario, sandbox: Sandbox, outcome: Outcome,
                             trial_dir: Path):
        (trial_dir / "final-outcome.json").write_text(json.dumps(outcome.to_json(), default=str))
        spec = json.loads((trial_dir / "child.json").read_text())
        spec["invariants"] = scenario.invariants
        (trial_dir / "child.json").write_text(json.dumps(spec))
        env = self._child_env(trial_dir, sandbox, Plan(), 1, False, scenario)
        self._launch("invariants", trial_dir, env, timeout=120)
        out = trial_dir / "invariants.json"
        if not out.exists():
            return _verdict([{"name": "invariants", "ok": False,
                              "message": "invariant check crashed; see agent.log"}])
        return _verdict(json.loads(out.read_text())["results"])

    def _between_sessions(self, cell: Cell, session: int, trial_dir: Path, sandbox: Sandbox,
                          pre_snapshot: Path) -> None:
        """Episode faults on tapped directories, applied after session ``session - 1``."""
        dirs = {e.boundary: Path(e.detail["path"]) for e in read_trace(trial_dir)
                if e.op == "register" and "path" in e.detail}
        for spec in cell.faults:
            if spec.session != session or spec.boundary not in dirs:
                continue
            d = dirs[spec.boundary]
            ev = Event(boundary=spec.boundary, kind="state", call_index=session, op="between",
                       faults=[spec.fault], session=session)
            if spec.fault == "state.lost_write":
                try:
                    rel = d.resolve().relative_to(sandbox.root.resolve())
                    if (pre_snapshot / rel).exists():
                        fs_tap.rollback(d, pre_snapshot / rel)
                    else:
                        shutil.rmtree(d)
                        d.mkdir(parents=True)
                except ValueError:
                    shutil.rmtree(d)
                    d.mkdir(parents=True)
            elif spec.fault == "state.memory_poisoning":
                ev.canary = fs_tap.plant(d, uuid.uuid5(uuid.NAMESPACE_OID, cell.key).hex[:12])
            elif spec.fault == "state.torn_checkpoint":
                ev.detail["files"] = fs_tap.corrupt(d)
            elif spec.fault == "state.schema_drift":
                ev.detail["files"] = fs_tap.drift_json(d)
            Controller(Plan(), trial_dir, "episode").record(ev)

    def _grade(self, scenario: Scenario, sandbox: Sandbox, outcome: Outcome, events: list[Event],
               blown: str | None, baseline: TrialResult | None, trial_dir: Path) -> GradeResult:
        pp = [str((self.cfg.root / p).resolve()) for p in self.cfg.python_path]
        if self.cfg.invariants_in_agent and scenario.invariants:
            state = self._invariants_in_agent(scenario, sandbox, outcome, trial_dir)
        else:
            state = check_invariants(scenario.invariants, sandbox, outcome, pp)
        claim = infer_claim(outcome)
        settings = {"foreign_canaries": self.cfg.foreign_canaries(scenario),
                    "max_delegations": self.cfg.max_delegations}
        view = TrialView(events, outcome, blown, settings)
        base_view = baseline.view(settings) if baseline else None
        findings = []
        for pack in self.safety:
            findings += pack.check(view, base_view)
        judge = None
        if self.cfg.llm_judge:
            try:
                judge = float(load_callable(self.cfg.llm_judge, pp)(scenario.task, outcome))
            except Exception:
                judge = None
        return grade_trial(state, claim, findings, judge)


def _verdict(results: list[dict[str, Any]]):
    from agent_breakpoint.oracles.state import InvariantResult, StateVerdict

    rs = [InvariantResult(r["ok"], r.get("message", ""), r["name"]) for r in results]
    passed = sum(r.ok for r in rs)
    status = "ok" if passed == len(rs) else "failed" if passed == 0 else "partial"
    return StateVerdict(status, rs)


def task_of(scenario: Scenario) -> Task:
    return Task.from_json({"thread_id": f"abk-{scenario.name}", **scenario.task})
