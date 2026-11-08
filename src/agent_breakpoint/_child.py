"""Runs inside the agent's process: ``python -m agent_breakpoint._child (run|seed) <trial_dir>``.

Imports only the stdlib and ``agent_breakpoint.contract`` before calling user code, so it works
inside the agent's own virtualenv as long as ``agent_breakpoint`` is on PYTHONPATH.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

from agent_breakpoint.contract import Outcome, Sandbox, Task


def _load(ref: str):
    module, _, attr = ref.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _call(fn, *args):
    result = fn(*args)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return result


def _write(trial: Path, name: str, data: dict) -> None:
    tmp = trial / f".{name}.tmp"
    tmp.write_text(json.dumps(data, default=str))
    tmp.replace(trial / name)


def _no_taps() -> None:
    """Seeding and invariant checks are the harness's own work: no faults, no trace."""
    for var in ("AGENT_BREAKPOINT_PLAN", "AGENT_BREAKPOINT_TRIAL_DIR"):
        os.environ.pop(var, None)


def main(argv: list[str]) -> int:
    mode, trial = argv[0], Path(argv[1])
    if mode in ("seed", "invariants", "call"):
        _no_taps()
    spec = json.loads((trial / "child.json").read_text())
    for p in reversed(spec.get("python_path", [])):
        if p not in sys.path:
            sys.path.insert(0, p)
    sandbox = Sandbox.from_json(spec["sandbox"])
    if mode == "seed":
        if spec.get("seed"):
            _call(_load(spec["seed"]), sandbox)
        return 0

    if mode == "invariants":
        outcome = Outcome.from_json(json.loads((trial / "final-outcome.json").read_text()))
        results = []
        for ref in spec.get("invariants", []):
            name = ref.rpartition(":")[2]
            try:
                value = _call(_load(ref), sandbox, outcome)
            except Exception as exc:  # a broken invariant counts as failed
                results.append({"name": name, "ok": False, "message": f"invariant raised {exc!r}"})
                continue
            if isinstance(value, tuple):
                results.append({"name": name, "ok": bool(value[0]), "message": str(value[1])})
            else:
                results.append({"name": name, "ok": bool(getattr(value, "ok", value)),
                                "message": str(getattr(value, "message", ""))})
        _write(trial, "invariants.json", {"results": results})
        return 0
    task = Task.from_json(spec["task"])
    if mode == "call":  # compat checks and other helpers: fn(sandbox, task) -> JSON-able
        result = _call(_load(spec["call"]), sandbox, task)
        _write(trial, "call-result.json", result if result is not None else [])
        return 0
    session = os.environ.get("AGENT_BREAKPOINT_SESSION", "1")
    out_name = f"outcome-s{session}.json"
    try:
        result = _call(_load(spec["entrypoint"]), task, sandbox)
        if isinstance(result, Outcome):
            outcome = result
        elif isinstance(result, dict):
            outcome = Outcome.from_json(result)
        else:
            outcome = Outcome(answer="" if result is None else str(result))
    except KeyboardInterrupt:
        outcome = Outcome(error="interrupted (SIGINT)")
    except BaseException as exc:  # the agent crashed: a loud failure, with the traceback
        outcome = Outcome(error=f"{type(exc).__name__}: {exc}",
                          details={"traceback": traceback.format_exc()[-4000:]})
    _write(trial, out_name, outcome.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
