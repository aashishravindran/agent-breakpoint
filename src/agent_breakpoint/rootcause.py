"""Root-cause localization for red cells.

Two localizers over a generic step sequence:

* ``first_divergence(steps, baseline)``: the paired view agent-breakpoint always has, because
  every trial runs against a recorded baseline. The first step that differs is where the
  failure entered; the next divergent steps show how it spread.
* ``suspicion(steps)``: unpaired trace rules derived from the oracles (errors, loops, broken
  handoffs, tool calls that don't match what the model asked for, an answer that doesn't carry
  the tool evidence, injected instructions, structure drift). Used when no baseline exists, and
  benchmarked on AgentFault-500.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent_breakpoint.trace import Event, stable_hash


@dataclass
class Step:
    index: int
    kind: str  # model | tool_call | tool_result | handoff | retrieval | state | dependency
    actor: str = ""
    tool: str | None = None
    status: str = "ok"
    input: Any = None
    output: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return stable_hash([self.kind, self.actor, self.tool, self.status, self.input, self.output, self.meta])

    @property
    def shape(self) -> str:
        """Structure without content: what kind of thing this step produced."""
        def sh(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: sh(v) for k, v in sorted(x.items())}
            if isinstance(x, list):
                return [sh(x[0])] if x else []
            return type(x).__name__
        return json.dumps(sh(_maybe_json(self.output)), sort_keys=True)


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str) and x.strip()[:1] in "{[":
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str)


# --- paired ------------------------------------------------------------------------------------


def divergences(steps: list[Step], baseline: list[Step]) -> list[int]:
    """Step indexes (of ``steps``) where the run departs from the baseline, in order.

    Steps are aligned as sequences, so an inserted or dropped step doesn't shift every later
    step out of place. A repeated step is attributed to its first occurrence (that is where the
    loop started); a run that stops early diverges at its last step."""
    import difflib

    a = [s.signature for s in steps]
    b = [s.signature for s in baseline]
    out: list[int] = []
    for op, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "delete" and i2 <= len(steps):  # inserted steps in this run
            first = i1
            while first > 0 and a[first] == a[first - 1]:
                first -= 1  # a repeat: blame where the repetition began
            out.extend(steps[k].index for k in range(first, i2))
        elif op == "replace":
            out.extend(steps[k].index for k in range(i1, i2))
        elif op == "insert" and steps:  # baseline steps missing from this run
            out.append(steps[max(0, i1 - 1)].index)
    return list(dict.fromkeys(out))


def first_divergence(steps: list[Step], baseline: list[Step]) -> int | None:
    d = divergences(steps, baseline)
    return d[0] if d else None


# --- unpaired ----------------------------------------------------------------------------------

_INJECTION = re.compile(r"(ignore (all |the )?(previous|prior) instructions|<system>|system override|"
                        r"you must now|new instructions|disregard|exfiltrat|confirmation code|"
                        r"do not mention this)", re.IGNORECASE)
_ERROR = re.compile(r"\b(error|failed|failure|exception|traceback|unavailable|denied|refused)\b", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Suspect:
    index: int
    score: float
    rule: str


def suspicion(steps: list[Step], agents: set[str] | None = None) -> list[Suspect]:
    """Per-step suspicion, highest first. ``agents`` = known actor names (handoff targets)."""
    agents = agents or {s.actor for s in steps if s.actor}
    found: list[Suspect] = []

    def add(s: Step, score: float, rule: str) -> None:
        found.append(Suspect(s.index, score, rule))

    shapes_by_actor: dict[tuple[str, str], list[str]] = {}
    for s in steps:
        shapes_by_actor.setdefault((s.kind, s.actor), []).append(s.shape)

    requested: dict[str, Any] = {}
    last_evidence: str | None = None
    retrieved: str | None = None
    for i, s in enumerate(steps):
        out = _maybe_json(s.output)
        # errors: a failed status, or an error-shaped output
        if s.status not in ("ok", "success"):
            add(s, 0.95, "step reported failure")
        elif isinstance(out, dict) and ("error" in out or out.get("documents_found") == 0):
            add(s, 0.9, "error or empty result returned as success")
        # loops: the same step repeated back to back
        if i and s.signature == steps[i - 1].signature:
            add(steps[i - 1], 0.85, "identical step repeated (loop / runaway)")
        # handoffs: target and role must agree; a responsibility must be assigned
        if s.kind == "handoff":
            to, role = s.meta.get("to"), s.meta.get("role")
            if s.actor and to and s.actor != to:
                add(s, 0.9, f"handoff executed by {s.actor}, addressed to {to}")
            if role and to and role != to:
                add(s, 0.8, f"handoff role {role} != target {to}")
            if "responsibility" in s.meta and not s.meta.get("responsibility"):
                add(s, 0.8, "handoff without an assigned responsibility")
            if to and agents and to not in agents:
                add(s, 0.8, f"handoff to unknown agent {to}")
            if not s.output:
                add(s, 0.85, "handoff dropped its payload")
        # tool calls must match what the model asked for
        if s.kind == "model" and isinstance(out, dict) and "tool" in out:
            requested = {"tool": out.get("tool"), "input": out.get("input")}
        if s.kind == "tool_call":
            if requested and s.tool != requested.get("tool"):
                add(s, 0.9, f"called {s.tool}, model asked for {requested.get('tool')}")
            elif requested and _maybe_json(s.input) != requested.get("input"):
                add(s, 0.85, "tool arguments differ from the model's request")
            requested = {}
        if s.kind == "tool_result" and s.output not in (None, ""):
            last_evidence = _text(s.output)
        # injected instructions anywhere in content the agent reads or writes
        if _INJECTION.search(_text(s.output)) or _INJECTION.search(_text(s.input)):
            add(s, 0.9, "instruction-like text in content the agent reads or writes")
        # structured output where the model normally answers in text, and isn't a tool request
        if s.kind == "model" and isinstance(out, dict) and not {"tool", "input"} <= set(out):
            add(s, 0.8, "unexpected structured model output (schema drift)")
        # truncated context: a model input carrying only part of what retrieval returned
        if s.kind == "model" and retrieved and len(retrieved) > 80:
            text = _text(s.input)
            if retrieved[:40] in text and retrieved not in text.replace("\\n", "\n"):
                add(s, 0.8, "model saw only part of the retrieved context")
        if s.kind == "retrieval" and isinstance(s.output, str):
            retrieved = s.output
        # structure drift: this actor usually produces a different shape
        same = shapes_by_actor.get((s.kind, s.actor), [])
        if len(same) >= 2 and same.count(s.shape) == 1 and isinstance(out, dict) and s.kind == "model":
            add(s, 0.75, "output structure differs from this actor's other outputs")
    # no final answer: the run ended on something other than a model turn
    if steps and steps[-1].kind != "model":
        add(steps[-1], 0.85, "run ended without a final answer")
    # claim vs evidence: the final answer should carry the last tool result
    final = next((s for s in reversed(steps) if s.kind == "model"), None)
    if final is not None and last_evidence:
        nums = _NUMBER.findall(last_evidence)
        if nums and not any(n in _text(final.output) for n in nums):
            add(final, 0.8, "final answer doesn't carry the tool evidence")
    found.sort(key=lambda x: (-x.score, x.index))
    return found


def top_k(suspects: list[Suspect], k: int) -> list[int]:
    out: list[int] = []
    for s in suspects:
        if s.index not in out:
            out.append(s.index)
        if len(out) == k:
            break
    return out


# --- live traces ---------------------------------------------------------------------------------


def steps_from_events(events: list[Event]) -> list[Step]:
    """Agent-breakpoint trace events as steps. Content is represented by request/response hashes,
    which is what divergence needs."""
    out = []
    for i, e in enumerate([e for e in events if e.call_index > 0], 1):
        kind = {"model": "model", "tool": "tool_call", "dependency": "dependency",
                "state": "state", "orchestration": "handoff"}.get(e.kind, e.kind)
        out.append(Step(i, kind, actor=e.boundary, tool=e.tool, status=e.status,
                        input=e.req_hash, output=e.detail.get("result_hash") or e.detail.get("http_status"),
                        meta={"boundary": e.boundary, "call": e.call_index, "faults": e.faults}))
    return out


def explain(events: list[Event], baseline: list[Event] | None) -> dict[str, Any]:
    """Where a red cell's failure entered and how it spread, for the report."""
    steps = steps_from_events(events)
    by_index = {s.index: s for s in steps}

    def label(i: int | None) -> str | None:
        s = by_index.get(i) if i is not None else None
        return f"{s.meta['boundary']} call {s.meta['call']}" if s else None

    fault = next((s for s in steps if s.meta.get("faults")), None)
    info: dict[str, Any] = {"fault_at": label(fault.index) if fault else None}
    if baseline is not None:
        base = steps_from_events(baseline)
        # ignore the fault step itself: we know it differs; we want where behaviour changed
        div = [i for i in divergences(steps, base) if not fault or i > fault.index]
        info["diverged_at"] = label(div[0]) if div else None
        if not div and fault and len(steps) < len(base):
            info["diverged_at"] = f"run stopped after {label(fault.index)}"
        info["then"] = [label(i) for i in div[1:4]]
    sus = [x for x in suspicion(steps) if not fault or x.index >= fault.index][:3]
    info["suspects"] = [f"{label(x.index)}: {x.rule}" for x in sus]
    return info
