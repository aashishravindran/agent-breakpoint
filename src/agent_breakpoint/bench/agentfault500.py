"""Benchmark agent-breakpoint's trace oracles on AgentFault-500 (Oct 2026).

AgentFault-500 is 160 clean and 500 fault-injected trajectories from a mock-LLM, four-agent
LangGraph workflow, each injected trajectory derived from a baseline by mutating one step
(``is_root_cause``). It has trajectories only: no tasks to rerun and no end state, so this
benchmarks trace grading and root-cause localization, not the state oracle.

Reported, all with 95% bootstrap intervals resampled by baseline group:

* detection AUROC (injected vs clean): claim oracle alone; unpaired trace rules
* root-cause step accuracy, exact and top-3: unpaired trace rules; paired first divergence from
  the record's own baseline (the view agent-breakpoint has at run time)

Groups are split by a hash of the baseline id: 80% dev (rules were designed looking only at
these) and 20% held out. Both are reported; quote the held-out numbers.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_breakpoint.contract import Outcome
from agent_breakpoint.oracles.claim import infer_claim
from agent_breakpoint.rootcause import Step, divergences, suspicion, top_k

REPO = "Skywalkr-dev/AgentFault-Infra"
COMMIT_FILES = {  # sha256 from the release manifest (content hash d5d2cca9…)
    "baseline.jsonl": "805abf91a6de13f35738a6201035af893db336e72538843c64c8831e188abe41",
    "injected.jsonl": "b3f0a4b2426ea7fdf38cf4e90fb2b873b4d2a6496941b1e1e278f3859e2eec28",
}
DEFAULT_DIR = Path.home() / ".cache" / "agent-breakpoint" / "agentfault-500"
KIND = {"RETRIEVAL": "retrieval", "LLM_CALL": "model", "TOOL_CALL": "tool_call",
        "TOOL_RESULT": "tool_result", "HANDOFF": "handoff"}


def download(dest: Path = DEFAULT_DIR) -> Path:
    """Fetch the two files from the generator repo and verify them against the manifest hashes.
    The data is CC BY 4.0 (IEEE DataPort); it is cached locally, never vendored."""
    import httpx

    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in COMMIT_FILES.items():
        path = dest / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            url = f"https://raw.githubusercontent.com/{REPO}/HEAD/dataset/agentfault-500/{name}"
            data = httpx.get(url, timeout=60, follow_redirects=True).raise_for_status().content
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"{name}: checksum mismatch; the release changed upstream")
            path.write_bytes(data)
    return dest


def baseline_id(trajectory_id: str) -> str:
    return trajectory_id.split("_fault_")[0]


def held_out(group: str) -> bool:
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 5 == 0


def to_steps(record: dict[str, Any]) -> list[Step]:
    return [Step(s["step_index"], KIND.get(s["step_type"], s["step_type"].lower()),
                 actor=s.get("agent_id") or "",
                 tool=s.get("tool_name"), status=s.get("status") or "unknown", input=s.get("input"),
                 output=s.get("output"), meta=s.get("metadata") or {}) for s in record["steps"]]


def final_answer(record: dict[str, Any]) -> str:
    for s in reversed(record["steps"]):
        if s["step_type"] == "LLM_CALL":
            out = s.get("output")
            return out if isinstance(out, str) else json.dumps(out)
    return ""


@dataclass
class Row:
    group: str
    family: str  # "clean" for baselines
    injected: bool
    claim_score: float
    trace_score: float
    truth: set[int]
    unpaired_top: list[int]
    paired_top: list[int]


def score(data_dir: Path) -> list[Row]:
    base = {r["trajectory_id"]: r for r in map(json.loads, (data_dir / "baseline.jsonl").open())}
    inj = [json.loads(line) for line in (data_dir / "injected.jsonl").open()]
    agents = {s["agent_id"] for r in base.values() for s in r["steps"] if s.get("agent_id")}
    rows = []
    for rec in [*base.values(), *inj]:
        injected = bool(rec.get("fault_injected"))
        group = baseline_id(rec["trajectory_id"])
        steps = to_steps(rec)
        sus = suspicion(steps, agents)
        claim = infer_claim(Outcome(answer=final_answer(rec)))
        paired: list[int] = []
        if injected and group in base:
            paired = divergences(steps, to_steps(base[group]))[:3]
        rows.append(Row(
            group, rec.get("fault_type", "clean").split("_")[0] if injected else "clean", injected,
            claim_score={"success": 0.0, "partial": 0.5, "failure": 1.0}[claim],
            trace_score=sus[0].score if sus else 0.0,
            truth={s["step_index"] for s in rec["steps"] if s.get("is_root_cause")},
            unpaired_top=top_k(sus, 3), paired_top=paired))
    return rows


def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney U / (n_pos * n_neg), ties count half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: dict[float, float] = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        ranks[allv[i][0]] = (i + j + 1) / 2  # average 1-based rank
        i = j
    r = sum(ranks[v] for v in pos)
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _metrics(rows: list[Row]) -> dict[str, float]:
    pos = [r for r in rows if r.injected]
    neg = [r for r in rows if not r.injected]
    loc = [r for r in pos if r.truth]

    def acc(pick, k: int) -> float:
        return sum(bool(set(pick(r)[:k]) & r.truth) for r in loc) / len(loc) if loc else float("nan")

    return {
        "auroc_claim": auroc([r.claim_score for r in pos], [r.claim_score for r in neg]),
        "auroc_trace": auroc([r.trace_score for r in pos], [r.trace_score for r in neg]),
        "rootcause_unpaired_exact": acc(lambda r: r.unpaired_top, 1),
        "rootcause_unpaired_top3": acc(lambda r: r.unpaired_top, 3),
        "rootcause_paired_exact": acc(lambda r: r.paired_top, 1),
        "rootcause_paired_top3": acc(lambda r: r.paired_top, 3),
    }


def _bootstrap(rows: list[Row], n: int = 1000, seed: int = 7) -> dict[str, tuple[float, float]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.group].append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n):
        pick = [r for _ in keys for r in groups[rng.choice(keys)]]
        for k, v in _metrics(pick).items():
            if v == v:  # skip NaN
                samples[k].append(v)
    return {k: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]) for k, v in samples.items()}


def run(data_dir: Path | None = None, bootstrap: int = 1000) -> dict[str, Any]:
    data_dir = data_dir or download()
    rows = score(data_dir)
    out: dict[str, Any] = {"dataset": "AgentFault-500", "source": f"github.com/{REPO}",
                           "records": len(rows), "splits": {}}
    for name, subset in (("held_out", [r for r in rows if held_out(r.group)]),
                         ("dev", [r for r in rows if not held_out(r.group)]), ("all", rows)):
        m = _metrics(subset)
        ci = _bootstrap(subset, bootstrap) if bootstrap else {}
        fams: dict[str, dict[str, float]] = {}
        for fam in sorted({r.family for r in subset if r.injected}):
            fr = [r for r in subset if r.family == fam or not r.injected]
            fm = _metrics(fr)
            fams[fam] = {"n": sum(r.family == fam for r in fr), "auroc_trace": fm["auroc_trace"],
                         "rootcause_unpaired_top3": fm["rootcause_unpaired_top3"],
                         "rootcause_paired_exact": fm["rootcause_paired_exact"]}
        out["splits"][name] = {
            "trajectories": len(subset), "injected": sum(r.injected for r in subset),
            "groups": len({r.group for r in subset}),
            "metrics": {k: {"value": v, "ci95": ci.get(k)} for k, v in m.items()},
            "by_family": fams}
    return out
