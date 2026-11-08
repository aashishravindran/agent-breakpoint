"""Combination search for the full tier.

Lineage-driven fault injection, simplified: the baseline trace is the lineage of a successful
outcome. Single faults the agent tolerated show where it has redundancy (a retry, a fallback, a
second source). Pairs and triples of tolerated faults on *different* boundaries of that lineage
are the candidates most likely to defeat that redundancy together. Any failing combination is
then shrunk with delta debugging (ddmin) to the smallest fault set that still breaks the task.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from agent_breakpoint.controller import FaultSpec
from agent_breakpoint.discovery import FailurePointGraph
from agent_breakpoint.grades import Grade
from agent_breakpoint.planner import Cell, make_cell


def candidates(graph: FailurePointGraph, singles: list[tuple[Cell, Grade]], max_size: int = 2,
               limit: int = 50) -> list[Cell]:
    tolerated = [c for c, g in singles if g in (Grade.TOLERATED, Grade.DEGRADED)
                 and len(c.faults) == 1]
    # Prefer boundaries adjacent in the lineage: they back each other up most often.
    def weight(combo: tuple[Cell, ...]) -> int:
        bs = [c.boundary for c in combo]
        return -sum(graph.edges.get(a, {}).get(b, 0) + graph.edges.get(b, {}).get(a, 0)
                    for a, b in itertools.combinations(bs, 2))

    out: list[Cell] = []
    for size in range(2, max_size + 1):
        combos = [combo for combo in itertools.combinations(tolerated, size)
                  if len({c.boundary for c in combo}) == size]
        combos.sort(key=lambda combo: (weight(combo), [c.key for c in combo]))
        for combo in combos:
            specs = [c.faults[0] for c in combo]
            out.append(make_cell(graph, specs, " + ".join(c.label or c.key for c in combo)))
            if len(out) >= limit:
                return out
    return out


def ddmin(faults: list[FaultSpec], fails: Callable[[list[FaultSpec]], bool]) -> list[FaultSpec]:
    """Zeller's ddmin: the 1-minimal subset of ``faults`` for which ``fails`` still holds."""
    assert fails(faults), "ddmin needs a failing input"
    n = 2
    current = list(faults)
    while len(current) >= 2:
        chunk = max(1, len(current) // n)
        subsets = [current[i:i + chunk] for i in range(0, len(current), chunk)]
        reduced = False
        for s in subsets:
            if fails(s):
                current, n, reduced = s, 2, True
                break
        if not reduced:
            for s in subsets:
                complement = [f for f in current if f not in s]
                if complement and fails(complement):
                    current, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(current):
                break
            n = min(len(current), n * 2)
    return current
