"""Statistics for cells run N times: grade distributions, success-rate drop with confidence
intervals, flaky-cell flags. No single run is a verdict."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from agent_breakpoint.grades import Grade

_Z = {0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96, 0.99: 2.5758}


def z_for(confidence: float) -> float:
    return _Z.get(round(confidence, 2), 1.96)


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff(s1: int, n1: int, s2: int, n2: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score interval (method 10)."""
    p1 = s1 / n1 if n1 else 0.0
    p2 = s2 / n2 if n2 else 0.0
    l1, u1 = wilson(s1, n1, confidence)
    l2, u2 = wilson(s2, n2, confidence)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, max(-1.0, lo), min(1.0, hi)


GOOD = {Grade.TOLERATED}
OK = {Grade.TOLERATED, Grade.DEGRADED, Grade.FAILED_LOUDLY}


@dataclass
class CellStats:
    n: int
    distribution: dict[str, int]
    success_rate: float
    success_ci: tuple[float, float]
    drop: float  # baseline success rate - cell success rate
    drop_ci: tuple[float, float]
    worst: Grade
    modal: Grade
    flaky: bool

    def to_json(self) -> dict:
        return {"n": self.n, "distribution": self.distribution,
                "success_rate": round(self.success_rate, 4),
                "success_ci": [round(x, 4) for x in self.success_ci],
                "drop": round(self.drop, 4), "drop_ci": [round(x, 4) for x in self.drop_ci],
                "worst": self.worst.value, "modal": self.modal.value, "flaky": self.flaky}


def cell_stats(grades: list[Grade], baseline: list[Grade], confidence: float = 0.95) -> CellStats:
    n = len(grades)
    dist = Counter(g.value for g in grades)
    s = sum(g in GOOD for g in grades)
    bs = sum(g in GOOD for g in baseline)
    bn = len(baseline)
    d, lo, hi = newcombe_diff(bs, bn, s, n, confidence) if n and bn else (0.0, -1.0, 1.0)
    worst = Grade.worst(grades)
    modal = Counter(grades).most_common(1)[0][0] if grades else Grade.TOLERATED
    # Flaky: the same fault sometimes lands on the acceptable side and sometimes not.
    flaky = len({g in OK for g in grades}) > 1 or (len(set(grades)) > 1 and worst.severity >= 3)
    return CellStats(n, dict(dist), s / n if n else 0.0, wilson(s, n, confidence), d, (lo, hi),
                     worst, modal, flaky)
