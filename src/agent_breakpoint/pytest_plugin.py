"""pytest plugin: an ``agent_breakpoint`` fixture.

    def test_refund_agent_has_no_silent_failures(agent_breakpoint):
        matrix = agent_breakpoint.quick("refund")
        agent_breakpoint.assert_no(matrix, "failed_silently", "unsafe")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_breakpoint.config import Config, load_config
from agent_breakpoint.grades import Grade


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agent-breakpoint")
    group.addoption("--agent-breakpoint-config", default=None,
                    help="path to agent-breakpoint.yaml (default: ./agent-breakpoint.yaml)")
    group.addoption("--agent-breakpoint-tier", default="quick")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "agent_breakpoint: fault-injection tests (slow, may cost $)")


class AgentBreakpoint:
    def __init__(self, cfg: Config, tier: str) -> None:
        self.cfg = cfg
        self.tier = tier

    def run(self, scenario: str | None = None, tier: str | None = None, **kwargs: Any) -> dict[str, Any]:
        from agent_breakpoint.suite import run_tier

        return run_tier(self.cfg, tier or self.tier, scenario, **kwargs).to_json()

    def quick(self, scenario: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.run(scenario, "quick", **kwargs)

    @staticmethod
    def cells_with(matrix: dict[str, Any], *grades: str | Grade) -> list[dict[str, Any]]:
        want = {g if isinstance(g, Grade) else Grade.parse(g) for g in grades}
        return [c for c in matrix["cells"] if Grade(c["stats"]["worst"]) in want]

    def assert_no(self, matrix: dict[str, Any], *grades: str | Grade) -> None:
        grades = grades or tuple(self.cfg.gate.fail_on)
        bad = self.cells_with(matrix, *grades)
        if bad:
            lines = []
            for c in bad:
                worst = Grade(c["stats"]["worst"])
                t = next(t for t in c["trials"] if Grade(t["grade"]) == worst)
                why = "; ".join(r for r in t["reasons"] if not r.startswith(("state=", "claim=")))
                lines.append(f"  {worst.label}: {c.get('label') or c['key']} — {why}")
            pytest.fail(f"{len(bad)} cell(s) graded {', '.join(str(g) for g in grades)}:\n"
                        + "\n".join(lines), pytrace=False)


@pytest.fixture
def agent_breakpoint(request: pytest.FixtureRequest) -> AgentBreakpoint:
    path = request.config.getoption("--agent-breakpoint-config") or "agent-breakpoint.yaml"
    if not Path(path).exists():
        pytest.skip(f"{path} not found")
    return AgentBreakpoint(load_config(path), request.config.getoption("--agent-breakpoint-tier"))
