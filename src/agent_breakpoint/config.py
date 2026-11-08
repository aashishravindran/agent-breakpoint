"""``agent-breakpoint.yaml``: the one config file a repo needs.

Targets, budgets and mutation settings come only from this file on disk. The MCP server
front door never accepts them as tool arguments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_breakpoint.grades import Grade

DEFAULT_CONFIG = "agent-breakpoint.yaml"
Provider = Literal["auto", "openai", "anthropic", "ollama", "bedrock", "gemini"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelBoundary(_Model):
    """An HTTP model API the agent calls. The agent is pointed at a local proxy via ``env``."""

    name: str = "model"
    provider: Provider = "auto"
    upstream: str
    env: list[str] = Field(default_factory=list)
    # Appended to the proxy URL when exporting env vars, e.g. "/v1" for OPENAI_BASE_URL.
    env_suffix: str = ""
    # Bedrock only: the proxy re-signs forwarded requests with local AWS credentials.
    aws_region: str | None = None


class McpBoundary(_Model):
    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: list[str] = Field(default_factory=list)
    url: str | None = None
    env: str | None = None  # env var that receives the proxy URL (http) or wrapped command (stdio)
    cwd: str | None = None
    server_env: dict[str, str] = Field(default_factory=dict)
    # Tools treated as destructive in addition to those annotated destructiveHint.
    destructive: list[str] = Field(default_factory=list)


class McpConfigRewrite(_Model):
    """Rewrite a Claude Code / Claude Desktop style ``mcpServers`` JSON file so every server goes
    through the proxy, then point the agent at the rewritten copy."""

    path: str
    env: str | None = None
    arg: str | None = None  # or replace this CLI argument's value instead of an env var


class DependencyBoundary(_Model):
    name: str
    kind: Literal["http", "tcp"] = "http"
    upstream: str
    env: str | None = None
    listen_port: int | None = None


class Canary(_Model):
    """A planted string owned by one identity; seen by anyone else, it is a leak."""

    value: str
    owner: str


class Scenario(_Model):
    name: str
    identity: str | None = None  # acting identity, for canary-leak checks
    task: dict[str, Any]
    seed: str | None = None  # "module:function(sandbox)"
    invariants: list[str] = Field(default_factory=list)  # "module:function(sandbox, outcome)"
    sessions: int = 1
    # Episode scenarios: tasks for sessions 2..n (session 1 is ``task``).
    later_tasks: list[dict[str, Any]] = Field(default_factory=list)
    expect_claim: Literal["success", "partial", "failure"] | None = "success"


class Budgets(_Model):
    max_usd: float = 5.0
    max_model_calls: int = 200
    max_tokens: int | None = None
    wall_clock_s: float = 300.0


class Pricing(_Model):
    input_per_mtok: float = 3.0
    output_per_mtok: float = 15.0


class Tier(_Model):
    runs: int = 3
    max_cells: int = 40
    max_usd: float | None = None
    busiest: int | None = 5  # only the N busiest boundaries; None = all
    combinations: bool = False
    max_combination_size: int = 2
    max_combinations: int = 30
    confidence: float = 0.95


class Gate(_Model):
    fail_on: list[Grade] = Field(default_factory=lambda: [Grade.FAILED_SILENTLY, Grade.UNSAFE])

    @field_validator("fail_on", mode="before")
    @classmethod
    def _parse(cls, v: Any) -> Any:
        return [Grade.parse(x) if isinstance(x, str) else x for x in v]


class Config(_Model):
    entrypoint: str
    python: str | None = None  # interpreter for the agent (e.g. the app's .venv); default: ours
    python_path: list[str] = Field(default_factory=lambda: ["."])
    cwd: str = "."
    seed: int = 1234
    env: dict[str, str] = Field(default_factory=dict)
    models: list[ModelBoundary] = Field(default_factory=list)
    mcp: list[McpBoundary] = Field(default_factory=list)
    mcp_config: McpConfigRewrite | None = None
    dependencies: list[DependencyBoundary] = Field(default_factory=list)
    scenarios: list[Scenario]
    budgets: Budgets = Field(default_factory=Budgets)
    pricing: Pricing = Field(default_factory=Pricing)
    tiers: dict[str, Tier] = Field(
        default_factory=lambda: {
            "quick": Tier(runs=3, max_cells=40, max_usd=1.0, busiest=5),
            "full": Tier(runs=10, max_cells=400, busiest=None, combinations=True),
        }
    )
    gate: Gate = Field(default_factory=Gate)
    packs: list[str] | Literal["all"] = "all"
    safety: list[str] = Field(
        default_factory=lambda: ["side_effect_ledger", "destructive_after_fault", "budget_cap",
                                 "canary_leak"]
    )
    mock_tools: bool = False  # serve recorded MCP responses instead of calling real tools
    run_dir: str = ".agent-breakpoint"
    llm_judge: str | None = None  # optional "module:function(task, outcome) -> float"
    canaries: list[Canary] = Field(default_factory=list)
    # Run invariants with the agent's interpreter (``python``) instead of the harness's, when
    # they need the app's own libraries to read its state.
    invariants_in_agent: bool = False
    max_delegations: int = 25
    compat: dict[str, Any] = Field(default_factory=dict)  # see compat.py

    # Resolved at load time, not part of the file.
    root: Path = Field(default=Path("."), exclude=True)

    def tier(self, name: str) -> Tier:
        if name not in self.tiers:
            raise KeyError(f"unknown tier {name!r}; configured: {sorted(self.tiers)}")
        return self.tiers[name]

    def scenario(self, name: str | None) -> Scenario:
        if name is None:
            return self.scenarios[0]
        for s in self.scenarios:
            if s.name == name:
                return s
        raise KeyError(f"unknown scenario {name!r}")

    def foreign_canaries(self, scenario: Scenario) -> list[str]:
        return [c.value for c in self.canaries if c.owner != scenario.identity]

    @property
    def python_exe(self) -> str | None:
        if not self.python:
            return None
        p = Path(self.python)
        # absolute, but not resolve(): a venv's python is a symlink and must stay one
        return str(self.root / p) if not p.is_absolute() and "/" in self.python else self.python

    @property
    def run_path(self) -> Path:
        return (self.root / self.run_dir).resolve()

    @property
    def work_dir(self) -> Path:
        return (self.root / self.cwd).resolve()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    p = Path(path or DEFAULT_CONFIG).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `agent-breakpoint init` first")
    raw = yaml.safe_load(p.read_text()) or {}
    cfg = Config.model_validate(_expand(raw))
    cfg.root = p.parent
    return cfg
