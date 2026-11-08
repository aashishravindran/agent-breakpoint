"""The onboarding ladder: ``init`` detects the stack and writes files; ``record`` turns one
normal run into a scenario and suggests invariants from the end-state diff."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FRAMEWORKS = {
    "langgraph": "LangGraph", "strands": "Strands Agents", "claude_agent_sdk": "Claude Agent SDK",
    "claude-agent-sdk": "Claude Agent SDK", "pydantic_ai": "Pydantic AI", "pydantic-ai": "Pydantic AI",
    "openai": "OpenAI SDK", "anthropic": "Anthropic SDK", "langchain": "LangChain",
    "google-generativeai": "Gemini SDK", "ollama": "Ollama", "mcp": "MCP SDK",
}

PROVIDERS = [
    # (provider, env var that holds a key or URL, base-url env var to override, suffix, upstream)
    ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "", "https://api.anthropic.com"),
    ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", "/v1", "https://api.openai.com"),
    ("ollama", "OLLAMA_BASE_URL", "OLLAMA_BASE_URL", "", "http://localhost:11434"),
    ("deepseek", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "/v1", "https://api.deepseek.com"),
    ("bedrock", "AWS_PROFILE", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "",
     "https://bedrock-runtime.us-east-1.amazonaws.com"),
]

MCP_CONFIG_FILES = [".mcp.json", "mcp.json", ".cursor/mcp.json", ".vscode/mcp.json",
                    "claude_desktop_config.json"]


@dataclass
class Detection:
    root: Path
    frameworks: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    mcp_config: str | None = None
    mcp_servers: list[str] = field(default_factory=list)
    checkpointer: bool = False


def _dep_text(root: Path) -> str:
    text = ""
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt", "setup.cfg"):
        p = root / name
        if p.exists():
            text += p.read_text().lower() + "\n"
    return text


def detect(root: Path) -> Detection:
    d = Detection(root)
    deps = _dep_text(root)
    for key, label in FRAMEWORKS.items():
        pattern = rf"(^|[\s\"'\[,]){re.escape(key)}([\s<>=~!\[\"',;]|$)"
        if re.search(pattern, deps, re.MULTILINE) and label not in d.frameworks:
            d.frameworks.append(label)
    env_text = (root / ".env").read_text() if (root / ".env").exists() else ""
    for prov, key, *_ in PROVIDERS:
        if os.environ.get(key) or re.search(rf"^{key}=", env_text, re.MULTILINE):
            d.providers.append(prov)
    if not d.providers and "ollama" in deps:
        d.providers.append("ollama")
    for name in MCP_CONFIG_FILES:
        p = root / name
        if p.exists():
            try:
                servers = json.loads(p.read_text()).get("mcpServers", {})
            except ValueError:
                continue
            d.mcp_config, d.mcp_servers = name, sorted(servers)
            break
    src = "\n".join(p.read_text(errors="ignore") for p in list(root.rglob("*.py"))[:400]
                    if ".venv" not in p.parts and "site-packages" not in p.parts)
    d.checkpointer = "checkpointer=" in src
    return d


def config_yaml(d: Detection) -> str:
    lines = [
        "# agent-breakpoint config. Docs: https://github.com/aashishravindran/agent-breakpoint",
        "# Detected: " + (", ".join(d.frameworks) or "no known framework"),
        "entrypoint: agent_breakpoint_entry:run",
        'python_path: ["."]',
        "# python: .venv/bin/python   # run the agent with the app's own interpreter",
        "",
        "models:",
    ]
    provs = d.providers or ["openai"]
    for prov, _key, base_env, suffix, upstream in PROVIDERS:
        if prov in provs:
            lines += [f"  - name: {prov}", f"    provider: {'openai' if prov == 'deepseek' else prov}",
                      f"    upstream: {upstream}", f"    env: [{base_env}]"]
            if suffix:
                lines.append(f"    env_suffix: {suffix}")
    if d.mcp_config:
        lines += ["", "mcp_config:", f"  path: {d.mcp_config}",
                  "  env: AGENT_MCP_CONFIG   # point your agent at the rewritten copy",
                  "", "mcp:"]
        lines += [f"  - name: {s}\n    destructive: []   # tools with side effects" for s in d.mcp_servers]
    lines += [
        "", "scenarios:", "  - name: main",
        '    task: {prompt: "Describe one real task your agent should complete."}',
        "    # seed: agent_breakpoint_entry:seed",
        "    # invariants: [agent_breakpoint_entry:task_done]   # turns on the state oracle",
        "", "budgets: {max_usd: 2.0, max_model_calls: 100, wall_clock_s: 300}",
        "gate: {fail_on: [failed_silently, unsafe]}", "",
    ]
    return "\n".join(lines)


ENTRY_STUB = '''"""agent-breakpoint entrypoint: start your agent on a task, return what it claims.

Nothing about your agent changes: the harness points its model and MCP URLs at local proxies
through the environment. Keep durable state under ``sandbox.root`` so trials are isolated.
"""

from agent_breakpoint import Outcome, Sandbox, Task, tap  # noqa: F401


def run(task: Task, sandbox: Sandbox) -> Outcome:
    # Example for a LangGraph app:
    #   graph = builder.compile(checkpointer=tap(sandbox.checkpointer()))
    #   result = graph.invoke({"messages": [task.prompt]},
    #                         {"configurable": {"thread_id": task.thread_id}})
    #   return Outcome(answer=result["messages"][-1].content, claim=result.get("status"))
    raise NotImplementedError("call your agent here")


def seed(sandbox: Sandbox) -> None:
    """Optional: write the starting state for each trial (fixtures, a test database)."""


def task_done(sandbox: Sandbox, outcome: Outcome) -> bool:
    """Optional invariant: inspect the sandbox and say whether the task really happened."""
    return True
'''


def write_init(root: Path, force: bool = False) -> list[Path]:
    d = detect(root)
    written = []
    targets = {root / "agent-breakpoint.yaml": config_yaml(d)}
    if not (root / "agent_breakpoint_entry.py").exists():
        targets[root / "agent_breakpoint_entry.py"] = ENTRY_STUB
    for path, content in targets.items():
        if path.exists() and not force:
            continue
        path.write_text(content)
        written.append(path)
    gi = root / ".gitignore"
    line = ".agent-breakpoint/"
    if not gi.exists() or line not in gi.read_text():
        with gi.open("a") as fh:
            fh.write(f"\n{line}\n")
    return written


# --- invariant suggestions from the baseline end-state diff ------------------------------------


def _sqlite_counts(path: Path) -> dict[str, int]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        out = {t: con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
        con.close()
        return out
    except sqlite3.Error:
        return {}


def _describe(path: Path) -> Any:
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text())
        except ValueError:
            return None
    if path.suffix in (".sqlite", ".db", ".sqlite3"):
        return {"__sqlite__": _sqlite_counts(path)}
    return None


def suggest_invariants(before: Path, after: Path) -> str:
    """Python source with one candidate invariant per changed file; a person approves them."""
    funcs = []
    for p in sorted(after.rglob("*")):
        if not p.is_file() or p.name.startswith(".agent-breakpoint"):
            continue
        rel = p.relative_to(after)
        old = before / rel
        if old.exists() and old.read_bytes() == p.read_bytes():
            continue
        new_v, old_v = _describe(p), _describe(old) if old.exists() else None
        fname = re.sub(r"\W+", "_", str(rel)).strip("_").lower()
        if isinstance(new_v, list):
            n_old = len(old_v) if isinstance(old_v, list) else 0
            funcs.append(f'''def {fname}_grew_by_{len(new_v) - n_old}(sandbox: Sandbox, outcome: Outcome):
    """Baseline: {rel} went from {n_old} to {len(new_v)} entries."""
    data = json.loads(sandbox.path({str(rel)!r}).read_text())
    return len(data) == {len(new_v)}, f"{{len(data)}} entries"
''')
        elif isinstance(new_v, dict) and "__sqlite__" in new_v:
            old_c = (old_v or {}).get("__sqlite__", {}) if isinstance(old_v, dict) else {}
            for table, n in new_v["__sqlite__"].items():
                if old_c.get(table) != n:
                    funcs.append(f'''def {fname}_{table}_rows(sandbox: Sandbox, outcome: Outcome):
    """Baseline: {rel}:{table} has {n} rows (was {old_c.get(table, 0)})."""
    con = sqlite3.connect(sandbox.path({str(rel)!r}))
    n = con.execute('select count(*) from "{table}"').fetchone()[0]
    return n == {n}, f"{{n}} rows"
''')
        elif isinstance(new_v, dict):
            changed = sorted(k for k in new_v if not isinstance(old_v, dict) or old_v.get(k) != new_v[k])[:8]
            checks = " and ".join(f"data.get({k!r}) == {new_v[k]!r}" for k in changed
                                  if isinstance(new_v[k], (str, int, float, bool)))
            if checks:
                funcs.append(f'''def {fname}_fields(sandbox: Sandbox, outcome: Outcome):
    """Baseline changed {', '.join(changed)} in {rel}."""
    data = json.loads(sandbox.path({str(rel)!r}).read_text())
    return {checks}
''')
        else:
            funcs.append(f'''def {fname}_exists(sandbox: Sandbox, outcome: Outcome):
    """Baseline wrote {rel}."""
    return sandbox.path({str(rel)!r}).exists()
''')
    header = ('"""Invariants suggested from the baseline run\'s end-state diff. Review each one:\n'
              'keep those that define success, delete the rest, then list them under the\n'
              'scenario\'s `invariants:` in agent-breakpoint.yaml."""\n\n'
              "import json\nimport sqlite3\n\nfrom agent_breakpoint import Outcome, Sandbox\n\n\n")
    return header + "\n\n".join(funcs) if funcs else header + "# The baseline changed no files.\n"
