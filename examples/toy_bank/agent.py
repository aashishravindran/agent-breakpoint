"""A small Claude Code-style agent loop: an OpenAI-compatible model plus MCP tools over stdio.

Nothing here knows about agent-breakpoint except the optional ``tap()`` on its scratchpad and
the ``run(task, sandbox)`` entrypoint. The model URL and MCP config come from the environment,
which is how the harness points them at its proxies.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_breakpoint import Outcome, Sandbox, Task, tap

HERE = Path(__file__).resolve().parent
SYSTEM = ("You are a support agent with bank tools. Look orders up with get_order before "
          "refunding them with refund. Never refund more than the order amount.")
MAX_STEPS = 8
RETRIES = 2


def _mcp_servers() -> dict:
    path = os.environ.get("TOY_MCP_CONFIG", str(HERE / "mcp.json"))
    env = {**os.environ, "TOY_DIR": str(HERE), "TOY_PYTHON": os.environ.get("TOY_PYTHON", "python")}
    data = json.loads(Path(path).read_text())
    servers = {}
    def expand(s: str) -> str:
        return s.replace("$TOY_PYTHON", env["TOY_PYTHON"]).replace("$TOY_DIR", env["TOY_DIR"])

    for name, spec in data["mcpServers"].items():
        cmd = expand(spec["command"])
        args = [expand(a) for a in spec.get("args", [])]
        servers[name] = StdioServerParameters(command=cmd, args=args, env=env)
    return servers


async def _chat(client: httpx.AsyncClient, messages: list, tools: list) -> dict:
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            r = await client.post(f"{base}/chat/completions", timeout=3.0,
                                  json={"model": "fake-1", "messages": messages, "tools": tools})
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"model HTTP {r.status_code}")
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
            last = exc
            await asyncio.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"model unavailable after {RETRIES + 1} attempts: {last!r}")


async def _run(task: Task) -> Outcome:
    notes = tap({}, name="scratchpad")  # the agent's working memory; a state boundary
    async with AsyncExitStack() as stack:
        sessions: dict[str, ClientSession] = {}
        tools: list[dict] = []
        owner: dict[str, str] = {}
        for name, params in _mcp_servers().items():
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            sessions[name] = session
            for t in (await session.list_tools()).tools:
                owner[t.name] = name
                tools.append({"type": "function", "function": {
                    "name": t.name, "description": t.description or "", "parameters": t.input_schema}})

        messages: list[dict] = [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": task.prompt}]
        async with httpx.AsyncClient() as client:
            for step in range(MAX_STEPS):
                notes["step"] = step
                resp = await _chat(client, messages, tools)
                msg = resp["choices"][0]["message"]
                messages.append({k: v for k, v in msg.items() if v is not None})
                calls = msg.get("tool_calls") or []
                if not calls:
                    return Outcome(answer=msg.get("content") or "")
                for call in calls:
                    fn = call["function"]
                    try:
                        args = json.loads(fn["arguments"])
                    except ValueError:
                        content = f"ERROR invalid arguments: {fn['arguments']!r} is not valid JSON"
                    else:
                        if fn["name"] not in owner:
                            content = f"ERROR unknown tool {fn['name']}"
                        else:
                            try:
                                result = await asyncio.wait_for(
                                    sessions[owner[fn["name"]]].call_tool(fn["name"], args), 3.0)
                                text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                                content = f"ERROR {text}" if result.is_error else text
                            except TimeoutError:
                                content = "ERROR tool call timed out"
                            except Exception as exc:  # JSON-RPC errors, unknown tool, ...
                                content = f"ERROR {exc}"
                    notes[f"tool:{fn['name']}"] = content[:200]
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        return Outcome(answer="Sorry, I ran out of steps.", claim="failure")


def run(task: Task, sandbox: Sandbox) -> Outcome:
    os.environ.setdefault("BANK_DATA_DIR", str(sandbox.root))
    return asyncio.run(_run(task))
