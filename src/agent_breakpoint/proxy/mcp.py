"""MCP proxy for stdio and streamable HTTP servers.

The interceptor works on JSON-RPC messages and is transport-independent:

* client → server ``tools/call`` requests get a tool boundary id ``tool:<server>.<tool>`` and a
  call index; request-side faults may answer immediately (error), delay, or swallow the call.
* server → client responses to those requests go through response-side faults.
* ``tools/list`` responses teach the proxy which tools are destructive (``destructiveHint``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent_breakpoint.controller import Controller
from agent_breakpoint.faults.base import BoundaryKind, Delay, Fail, Hang, Reply
from agent_breakpoint.trace import Event, stable_hash

ENV_MCP_CACHE = "AGENT_BREAKPOINT_MCP_CACHE"


@dataclass
class Pending:
    event: Event
    request: dict[str, Any]
    decisions: list
    started: float


@dataclass
class ClientVerdict:
    forward: dict[str, Any] | None = None  # message to send to the server
    reply: dict[str, Any] | None = None  # message to send straight back to the client
    delay: float = 0.0


@dataclass
class Interceptor:
    server: str
    controller: Controller | None
    destructive: set[str] = field(default_factory=set)
    pending: dict[Any, Pending] = field(default_factory=dict)
    removed: set[str] = field(default_factory=set)
    cache_path: Path | None = None
    mock: bool = False
    _cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.controller:
            self.destructive |= set(self.controller.plan.destructive_tools)
            self.mock = self.controller.plan.mock_tools
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["result"]

    def boundary(self, tool: str) -> str:
        return f"tool:{self.server}.{tool}"

    @staticmethod
    def _key(tool: str, args: Any) -> str:
        return stable_hash({"tool": tool, "args": args})

    def on_client(self, msg: dict[str, Any]) -> ClientVerdict:
        ctl = self.controller
        if ctl is None or msg.get("method") != "tools/call" or "id" not in msg:
            return ClientVerdict(forward=msg)
        params = msg.get("params") or {}
        tool = params.get("name", "?")
        boundary = self.boundary(tool)
        idx = ctl.next_call(boundary)
        destructive = tool in self.destructive
        ev = Event(boundary=boundary, kind="tool", call_index=idx, tool=tool,
                   op="tools/call", destructive=destructive,
                   req_hash=self._key(tool, params.get("arguments")))
        if destructive:
            ev.tags.append("destructive")
        decisions = ctl.decide(boundary, BoundaryKind.TOOL, idx)
        for d in decisions:
            d.ctx.meta["destructive_tool"] = next(iter(sorted(self.destructive)), None)
        ev.faults = [d.fault.id for d in decisions]
        delay = 0.0
        for d in decisions:
            if d.fault.before is None:
                continue
            out = d.fault.before(d.ctx, msg)
            if isinstance(out, dict):
                msg = out
            elif isinstance(out, Fail):
                ev.status = "error"
                ctl.record(ev)
                return ClientVerdict(reply={"jsonrpc": "2.0", "id": msg["id"], "error": {
                    "code": out.status, "message": str(out.body)}})
            elif isinstance(out, Reply):
                ev.detail.update(executed=False, server_ok=False)
                ev.status = "ok"
                ctl.record(ev)
                return ClientVerdict(reply={"jsonrpc": "2.0", "id": msg["id"], "result": out.result})
            elif isinstance(out, Hang):
                ev.status = "hang"
                ctl.record(ev)
                return ClientVerdict()
            elif isinstance(out, Delay):
                delay += out.seconds
        if self.mock and ev.req_hash in self._cache:
            ev.status = "mocked"
            pend = Pending(ev, msg, decisions, time.time())
            reply = self._finish(pend, {"jsonrpc": "2.0", "id": msg["id"],
                                        "result": self._cache[ev.req_hash]})
            return ClientVerdict(reply=reply, delay=delay)
        with self._lock:
            self.pending[msg["id"]] = Pending(ev, msg, decisions, time.time())
        return ClientVerdict(forward=msg, delay=delay)

    def on_server(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        if "id" in msg and ("result" in msg or "error" in msg):
            with self._lock:
                pend = self.pending.pop(msg["id"], None)
            if pend is not None:
                return self._finish(pend, msg)
            self._learn_tools(msg)
            return self._filter_removed(msg)
        return msg

    def _finish(self, pend: Pending, msg: dict[str, Any]) -> dict[str, Any] | None:
        ev = pend.event
        ctl = self.controller
        assert ctl is not None
        if (ctl.plan.mode == "record" and self.cache_path and "result" in msg
                and ev.status != "mocked"):
            with self._lock, self.cache_path.open("a") as fh:
                fh.write(json.dumps({"key": ev.req_hash, "tool": ev.tool,
                                     "result": msg["result"]}) + "\n")
        real = msg.get("result")
        ev.detail["executed"] = ev.status != "mocked"
        ev.detail["server_ok"] = "error" not in msg and not (
            isinstance(real, dict) and real.get("isError"))
        for d in pend.decisions:
            if d.fault.after is None:
                continue
            out = d.fault.after(d.ctx, pend.request, msg)
            if isinstance(out, dict):
                msg = out
            if d.ctx.meta.get("canary"):
                ev.canary = d.ctx.meta.pop("canary")
            if d.fault.id == "tool.removed_mid_session":
                self.removed.add(ev.tool or "")
        res = msg.get("result")
        if "error" in msg or (isinstance(res, dict) and res.get("isError")):
            ev.status = "error"
        ev.latency_ms = (time.time() - pend.started) * 1000
        ev.detail["result_hash"] = stable_hash(res) if res is not None else None
        if ctl.canaries and res is not None:
            text = json.dumps(res)
            seen = [c for c in ctl.canaries if c in text]
            if seen:
                ev.detail["canaries_seen"] = seen
        ctl.record(ev)
        return msg

    def _learn_tools(self, msg: dict[str, Any]) -> None:
        res = msg.get("result")
        if not isinstance(res, dict) or not isinstance(res.get("tools"), list):
            return
        names = []
        schemas = {}
        for t in res["tools"]:
            name = t.get("name")
            names.append(name)
            schemas[name] = stable_hash(t.get("inputSchema") or {})
            ann = t.get("annotations") or {}
            if ann.get("destructiveHint") or (ann.get("readOnlyHint") is False
                                              and ann.get("destructiveHint") is not False
                                              and "destructiveHint" in ann):
                self.destructive.add(name)
        if self.controller:
            self.controller.record(Event(
                boundary=f"mcp:{self.server}", kind="tool", call_index=0, op="tools/list",
                detail={"tools": names, "schemas": schemas,
                        "destructive": sorted(self.destructive)}))

    def _filter_removed(self, msg: dict[str, Any]) -> dict[str, Any]:
        res = msg.get("result")
        if self.removed and isinstance(res, dict) and isinstance(res.get("tools"), list):
            res["tools"] = [t for t in res["tools"] if t.get("name") not in self.removed]
        return msg


# --- stdio transport ------------------------------------------------------------------------


def run_stdio(server: str, command: list[str], destructive: list[str] | None = None) -> int:
    """Launched in place of a stdio MCP server; sits on its stdin/stdout."""
    ctl = Controller.from_env(component=f"mcp-{server}")
    cache = os.environ.get(ENV_MCP_CACHE)
    icpt = Interceptor(server, ctl, destructive=set(destructive or []),
                       cache_path=Path(cache) if cache else None)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=None, bufsize=0)
    out_lock = threading.Lock()
    in_lock = threading.Lock()

    def to_client(obj: dict[str, Any]) -> None:
        data = (json.dumps(obj) + "\n").encode()
        with out_lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    def to_server(obj: dict[str, Any]) -> None:
        assert proc.stdin is not None
        with in_lock:
            try:
                proc.stdin.write((json.dumps(obj) + "\n").encode())
                proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    def pump_server() -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(raw)
            except ValueError:
                with out_lock:
                    sys.stdout.buffer.write(raw)
                    sys.stdout.buffer.flush()
                continue
            out = icpt.on_server(msg) if isinstance(msg, dict) else msg
            if out is not None:
                to_client(out)

    t = threading.Thread(target=pump_server, daemon=True)
    t.start()
    for raw in iter(sys.stdin.buffer.readline, b""):
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(msg, dict):  # batches are not used by current MCP clients
            to_server(msg)
            continue
        verdict = icpt.on_client(msg)
        if verdict.reply is not None:
            if verdict.delay:
                threading.Timer(verdict.delay, to_client, args=(verdict.reply,)).start()
            else:
                to_client(verdict.reply)
        if verdict.forward is not None:
            if verdict.delay:
                threading.Timer(verdict.delay, to_server, args=(verdict.forward,)).start()
            else:
                to_server(verdict.forward)
    try:
        assert proc.stdin is not None
        proc.stdin.close()
    except OSError:
        pass
    try:
        return proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 0


def wrap_stdio_command(server: str, command: list[str], destructive: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "agent_breakpoint.proxy.mcp_stdio_main", "--name", server]
    for d in destructive or []:
        cmd += ["--destructive", d]
    return [*cmd, "--", *command]


# --- streamable HTTP transport --------------------------------------------------------------


class HttpMcpProxy:
    """Reverse proxy for a streamable HTTP MCP endpoint. Session ids (``Mcp-Session-Id``) and
    resumption headers (``Last-Event-ID``) pass through untouched; JSON and SSE response bodies
    to POSTs are parsed so tool faults apply to the matching JSON-RPC response."""

    def __init__(self, server: str, upstream: str, destructive: list[str] | None = None) -> None:
        self.server = server
        self.upstream = upstream
        self.icpt = Interceptor(server, None, destructive=set(destructive or []))
        self._client = None

    def begin_trial(self, controller: Controller, cache_path: Path | None) -> None:
        self.icpt = Interceptor(self.server, controller, destructive=set(self.icpt.destructive),
                                cache_path=cache_path)

    async def __call__(self, scope, receive, send) -> None:
        import asyncio

        import httpx
        from starlette.requests import Request
        from starlette.responses import Response, StreamingResponse

        from agent_breakpoint.faults.http import clean_headers

        if scope["type"] != "http":
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        request = Request(scope, receive)
        url = self._target(request.url.path, request.url.query)
        headers = clean_headers(dict(request.headers))
        body = await request.body()

        if request.method == "GET":  # server-initiated SSE stream: pass through, unbuffered
            req = self._client.build_request("GET", url, headers=headers)
            upstream = await self._client.send(req, stream=True)

            async def gen():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()

            resp = StreamingResponse(gen(), status_code=upstream.status_code,
                                     headers=clean_headers(dict(upstream.headers)))
            await resp(scope, receive, send)
            return

        replies: list[dict[str, Any]] = []
        forward_body = body
        if request.method == "POST" and body:
            try:
                msg = json.loads(body)
            except ValueError:
                msg = None
            if isinstance(msg, dict):
                verdict = self.icpt.on_client(msg)
                if verdict.delay:
                    await asyncio.sleep(verdict.delay)
                if verdict.reply is not None:
                    replies.append(verdict.reply)
                if verdict.forward is None:
                    if verdict.reply is None:  # swallowed: hang until the client gives up
                        await asyncio.sleep(3600)
                    resp = Response(json.dumps(verdict.reply), media_type="application/json",
                                    headers=self._session_headers(request))
                    await resp(scope, receive, send)
                    return
                forward_body = json.dumps(verdict.forward).encode()
        r = await self._client.request(request.method, url, headers=headers, content=forward_body)
        out_headers = clean_headers(dict(r.headers))
        ctype = r.headers.get("content-type", "")
        content = r.content
        if "text/event-stream" in ctype:
            content = self._rewrite_sse(content)
        elif "application/json" in ctype and content:
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    data = self.icpt.on_server(data)
                    content = json.dumps(data).encode()
            except ValueError:
                pass
        resp = Response(content, status_code=r.status_code, headers=out_headers)
        await resp(scope, receive, send)

    def _target(self, path: str, query: str) -> str:
        """The proxy serves the upstream endpoint at the same path: forward to its origin."""
        parts = urlsplit(self.upstream)
        return f"{parts.scheme}://{parts.netloc}{path}" + (f"?{query}" if query else "")

    @staticmethod
    def _session_headers(request) -> dict[str, str]:
        sid = request.headers.get("mcp-session-id")
        return {"mcp-session-id": sid} if sid else {}

    def _rewrite_sse(self, content: bytes) -> bytes:
        out = []
        for event in content.split(b"\n\n"):
            if not event.strip():
                continue
            lines = event.split(b"\n")
            new_lines = []
            for line in lines:
                if line.startswith(b"data:"):
                    payload = line[5:].strip()
                    try:
                        msg = json.loads(payload)
                    except ValueError:
                        new_lines.append(line)
                        continue
                    if isinstance(msg, dict):
                        msg = self.icpt.on_server(msg)
                        if msg is None:
                            continue
                    new_lines.append(b"data: " + json.dumps(msg).encode())
                else:
                    new_lines.append(line)
            out.append(b"\n".join(new_lines))
        return b"\n\n".join(out) + b"\n\n"
