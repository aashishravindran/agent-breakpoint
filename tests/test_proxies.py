import asyncio
import json
import socket
import threading

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_breakpoint.config import Budgets, DependencyBoundary, ModelBoundary, Pricing
from agent_breakpoint.controller import Controller, FaultSpec, Plan
from agent_breakpoint.proxy.dependency import HttpDependencyProxy, TcpDependencyProxy
from agent_breakpoint.proxy.mcp import HttpMcpProxy
from agent_breakpoint.proxy.model import ModelProxy, detect_provider, usage_of
from agent_breakpoint.proxy.server import BackgroundServer
from agent_breakpoint.trace import read_trace


def test_provider_detection():
    assert detect_provider("/v1/messages", {"anthropic-version": "x"}) == "anthropic"
    assert detect_provider("/api/chat", {}) == "ollama"
    assert detect_provider("/model/anthropic.claude/converse", {}) == "bedrock"
    assert detect_provider("/v1beta/models/g:generateContent", {}) == "gemini"
    assert detect_provider("/v1/chat/completions", {}) == "openai"
    assert usage_of(b'{"usage": {"input_tokens": 12, "output_tokens": 3}}') == (12, 3)


@pytest.fixture
def upstream():
    calls = []

    async def echo(request):
        body = await request.json()
        calls.append(body)
        return JSONResponse({"choices": [{"message": {"content": "hi"}}],
                             "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})

    app = Starlette(routes=[Route("/v1/chat/completions", echo, methods=["POST"]),
                            Route("/policy", lambda r: JSONResponse({"allow": ["alice"]}))])
    with BackgroundServer(app) as s:
        yield s.url, calls


def _model_proxy(url, tmp_path, faults, **plan):
    proxy = ModelProxy(ModelBoundary(name="llm", upstream=url), Budgets(max_model_calls=5), Pricing())
    ctl = Controller(Plan(seed=1, faults=faults, **plan), tmp_path, "test")
    return proxy, ctl


def test_model_proxy_faults_record_and_replay(upstream, tmp_path):
    url, calls = upstream
    cache = tmp_path / "cache.jsonl"
    proxy, ctl = _model_proxy(url, tmp_path, [], mode="record")
    with BackgroundServer(proxy) as s:
        proxy.begin_trial(ctl, cache)
        for _ in range(2):
            assert httpx.post(f"{s.url}/v1/chat/completions", json={"messages": [1]}).status_code == 200
        assert len(calls) == 2 and proxy.usage.usd > 0
        # inject: call 1 replayed from cache (free), call 2 rate-limited
        ctl2 = Controller(Plan(seed=1, faults=[FaultSpec("model:llm", "model.rate_limit", 2)],
                               replay_cache=str(cache), replay_until={"model:llm": 1}),
                          tmp_path / "t2", "test")
        proxy.begin_trial(ctl2, None)
        r1 = httpx.post(f"{s.url}/v1/chat/completions", json={"messages": [1]})
        r2 = httpx.post(f"{s.url}/v1/chat/completions", json={"messages": [1]})
        assert r1.status_code == 200 and r2.status_code == 429
        assert len(calls) == 2  # nothing reached upstream
        statuses = [e.status for e in read_trace(tmp_path / "t2")]
        assert statuses == ["replayed", "error"]


def test_model_proxy_enforces_budget(upstream, tmp_path):
    url, _ = upstream
    proxy, ctl = _model_proxy(url, tmp_path, [])
    proxy.budgets = Budgets(max_model_calls=2)
    with BackgroundServer(proxy) as s:
        proxy.begin_trial(ctl, None)
        codes = [httpx.post(f"{s.url}/v1/chat/completions", json={}).status_code for _ in range(3)]
    assert codes == [200, 200, 429] and proxy.usage.blown == "max_model_calls=2"


def test_http_dependency_proxy(upstream, tmp_path):
    url, _ = upstream
    dp = HttpDependencyProxy(DependencyBoundary(name="policy", upstream=url))
    ctl = Controller(Plan(faults=[FaultSpec("dep:policy", "dependency.server_error", 2),
                                  FaultSpec("dep:policy", "dependency.empty", 3)]), tmp_path, "t")
    with BackgroundServer(dp) as s:
        dp.begin_trial(ctl)
        r = [httpx.get(f"{s.url}/policy") for _ in range(3)]
    assert r[0].json() == {"allow": ["alice"]}
    assert r[1].status_code == 503
    assert r[2].json() == {"allow": []}


def test_tcp_dependency_proxy_reset(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]

    def serve():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            c.sendall(c.recv(100).upper())
            c.close()

    threading.Thread(target=serve, daemon=True).start()
    tp = TcpDependencyProxy(DependencyBoundary(name="db", kind="tcp", upstream=f"127.0.0.1:{port}")).start()
    tp.begin_trial(Controller(Plan(faults=[FaultSpec("dep:db", "dependency.reset", 2)]), tmp_path, "t"))
    try:
        def call():
            s = socket.create_connection(("127.0.0.1", tp.port), timeout=2)
            s.sendall(b"ping")
            data = s.recv(100)
            s.close()
            return data
        assert call() == b"PING"
        with pytest.raises(OSError):
            if call() == b"":
                raise ConnectionResetError
    finally:
        tp.stop()
        srv.close()


def test_streamable_http_mcp_proxy(tmp_path):
    from mcp.client import Client
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("svc")

    @server.tool()
    def balance(user: str) -> dict:
        return {"user": user, "balance": 100}

    app = server.streamable_http_app()
    with BackgroundServer(app) as up:
        proxy = HttpMcpProxy("svc", f"{up.url}/mcp")
        ctl = Controller(Plan(faults=[FaultSpec("tool:svc.balance", "tool.wrong_result", 2)]),
                         tmp_path, "t")
        proxy.begin_trial(ctl, None)
        with BackgroundServer(proxy) as px:
            async def go():
                async with Client(f"{px.url}/mcp") as c:
                    a = await c.call_tool("balance", {"user": "ada"})
                    b = await c.call_tool("balance", {"user": "ada"})
                    return a, b
            a, b = asyncio.run(go())
    assert json.loads(a.content[0].text)["balance"] == 100
    assert json.loads(b.content[0].text)["balance"] != 100
    events = [e for e in read_trace(tmp_path) if e.op == "tools/call"]
    assert [e.faults for e in events] == [[], ["tool.wrong_result"]]
