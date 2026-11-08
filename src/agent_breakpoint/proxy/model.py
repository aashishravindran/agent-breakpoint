"""Model proxy: the agent's model base URL points here; we forward to the real provider.

Path-agnostic, so OpenAI-compatible, Anthropic, Ollama, Bedrock and Gemini REST APIs all
work. Responses are buffered (streams included) so faults can rewrite them and the record/replay
cache can serve them byte-for-byte.
"""

from __future__ import annotations

import asyncio
import json
import re
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from starlette.requests import Request
from starlette.responses import Response

from agent_breakpoint.config import Budgets, ModelBoundary, Pricing
from agent_breakpoint.controller import Controller
from agent_breakpoint.faults.base import Action, BoundaryKind, Kill
from agent_breakpoint.faults.http import HttpRequest, HttpResponse, clean_headers
from agent_breakpoint.faults.model import error_response, has_tool_call
from agent_breakpoint.proxy.http_actions import fail_response, realize, status_of, to_response
from agent_breakpoint.trace import Event, stable_hash


def detect_provider(path: str, headers: dict[str, str], configured: str = "auto") -> str:
    if configured != "auto":
        return configured
    lower = {k.lower(): v for k, v in headers.items()}
    if "anthropic-version" in lower or path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/api/"):
        return "ollama"
    if path.startswith("/model/") or "x-amz-date" in lower:
        return "bedrock"
    if ":generateContent" in path or ":streamGenerateContent" in path:
        return "gemini"
    return "openai"


_USAGE_PATTERNS = [
    (r'"prompt_tokens"\s*:\s*(\d+)', r'"completion_tokens"\s*:\s*(\d+)'),
    (r'"input_tokens"\s*:\s*(\d+)', r'"output_tokens"\s*:\s*(\d+)'),
    (r'"inputTokens"\s*:\s*(\d+)', r'"outputTokens"\s*:\s*(\d+)'),
    (r'"prompt_eval_count"\s*:\s*(\d+)', r'"eval_count"\s*:\s*(\d+)'),
    (r'"promptTokenCount"\s*:\s*(\d+)', r'"candidatesTokenCount"\s*:\s*(\d+)'),
]


def usage_of(body: bytes) -> tuple[int, int]:
    text = body.decode("utf-8", "ignore")
    for pin, pout in _USAGE_PATTERNS:
        ins = [int(x) for x in re.findall(pin, text)]
        outs = [int(x) for x in re.findall(pout, text)]
        if ins or outs:
            # Streams repeat cumulative counts; take the max rather than the sum.
            return (max(ins) if ins else 0, max(outs) if outs else 0)
    return 0, 0


def request_tags(req: HttpRequest) -> list[str]:
    data = req.json()
    tags = []
    if isinstance(data, dict):
        msgs = data.get("messages") or data.get("contents") or []
        if isinstance(msgs, list) and sum(1 for m in msgs if isinstance(m, dict)
                                          and m.get("role") not in ("system", "developer")) > 1:
            tags.append("multi_turn")
        if data.get("tools") or data.get("toolConfig"):
            tags.append("tools_offered")
    if req.wants_stream:
        tags.append("stream")
    return tags


# Provider-minted ids that differ on every run but carry no meaning for the model's answer.
_VOLATILE = {"id", "tool_call_id", "toolUseId", "tool_use_id", "created", "system_fingerprint"}


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def cache_key(req: HttpRequest) -> str:
    """Hash of what determines the model's answer: path + body (prompt, model, tool schemas),
    minus volatile ids, so the same conversation hashes the same across runs."""
    data = req.json()
    body = _normalize(data) if data is not None else req.body.hex()
    return stable_hash({"path": req.path.split("?")[0], "body": body})


@dataclass
class TrialUsage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    blown: str | None = None


@dataclass
class ModelCache:
    """Recorded baseline responses: (boundary, call index) -> (request hash, response)."""

    entries: dict[tuple[str, int], tuple[str, HttpResponse]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> ModelCache:
        cache = cls()
        if path and Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache.entries[(rec["boundary"], rec["call_index"])] = (
                        rec["req_hash"], HttpResponse.from_record(rec["response"]))
        return cache


class ModelProxy:
    def __init__(self, boundary: ModelBoundary, budgets: Budgets, pricing: Pricing) -> None:
        self.cfg = boundary
        self.boundary = f"model:{boundary.name}"
        self.budgets = budgets
        self.pricing = pricing
        self.controller: Controller | None = None
        self.cache = ModelCache()
        self.record_path: Path | None = None
        self.usage = TrialUsage()
        self.on_kill: Callable[[int], None] | None = None
        self._lock = threading.Lock()
        self._client: httpx.AsyncClient | None = None
        self._signer = None

    # Called by the runner before each trial.
    def begin_trial(self, controller: Controller, record_path: Path | None) -> None:
        self.controller = controller
        self.cache = ModelCache.load(controller.plan.replay_cache)
        self.record_path = record_path
        self.usage = TrialUsage()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        try:
            response = await self.handle(request)
            await response(scope, receive, send)
        except asyncio.CancelledError:  # client gave up on a hung call, or shutdown
            return

    async def handle(self, request: Request) -> Response:
        ctl = self.controller
        path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        req = HttpRequest(request.method, path, dict(request.headers), await request.body())
        provider = detect_provider(path, req.headers, self.cfg.provider)
        if ctl is None:  # proxy up but no trial running: plain passthrough
            return to_response(await self._forward(req, provider))

        idx = ctl.next_call(self.boundary)
        tags = request_tags(req)
        ev = Event(boundary=self.boundary, kind="model", call_index=idx, provider=provider,
                   op=req.method, tags=tags, req_hash=cache_key(req))
        started = time.time()

        blown = self._check_budget()
        if blown:
            ev.status, ev.detail = "error", {"budget": blown}
            ctl.record(ev)
            return fail_response(error_response(provider, 429, "budget_exceeded",
                                                f"agent-breakpoint budget cap reached: {blown}"))

        decisions = ctl.decide(self.boundary, BoundaryKind.MODEL, idx, provider=provider,
                               tags=tuple(tags))
        ev.faults = [d.fault.id for d in decisions]

        # Request-side hooks: rewrite the request or short-circuit.
        for d in decisions:
            if d.fault.before is None:
                continue
            out = d.fault.before(d.ctx, req)
            if isinstance(out, HttpRequest):
                req = out
            elif isinstance(out, Kill):
                sig = d.ctx.meta.pop("signal", signal.SIGKILL)
                ev.status = "killed"
                ctl.record(ev)
                if self.on_kill:
                    self.on_kill(int(sig))
                return fail_response(error_response(provider, 503, "interrupted", "interrupted"))
            elif isinstance(out, Action):
                ev.status = status_of(out)
                ctl.record(ev)  # recorded before a hang so the trace shows it
                resp_or_none = await realize(out)
                if resp_or_none is not None:
                    return resp_or_none

        # Fork-at-fault replay: serve the cached baseline prefix when it still applies.
        resp: HttpResponse | None = None
        until = ctl.plan.replay_until.get(self.boundary, 0)
        cached = self.cache.entries.get((self.boundary, idx))
        if cached and idx <= until and not decisions:
            if cached[0] == ev.req_hash:
                resp, ev.status = cached[1], "replayed"
            else:
                ev.detail["replay_miss"] = True
        if resp is None:
            try:
                resp = await self._forward(req, provider)
            except httpx.HTTPError as exc:
                ev.status, ev.detail["upstream_error"] = "error", repr(exc)
                ctl.record(ev)
                return fail_response(error_response(provider, 502, "bad_gateway", repr(exc)))
            if ctl.plan.mode == "record" and self.record_path:
                self._record(idx, ev.req_hash or "", resp)

        if has_tool_call(resp):
            ev.tags.append("tool_call")

        # Response-side hooks.
        for d in decisions:
            if d.fault.after is None:
                continue
            out = d.fault.after(d.ctx, req, resp)
            if isinstance(out, HttpResponse):
                resp = out
            elif isinstance(out, Action):
                ev.status = status_of(out)
                ctl.record(ev)
                r = await realize(out)
                if r is not None:
                    return r

        ev.tokens_in, ev.tokens_out = usage_of(resp.body)
        if ev.status == "ok" and resp.status >= 400:
            ev.status = "error"
        ev.latency_ms = (time.time() - started) * 1000
        ev.detail.setdefault("http_status", resp.status)
        self._account(ev)
        ctl.record(ev)
        return to_response(resp)

    def _check_budget(self) -> str | None:
        u, b = self.usage, self.budgets
        if u.blown:
            return u.blown
        if u.calls >= b.max_model_calls:
            u.blown = f"max_model_calls={b.max_model_calls}"
        elif b.max_tokens is not None and u.tokens_in + u.tokens_out >= b.max_tokens:
            u.blown = f"max_tokens={b.max_tokens}"
        elif u.usd > b.max_usd:
            u.blown = f"max_usd={b.max_usd}"
        return u.blown

    def _account(self, ev: Event) -> None:
        with self._lock:
            self.usage.calls += 1
            if ev.status == "replayed":
                return  # replayed prefix costs nothing
            self.usage.tokens_in += ev.tokens_in
            self.usage.tokens_out += ev.tokens_out
            cost = (ev.tokens_in * self.pricing.input_per_mtok
                    + ev.tokens_out * self.pricing.output_per_mtok) / 1e6
            self.usage.usd += cost
            ev.detail["usd"] = round(cost, 6)

    def _record(self, idx: int, req_hash: str, resp: HttpResponse) -> None:
        assert self.record_path is not None
        rec = {"boundary": self.boundary, "call_index": idx, "req_hash": req_hash,
               "response": resp.to_record()}
        with self._lock, self.record_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    async def _forward(self, req: HttpRequest, provider: str) -> HttpResponse:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        url = self.cfg.upstream.rstrip("/") + req.path
        headers = clean_headers(req.headers)
        if provider == "bedrock":
            headers = self._resign(req.method, url, headers, req.body)
        r = await self._client.request(req.method, url, headers=headers, content=req.body)
        return HttpResponse(r.status_code, dict(r.headers), r.content)

    def _resign(self, method: str, url: str, headers: dict[str, str], body: bytes) -> dict[str, str]:
        """SigV4 signs for the host the client called (us), so re-sign for the real host with
        local AWS credentials."""
        from botocore.auth import SigV4Auth  # type: ignore[import-not-found]
        from botocore.awsrequest import AWSRequest  # type: ignore[import-not-found]
        from botocore.session import Session  # type: ignore[import-not-found]

        if self._signer is None:
            region = self.cfg.aws_region or _region_from_host(urlsplit(url).hostname or "")
            creds = Session().get_credentials()
            if creds is None:
                raise RuntimeError("bedrock proxying needs local AWS credentials")
            self._signer = SigV4Auth(creds, "bedrock", region)
        drop = {"authorization", "x-amz-date", "x-amz-security-token", "x-amz-content-sha256"}
        base = {k: v for k, v in headers.items() if k.lower() not in drop}
        aws_req = AWSRequest(method=method, url=url, data=body, headers=base)
        self._signer.add_auth(aws_req)
        return dict(aws_req.headers.items())


def _region_from_host(host: str) -> str:
    m = re.search(r"\.([a-z]{2}-[a-z]+-\d)\.", host)
    return m.group(1) if m else "us-east-1"
