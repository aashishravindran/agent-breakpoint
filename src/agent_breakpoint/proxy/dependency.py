"""Dependency proxies: sit between a tool server and what it depends on (policy engine, JWKS,
database, upstream API). HTTP dependencies get full request/response faults; anything else
gets TCP-level faults (latency, timeout, reset)."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from urllib.parse import urlsplit

import httpx
from starlette.requests import Request

from agent_breakpoint.config import DependencyBoundary
from agent_breakpoint.controller import Controller
from agent_breakpoint.faults.base import Action, BoundaryKind, Delay, Drop, Hang
from agent_breakpoint.faults.http import HttpRequest, HttpResponse, clean_headers
from agent_breakpoint.proxy.http_actions import realize, status_of, to_response
from agent_breakpoint.trace import Event, stable_hash


class HttpDependencyProxy:
    def __init__(self, cfg: DependencyBoundary) -> None:
        self.cfg = cfg
        self.boundary = f"dep:{cfg.name}"
        self.controller: Controller | None = None
        self._client: httpx.AsyncClient | None = None
        self._seen: set[str] = set()

    def begin_trial(self, controller: Controller) -> None:
        self.controller = controller
        self._seen = set()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        req = HttpRequest(request.method, path, dict(request.headers), await request.body())
        try:
            response = await self.handle(req)
            await response(scope, receive, send)
        except asyncio.CancelledError:
            return

    async def handle(self, req: HttpRequest):
        ctl = self.controller
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        if ctl is None:
            return to_response(await self._forward(req))
        idx = ctl.next_call(self.boundary)
        key = f"{req.method} {req.path.split('?')[0]}"
        tags = ["repeat"] if key in self._seen else []
        self._seen.add(key)
        ev = Event(boundary=self.boundary, kind="dependency", call_index=idx, op="http",
                   tags=tags, req_hash=stable_hash(key), detail={"endpoint": key})
        started = time.time()
        decisions = ctl.decide(self.boundary, BoundaryKind.DEPENDENCY, idx, op="http")
        ev.faults = [d.fault.id for d in decisions]
        for d in decisions:
            if d.fault.before is None:
                continue
            out = d.fault.before(d.ctx, req)
            if isinstance(out, Action):
                ev.status = status_of(out)
                if not isinstance(out, Delay):
                    ctl.record(ev)
                r = await realize(out)
                if r is not None:
                    return r
        try:
            resp = await self._forward(req)
        except httpx.HTTPError as exc:
            ev.status, ev.detail["upstream_error"] = "error", repr(exc)
            ctl.record(ev)
            return to_response(HttpResponse(502, {}, b""))
        firsts = ctl.meta[self.boundary].setdefault("firsts", {})
        firsts.setdefault(key, resp)
        for d in decisions:
            if d.fault.after is None:
                continue
            d.ctx.meta["first_response"] = firsts.get(key)
            out = d.fault.after(d.ctx, req, resp)
            if isinstance(out, HttpResponse):
                resp = out
        ev.latency_ms = (time.time() - started) * 1000
        if resp.status >= 400 and ev.status == "ok":
            ev.status = "error"
        ev.detail["http_status"] = resp.status
        ctl.record(ev)
        return to_response(resp)

    async def _forward(self, req: HttpRequest) -> HttpResponse:
        assert self._client is not None
        url = self.cfg.upstream.rstrip("/") + req.path
        r = await self._client.request(req.method, url, headers=clean_headers(req.headers),
                                       content=req.body)
        return HttpResponse(r.status_code, dict(r.headers), r.content)


class TcpDependencyProxy:
    """Byte-level proxy. Each accepted connection is one call on the boundary."""

    def __init__(self, cfg: DependencyBoundary) -> None:
        self.cfg = cfg
        self.boundary = f"dep:{cfg.name}"
        self.controller: Controller | None = None
        parts = urlsplit(cfg.upstream if "://" in cfg.upstream else f"tcp://{cfg.upstream}")
        self.up_host, self.up_port = parts.hostname or "127.0.0.1", parts.port or 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", cfg.listen_port or 0))
        self.port = self.sock.getsockname()[1]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.base_events.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"127.0.0.1:{self.port}"

    def begin_trial(self, controller: Controller) -> None:
        self.controller = controller

    def start(self) -> TcpDependencyProxy:
        ready = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def main() -> None:
                self._server = await asyncio.start_server(self._handle, sock=self.sock)
                ready.set()
                async with self._server:
                    await self._server.serve_forever()

            try:
                self._loop.run_until_complete(main())
            except (asyncio.CancelledError, RuntimeError):
                pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(10)
        return self

    def stop(self) -> None:
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._server.close)
            for task in asyncio.all_tasks(self._loop):
                self._loop.call_soon_threadsafe(task.cancel)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        ctl = self.controller
        decisions = []
        ev = None
        if ctl is not None:
            idx = ctl.next_call(self.boundary)
            ev = Event(boundary=self.boundary, kind="dependency", call_index=idx, op="tcp")
            decisions = ctl.decide(self.boundary, BoundaryKind.DEPENDENCY, idx, op="tcp")
            ev.faults = [d.fault.id for d in decisions]
        try:
            for d in decisions:
                if d.fault.before is None:
                    continue
                out = d.fault.before(d.ctx, None)
                if isinstance(out, Delay):
                    await asyncio.sleep(out.seconds)
                elif isinstance(out, Hang):
                    ev.status = "hang"  # type: ignore[union-attr]
                    ctl.record(ev)  # type: ignore[union-attr]
                    await asyncio.sleep(out.seconds)
                    return
                elif isinstance(out, (Drop, Action)):
                    ev.status = "drop"  # type: ignore[union-attr]
                    ctl.record(ev)  # type: ignore[union-attr]
                    sock = writer.get_extra_info("socket")
                    if sock is not None:  # RST rather than FIN
                        linger_zero = b"\x01\x00\x00\x00\x00\x00\x00\x00"
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger_zero)
                    writer.close()
                    return
            up_reader, up_writer = await asyncio.open_connection(self.up_host, self.up_port)
            if ev is not None and ctl is not None:
                ctl.record(ev)

            async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
                try:
                    while data := await src.read(65536):
                        dst.write(data)
                        await dst.drain()
                finally:
                    dst.close()

            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer),
                                 return_exceptions=True)
        except OSError:
            writer.close()
