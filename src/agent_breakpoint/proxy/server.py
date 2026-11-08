"""Run an ASGI app on an ephemeral localhost port in a background thread."""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import uvicorn


class BackgroundServer:
    def __init__(self, app, host: str = "127.0.0.1", port: int = 0) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.host, self.port = self.sock.getsockname()[:2]
        config = uvicorn.Config(app, log_level="error", lifespan="auto", timeout_graceful_shutdown=1)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"abk-proxy-{self.port}")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _run(self) -> None:
        asyncio.run(self.server.serve(sockets=[self.sock]))

    def start(self) -> BackgroundServer:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("proxy failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.server.force_exit = True
        self.thread.join(timeout=5)

    def __enter__(self) -> BackgroundServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
