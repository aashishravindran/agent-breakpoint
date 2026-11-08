"""Turn fault actions into ASGI responses, shared by the model and HTTP dependency proxies."""

from __future__ import annotations

import asyncio
import json

from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from agent_breakpoint.faults.base import Action, Delay, Drop, Fail, Hang
from agent_breakpoint.faults.http import HttpResponse, clean_headers


class ResetResponse(Response):
    """Announce a body, send half of nothing, then abort: the client sees a broken connection."""

    def __init__(self) -> None:
        super().__init__(status_code=200)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-length", b"1048576"), (b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"partial', "more_body": True})
        raise ConnectionResetError("connection reset (injected by agent-breakpoint)")


def to_response(resp: HttpResponse) -> Response:
    return Response(content=resp.body, status_code=resp.status, headers=clean_headers(resp.headers))


def fail_response(action: Fail) -> Response:
    body = action.body
    if isinstance(body, (dict, list)):
        content = json.dumps(body).encode()
        headers = {"content-type": "application/json", **action.headers}
    else:
        content = (body or "").encode() if isinstance(body, str) else (body or b"")
        headers = dict(action.headers)
    return Response(content=content, status_code=action.status, headers=headers)


async def realize(action: Action) -> Response | None:
    """Apply an action. Returns a response to short-circuit with, or None to continue."""
    if isinstance(action, Delay):
        await asyncio.sleep(action.seconds)
        return None
    if isinstance(action, Hang):
        await asyncio.sleep(action.seconds)
        return Response(status_code=504)
    if isinstance(action, Fail):
        return fail_response(action)
    if isinstance(action, Drop):
        return ResetResponse()
    return None


def status_of(action: Action) -> str:
    return {Delay: "ok", Hang: "hang", Fail: "error", Drop: "drop"}.get(type(action), "error")
