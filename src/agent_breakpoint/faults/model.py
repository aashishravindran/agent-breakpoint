"""Model boundary faults: timeouts, rate limits, 5xx, cut streams, malformed tool calls."""

from __future__ import annotations

import json
from typing import Any

from agent_breakpoint.faults.base import (
    BoundaryKind,
    Fail,
    Fault,
    FaultClass,
    FaultContext,
    Hang,
    Pack,
)
from agent_breakpoint.faults.http import HttpRequest, HttpResponse

# --- provider-shaped error bodies -----------------------------------------------------------


def error_response(provider: str | None, status: int, kind: str, message: str) -> Fail:
    headers: dict[str, str] = {"content-type": "application/json"}
    if status == 429:
        headers["retry-after"] = "1"
    if provider == "anthropic":
        etype = {429: "rate_limit_error", 529: "overloaded_error"}.get(status, "api_error")
        body: Any = {"type": "error", "error": {"type": etype, "message": message}}
    elif provider == "bedrock":
        etype = {429: "ThrottlingException", 503: "ServiceUnavailableException"}.get(
            status, "InternalServerException")
        headers["x-amzn-errortype"] = etype
        body = {"message": message}
    elif provider == "ollama":
        body = {"error": message}
    elif provider == "gemini":
        gstatus = {429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE"}.get(status, "INTERNAL")
        body = {"error": {"code": status, "message": message, "status": gstatus}}
    else:  # openai-compatible
        body = {"error": {"message": message, "type": kind, "code": kind}}
    return Fail(status=status, body=body, headers=headers)


def _timeout(ctx: FaultContext, req: HttpRequest) -> Hang:
    return Hang(seconds=float(ctx.params.get("seconds", 3600)))


def _rate_limit(ctx: FaultContext, req: HttpRequest) -> Fail:
    return error_response(ctx.provider, 429, "rate_limit_exceeded",
                          "Rate limit reached (injected by agent-breakpoint)")


def _server_error(ctx: FaultContext, req: HttpRequest) -> Fail:
    status = int(ctx.params.get("status", ctx.rng.choice([500, 502, 503])))
    if ctx.provider == "anthropic" and ctx.params.get("overloaded"):
        status = 529
    return error_response(ctx.provider, status, "server_error",
                          "Upstream error (injected by agent-breakpoint)")


# --- response mutations ---------------------------------------------------------------------


def _split_events(resp: HttpResponse) -> tuple[list[bytes], bytes]:
    if resp.is_sse:
        return [e + b"\n\n" for e in resp.body.split(b"\n\n") if e.strip()], b"\n\n"
    return [line + b"\n" for line in resp.body.split(b"\n") if line.strip()], b"\n"


def _stream_cut(ctx: FaultContext, req: HttpRequest, resp: HttpResponse) -> HttpResponse:
    """Cut the response mid-way. Streams lose their tail events (no [DONE] / message_stop);
    buffered bodies are truncated mid-JSON."""
    frac = float(ctx.params.get("fraction", 0.5))
    if resp.is_sse or resp.is_ndjson or resp.is_eventstream:
        if resp.is_eventstream:
            return resp.with_body(resp.body[: max(1, int(len(resp.body) * frac))])
        events, _ = _split_events(resp)
        keep = max(1, int(len(events) * frac))
        return resp.with_body(b"".join(events[:keep]))
    return resp.with_body(resp.body[: max(1, int(len(resp.body) * frac))])


def _truncate(s: str) -> str:
    return s[: max(1, len(s) // 2)]


def _malformed_tool_call(ctx: FaultContext, req: HttpRequest, resp: HttpResponse) -> HttpResponse | None:
    """Break the arguments of the first tool call so they no longer parse."""
    if resp.is_sse or resp.is_ndjson:
        return _malformed_stream(resp)
    data = resp.json()
    if not isinstance(data, dict):
        return None
    hit = False
    for choice in data.get("choices") or []:  # openai
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = _truncate(fn["arguments"])
                hit = True
                break
    content = data.get("content")
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use":  # anthropic
            block["input"] = _truncate(json.dumps(block.get("input", {})))
            hit = True
            break
    msg = data.get("message")
    if isinstance(msg, dict):  # ollama native
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            fn["arguments"] = _truncate(json.dumps(fn.get("arguments", {})))
            hit = True
            break
    out = data.get("output")
    if isinstance(out, dict):  # bedrock converse
        for block in (out.get("message") or {}).get("content") or []:
            if "toolUse" in block:
                block["toolUse"]["input"] = _truncate(json.dumps(block["toolUse"].get("input", {})))
                hit = True
                break
    for cand in data.get("candidates") or []:  # gemini
        for part in (cand.get("content") or {}).get("parts") or []:
            if "functionCall" in part:
                part["functionCall"]["args"] = _truncate(json.dumps(part["functionCall"].get("args", {})))
                hit = True
                break
    return resp.with_json(data) if hit else None


def _malformed_stream(resp: HttpResponse) -> HttpResponse | None:
    """Drop the last streamed fragment of tool-call arguments, so the assembled JSON is cut."""
    events, _ = _split_events(resp)
    markers = (b'"arguments"', b"input_json_delta", b'"partial_json"')
    idx = [i for i, e in enumerate(events) if any(m in e for m in markers)]
    if not idx:
        return None
    del events[idx[-1]]
    return resp.with_body(b"".join(events))


def has_tool_call(resp: HttpResponse) -> bool:
    body = resp.body
    return any(m in body for m in (b'"tool_calls"', b'"tool_use"', b'"toolUse"', b'"functionCall"'))


PACK = Pack(
    name="model",
    description="Crash, omission and wrong-value faults on the model API",
    faults=[
        Fault("model.timeout", BoundaryKind.MODEL, FaultClass.OMISSION,
              "Model call never answers", "Client timeouts and retries", before=_timeout),
        Fault("model.rate_limit", BoundaryKind.MODEL, FaultClass.EXHAUSTION,
              "429 with retry-after", "Backoff", before=_rate_limit),
        Fault("model.server_error", BoundaryKind.MODEL, FaultClass.CRASH,
              "5xx from the provider", "Retries", before=_server_error),
        Fault("model.stream_cut", BoundaryKind.MODEL, FaultClass.OMISSION,
              "Response cut mid-way", "Partial-output handling", after=_stream_cut),
        Fault("model.malformed_tool_call", BoundaryKind.MODEL, FaultClass.WRONG_VALUE,
              "Tool-call arguments truncated so they no longer parse", "Validation and repair",
              after=_malformed_tool_call, requires=("tool_call",)),
    ],
)
