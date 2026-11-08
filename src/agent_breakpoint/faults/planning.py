"""Planning and tool-selection faults, injected live on the model boundary.

They fill the AgentFault taxonomy rows the catalog lacked: the model skips a step it needed
(PLAN_MISSING_STEP / CTRL_PREMATURE_TERMINATION), calls the wrong tool (TOOL_WRONG_TOOL), or
calls the right tool with wrong arguments (TOOL_WRONG_ARGUMENT); and the model sees only part
of what a tool or retrieval returned (KNOW_CONTEXT_TRUNCATION). Each rewrites one response or
request, so a later step has to notice.
"""

from __future__ import annotations

import json
from typing import Any

from agent_breakpoint.faults.base import BoundaryKind, Fault, FaultClass, FaultContext, Pack
from agent_breakpoint.faults.http import HttpRequest, HttpResponse

DONE_TEXT = "All done: the task is complete."


def _offered_tools(req: HttpRequest) -> list[str]:
    data = req.json() or {}
    names = []
    for t in data.get("tools") or []:
        if isinstance(t, dict):
            names.append((t.get("function") or {}).get("name") or t.get("name")
                         or (t.get("toolSpec") or {}).get("name"))
    return [n for n in names if n]


def _perturb(value: Any, rng) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + rng.choice([1, 7, 100])
    if isinstance(value, float):
        return round(value * rng.choice([0.5, 2.0, 10.0]), 4)
    if isinstance(value, str):
        return value[::-1] if len(value) > 1 else value + "x"
    if isinstance(value, dict):
        return {k: _perturb(v, rng) for k, v in value.items()}
    if isinstance(value, list):
        return list(reversed(value))
    return value


def _tool_calls(data: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    """(container, name key, args key) for each tool call, across provider formats.
    OpenAI/Ollama keep args under function.arguments (OpenAI as a JSON string), Anthropic under
    content[].input, Bedrock under content[].toolUse.input."""
    out = []
    for choice in data.get("choices") or []:
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            out.append((call.get("function") or {}, "name", "arguments"))
    msg = data.get("message")
    if isinstance(msg, dict):
        for call in msg.get("tool_calls") or []:
            out.append((call.get("function") or {}, "name", "arguments"))
    content = data.get("content")
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            out.append((block, "name", "input"))
    for block in ((data.get("output") or {}).get("message") or {}).get("content") or []:
        if isinstance(block, dict) and "toolUse" in block:
            out.append((block["toolUse"], "name", "input"))
    return out


def _wrong_tool(ctx: FaultContext, req: HttpRequest, resp: HttpResponse) -> HttpResponse | None:
    data = resp.json()
    calls = _tool_calls(data) if isinstance(data, dict) else []
    if not calls:
        return None
    holder, name_key, _ = calls[0]
    others = [t for t in _offered_tools(req) if t != holder.get(name_key)]
    holder[name_key] = ctx.rng.choice(others) if others else f"{holder.get(name_key)}_v2"
    ctx.meta["swapped_to"] = holder[name_key]
    return resp.with_json(data)


def _wrong_args(ctx: FaultContext, req: HttpRequest, resp: HttpResponse) -> HttpResponse | None:
    data = resp.json()
    calls = _tool_calls(data) if isinstance(data, dict) else []
    if not calls:
        return None
    holder, _, args_key = calls[0]
    raw = holder.get(args_key)
    was_str = isinstance(raw, str)
    try:
        args = json.loads(raw) if was_str else raw
    except ValueError:
        return None
    if not isinstance(args, dict) or not args:
        return None
    args = _perturb(args, ctx.rng)
    holder[args_key] = json.dumps(args) if was_str else args
    return resp.with_json(data)


def _skipped_step(ctx: FaultContext, req: HttpRequest, resp: HttpResponse) -> HttpResponse | None:
    """The model declares victory instead of making the tool call it needed."""
    data = resp.json()
    if not isinstance(data, dict) or not _tool_calls(data):
        return None
    for choice in data.get("choices") or []:  # openai
        choice["message"] = {"role": "assistant", "content": DONE_TEXT}
        choice["finish_reason"] = "stop"
    if isinstance(data.get("message"), dict):  # ollama
        data["message"] = {"role": "assistant", "content": DONE_TEXT}
    if isinstance(data.get("content"), list):  # anthropic
        data["content"] = [{"type": "text", "text": DONE_TEXT}]
        data["stop_reason"] = "end_turn"
    out = data.get("output")
    if isinstance(out, dict) and isinstance(out.get("message"), dict):  # bedrock
        out["message"]["content"] = [{"text": DONE_TEXT}]
        data["stopReason"] = "end_turn"
    return resp.with_json(data)


def _truncate_tool_result(ctx: FaultContext, req: HttpRequest) -> HttpRequest | None:
    """The model sees only the first part of the latest tool or retrieval result."""
    data = req.json()
    msgs = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(msgs, list):
        return None
    frac = float(ctx.params.get("fraction", 0.3))
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 8:
            m["content"] = m["content"][: max(4, int(len(m["content"]) * frac))]
            return req.with_json(data)
        for block in m.get("content") if isinstance(m.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str) and len(inner) > 8:
                    block["content"] = inner[: max(4, int(len(inner) * frac))]
                    return req.with_json(data)
    return None


PACK = Pack(
    name="planning",
    description="Planning, tool-selection and context-truncation faults (AgentFault taxonomy rows)",
    faults=[
        Fault("model.skipped_step", BoundaryKind.MODEL, FaultClass.OMISSION,
              "Model answers 'done' instead of making the tool call it needed",
              "Plan completion vs the end state", after=_skipped_step, requires=("tool_call",)),
        Fault("model.wrong_tool", BoundaryKind.MODEL, FaultClass.WRONG_VALUE,
              "Tool call renamed to another offered tool", "Tool selection validation",
              after=_wrong_tool, requires=("tool_call",)),
        Fault("model.wrong_arguments", BoundaryKind.MODEL, FaultClass.WRONG_VALUE,
              "Tool-call arguments perturbed (numbers, strings, flags)", "Argument validation",
              after=_wrong_args, requires=("tool_call",)),
        Fault("context.truncated_tool_result", BoundaryKind.CONTEXT, FaultClass.OMISSION,
              "Latest tool or retrieval result cut to 30% before the model sees it",
              "Acting on partial knowledge", before=_truncate_tool_result, requires=("multi_turn",)),
    ],
)
