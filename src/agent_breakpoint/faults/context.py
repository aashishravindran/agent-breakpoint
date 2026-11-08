"""Context faults: rewrite the request the agent sends to the model."""

from __future__ import annotations

from typing import Any

from agent_breakpoint.faults.base import BoundaryKind, Fault, FaultClass, FaultContext, Pack
from agent_breakpoint.faults.http import HttpRequest

FILLER = (
    "agent-breakpoint context filler. This paragraph pads the prompt to test instruction "
    "retention under pressure; it carries no task information. "
)


def _messages(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    msgs = data.get("messages")
    return msgs if isinstance(msgs, list) else None


def _text_append(msg: dict[str, Any], extra: str) -> None:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = content + "\n\n" + extra
    elif isinstance(content, list):
        # anthropic/openai content blocks, bedrock converse {"text": ...}
        if content and isinstance(content[0], dict) and "type" in content[0]:
            content.append({"type": "text", "text": extra})
        else:
            content.append({"text": extra})
    else:
        msg["content"] = extra


def _oversized(ctx: FaultContext, req: HttpRequest) -> HttpRequest | None:
    data = req.json()
    if not isinstance(data, dict):
        return None
    chars = int(ctx.params.get("chars", 200_000))
    filler = (FILLER * (chars // len(FILLER) + 1))[:chars]
    msgs = _messages(data)
    if msgs:
        users = [m for m in msgs if m.get("role") == "user"]
        target = users[-1] if users else msgs[-1]
        _text_append(target, filler)
    elif isinstance(data.get("contents"), list) and data["contents"]:  # gemini
        data["contents"][-1].setdefault("parts", []).append({"text": filler})
    else:
        return None
    return req.with_json(data)


def _drop_system(ctx: FaultContext, req: HttpRequest) -> HttpRequest | None:
    data = req.json()
    if not isinstance(data, dict):
        return None
    changed = False
    for key in ("system", "systemInstruction", "system_instruction"):  # anthropic, bedrock, gemini
        if key in data:
            del data[key]
            changed = True
    msgs = _messages(data)
    if msgs:
        kept = [m for m in msgs if m.get("role") not in ("system", "developer")]
        if len(kept) != len(msgs):
            data["messages"] = kept
            changed = True
    return req.with_json(data) if changed else None


def _truncate_history(ctx: FaultContext, req: HttpRequest) -> HttpRequest | None:
    data = req.json()
    if not isinstance(data, dict):
        return None
    msgs = _messages(data)
    keep = int(ctx.params.get("keep", 1))
    if not msgs or len(msgs) <= keep + 1:
        return None
    system = [m for m in msgs if m.get("role") in ("system", "developer")]
    rest = [m for m in msgs if m.get("role") not in ("system", "developer")]
    tail = rest[-keep:]
    # Never start on a tool result without its call; providers reject that, which would test
    # the provider's validation instead of the agent.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    if not tail:
        return None
    data["messages"] = system + tail
    return req.with_json(data)


PACK = Pack(
    name="context",
    description="Instruction retention under pressure",
    faults=[
        Fault("context.oversized_prompt", BoundaryKind.CONTEXT, FaultClass.EXHAUSTION,
              "~200k chars of filler appended to the last user turn", "Instruction retention",
              before=_oversized),
        Fault("context.dropped_system_prompt", BoundaryKind.CONTEXT, FaultClass.OMISSION,
              "System prompt removed from the request", "Instruction retention",
              before=_drop_system),
        Fault("context.truncated_history", BoundaryKind.CONTEXT, FaultClass.OMISSION,
              "Only the last turn of history is sent", "Instruction retention",
              before=_truncate_history, requires=("multi_turn",)),
    ],
)
