import json

from agent_breakpoint.controller import FaultSpec
from agent_breakpoint.faults import BoundaryKind, Fail, FaultContext, Hang, Reply, registry, rng_for
from agent_breakpoint.faults.http import HttpRequest, HttpResponse
from agent_breakpoint.faults.model import has_tool_call


def ctx(kind=BoundaryKind.MODEL, provider="openai", **params):
    return FaultContext("model:llm", kind, 1, rng_for(1, "model:llm", 1), params, provider)


def test_rng_is_deterministic_per_boundary_and_call():
    a = rng_for(7, "tool:x.y", 3).random()
    assert a == rng_for(7, "tool:x.y", 3).random()
    assert a != rng_for(7, "tool:x.y", 4).random()
    assert a != rng_for(8, "tool:x.y", 3).random()


def test_fault_spec_matching():
    s = FaultSpec("tool:bank.*", "tool.timeout", at_call=2)
    assert s.matches("tool:bank.refund", 2) and not s.matches("tool:bank.refund", 3)
    sticky = FaultSpec("model:llm", "budget.quota_exhausted", at_call=2, sticky=True)
    assert sticky.matches("model:llm", 5) and not sticky.matches("model:llm", 1)
    ep = FaultSpec("model:llm", "model.timeout", at_call=1, session=2)
    assert ep.matches("model:llm", 1, session=2) and not ep.matches("model:llm", 1, session=1)


def test_catalog_covers_every_boundary_kind():
    kinds = {f.kind for f in registry().faults.values()}
    assert kinds == set(BoundaryKind)
    assert len(registry().faults) >= 35


def test_rate_limit_bodies_are_provider_shaped():
    f = registry().get("model.rate_limit")
    req = HttpRequest("POST", "/v1/messages", {}, b"{}")
    out = f.before(ctx(provider="anthropic"), req)
    assert isinstance(out, Fail) and out.status == 429
    assert out.body["error"]["type"] == "rate_limit_error"
    out = f.before(ctx(provider="bedrock"), req)
    assert out.headers["x-amzn-errortype"] == "ThrottlingException"


def test_malformed_tool_call_openai_and_anthropic():
    f = registry().get("model.malformed_tool_call")
    openai = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "refund", "arguments": json.dumps({"order_id": 42, "amount": 19.99})}}]}}]}
    resp = HttpResponse(200, {"content-type": "application/json"}, json.dumps(openai).encode())
    assert has_tool_call(resp)
    out = f.after(ctx(), HttpRequest("POST", "/v1/chat/completions"), resp)
    args = out.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    try:
        json.loads(args)
        raise AssertionError("arguments still parse")
    except ValueError:
        pass
    anth = {"content": [{"type": "tool_use", "name": "refund", "input": {"order_id": 42}}]}
    resp = HttpResponse(200, {"content-type": "application/json"}, json.dumps(anth).encode())
    out = f.after(ctx(provider="anthropic"), HttpRequest("POST", "/v1/messages"), resp)
    assert isinstance(out.json()["content"][0]["input"], str)


def test_stream_cut_drops_tail_events():
    f = registry().get("model.stream_cut")
    body = b"".join(f"data: {{\"i\": {i}}}\n\n".encode() for i in range(10)) + b"data: [DONE]\n\n"
    resp = HttpResponse(200, {"content-type": "text/event-stream"}, body)
    out = f.after(ctx(), HttpRequest("POST", "/"), resp)
    assert b"[DONE]" not in out.body and out.body.count(b"data:") < 11


def test_context_faults_rewrite_requests():
    req = HttpRequest("POST", "/v1/chat/completions", {}, json.dumps({"messages": [
        {"role": "system", "content": "rules"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x"}, {"role": "user", "content": "again"}]}).encode())
    dropped = registry().get("context.dropped_system_prompt").before(ctx(), req)
    assert all(m["role"] != "system" for m in dropped.json()["messages"])
    trunc = registry().get("context.truncated_history").before(ctx(), req)
    assert [m["role"] for m in trunc.json()["messages"]] == ["system", "user"]
    big = registry().get("context.oversized_prompt").before(ctx(chars=5000), req)
    assert len(big.body) > len(req.body) + 4000


def test_tool_faults():
    reg = registry()
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_order"}}
    resp = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps(
        {"order_id": 42, "amount": 19.99, "items": [1, 2, 3, 4]})}], "isError": False}}
    tctx = FaultContext("tool:bank.get_order", BoundaryKind.TOOL, 1, rng_for(1, "t", 1))
    assert isinstance(reg.get("tool.timeout").before(tctx, req), Hang)
    assert isinstance(reg.get("tool.error_in_200").before(tctx, req), Reply)
    wrong = reg.get("tool.wrong_result").after(tctx, req, resp)
    assert json.loads(wrong["result"]["content"][0]["text"])["amount"] != 19.99
    partial = reg.get("tool.partial_result").after(tctx, req, resp)
    assert len(json.loads(partial["result"]["content"][0]["text"])["items"]) == 2
    drift = reg.get("tool.schema_drift").after(tctx, req, resp)
    assert "amount_v2" in drift["result"]["content"][0]["text"]
    inj = reg.get("tool.injected_instructions").after(tctx, req, resp)
    assert tctx.meta["canary"].startswith("ABK-INJECT-")
    assert tctx.meta["canary"] in inj["result"]["content"][-1]["text"]
    assert resp["result"]["content"][0]["text"].count("19.99") == 1  # original untouched


def test_planning_faults():
    from agent_breakpoint.faults.taxonomy import family

    reg = registry()
    req = HttpRequest("POST", "/v1/chat/completions", {}, json.dumps({
        "tools": [{"type": "function", "function": {"name": "get_order"}},
                  {"type": "function", "function": {"name": "refund"}}],
        "messages": [{"role": "user", "content": "refund 42"},
                     {"role": "tool", "tool_call_id": "c1",
                      "content": '{"order_id": 42, "amount": 19.99}'}]}).encode())
    body = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "refund", "arguments": json.dumps({"order_id": 42, "amount": 19.99})}}]}}]}
    resp = HttpResponse(200, {"content-type": "application/json"}, json.dumps(body).encode())
    swapped = reg.get("model.wrong_tool").after(ctx(), req, resp).json()
    assert swapped["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_order"
    args = json.loads(reg.get("model.wrong_arguments").after(ctx(), req, resp).json()
                      ["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args != {"order_id": 42, "amount": 19.99} and set(args) == {"order_id", "amount"}
    done = reg.get("model.skipped_step").after(ctx(), req, resp).json()
    assert "tool_calls" not in done["choices"][0]["message"]
    cut = reg.get("context.truncated_tool_result").before(ctx(), req).json()
    assert len(cut["messages"][1]["content"]) < 15
    unmapped = [f for f in reg.faults if family(f) == "Unmapped"]
    assert not unmapped, unmapped
