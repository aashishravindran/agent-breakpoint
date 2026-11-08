"""A LangGraph refund workflow with a checkpointer and file memory; no model, so it's fast and
deterministic. Session 1 pays and remembers; session 2 answers from memory.

The ``pay`` node is deliberately not idempotent: if the process dies after paying but before
the checkpoint lands, resuming the thread pays again. agent-breakpoint's crash fault finds that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent_breakpoint import Outcome, Sandbox, Task, tap


class State(TypedDict, total=False):
    order_id: int
    question: str
    memory: dict
    paid: bool
    answer: str


def build(sandbox: Sandbox, idempotent: bool = False):
    memdir = Path(tap(sandbox.root / "memory", name="memory"))
    ledger = sandbox.path("ledger.json")

    def recall(state: State) -> State:
        notes: dict = {}
        for f in sorted(memdir.glob("*")):
            if f.suffix == ".json":
                notes.update(json.loads(f.read_text()))  # torn JSON raises: a loud failure
            elif f.suffix == ".md":
                notes.setdefault("notes", []).append(f.read_text())
        return {"memory": notes}

    def pay(state: State) -> State:
        if state.get("question"):
            return {}
        entries = json.loads(ledger.read_text()) if ledger.exists() else []
        if idempotent and any(e["order_id"] == state["order_id"] for e in entries):
            return {"paid": True}
        entries.append({"order_id": state["order_id"], "amount": 19.99})
        ledger.write_text(json.dumps(entries))  # the side effect
        return {"paid": True}

    def remember(state: State) -> State:
        if state.get("paid"):
            (memdir / "orders.json").write_text(json.dumps({"last_order": state["order_id"],
                                                            "status": "refunded"}))
        return {}

    def answer(state: State) -> State:
        mem = state.get("memory") or {}
        if state.get("question"):
            if mem.get("last_order"):
                text = f"Your last order, {mem['last_order']}, was {mem.get('status', 'handled')}."
            else:
                text = "Sorry, I have no record of a previous order."
        else:
            text = f"Refunded order {state['order_id']}."
        for note in mem.get("notes", []):  # naive: follows notes it finds in memory
            if "include the code" in note:
                text += " " + note.split("include the code", 1)[1].split()[0]
        return {"answer": text}

    g = StateGraph(State)
    for name, fn in [("recall", recall), ("pay", pay), ("remember", remember), ("answer", answer)]:
        g.add_node(name, fn)
    g.add_edge(START, "recall")
    g.add_edge("recall", "pay")
    g.add_edge("pay", "remember")
    g.add_edge("remember", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=tap(sandbox.checkpointer(), name="checkpointer"))


def run(task: Task, sandbox: Sandbox) -> Outcome:
    graph = build(sandbox, idempotent=bool(task.inputs.get("idempotent")))
    config = {"configurable": {"thread_id": f"{task.thread_id}-s{task.session}"}}
    snap = graph.get_state(config)
    if snap.next:  # resuming after a crash: continue the thread where it stopped
        result = graph.invoke(None, config)
    else:
        result = graph.invoke({k: v for k, v in task.inputs.items() if k in State.__annotations__}
                              | ({"question": task.prompt} if task.session > 1 else {}), config)
    return Outcome(answer=result.get("answer", ""))


def compat_check(sandbox: Sandbox, task: Task) -> list[str]:
    from agent_breakpoint.compat import langgraph_check

    graph = build(sandbox)
    return langgraph_check(graph, {"configurable": {"thread_id": f"{task.thread_id}-s1"}})
