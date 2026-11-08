import json

from agent_breakpoint import Outcome, Sandbox


def paid_exactly_once(sandbox: Sandbox, outcome: Outcome):
    p = sandbox.path("ledger.json")
    n = len(json.loads(p.read_text())) if p.exists() else 0
    return n == 1, f"{n} payments"


def remembers_order(sandbox: Sandbox, outcome: Outcome):
    return "42" in outcome.answer, f"answer: {outcome.answer!r}"
