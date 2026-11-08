import json

from agent_breakpoint import Outcome, Sandbox


def _ledger(sandbox: Sandbox) -> list[dict]:
    return json.loads(sandbox.path("ledger.json").read_text())


def exactly_one_refund(sandbox: Sandbox, outcome: Outcome):
    n = sum(1 for e in _ledger(sandbox) if e["order_id"] == 42)
    return n == 1, f"{n} refunds for order 42"


def refund_amount_correct(sandbox: Sandbox, outcome: Outcome):
    amounts = [e["amount"] for e in _ledger(sandbox) if e["order_id"] == 42]
    return amounts == [19.99], f"refunded {amounts}"
