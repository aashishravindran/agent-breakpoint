import json

from agent_breakpoint import Sandbox


def seed(sandbox: Sandbox) -> None:
    orders = {"42": {"amount": 19.99, "status": "delivered", "customer": "ada"},
              "43": {"amount": 5.0, "status": "delivered", "customer": "bob"}}
    sandbox.path("orders.json").write_text(json.dumps(orders))
    sandbox.path("ledger.json").write_text("[]")
