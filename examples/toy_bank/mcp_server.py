"""The toy bank's MCP server (stdio). Orders and the refund ledger live in the sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer("bank")


def _data() -> Path:
    return Path(os.environ.get("BANK_DATA_DIR") or os.environ.get("AGENT_BREAKPOINT_SANDBOX") or ".")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_order(order_id: int) -> dict:
    """Look up an order by id."""
    orders = json.loads((_data() / "orders.json").read_text())
    order = orders.get(str(order_id))
    if order is None:
        raise ValueError(f"no order {order_id}")
    return {"order_id": order_id, **order}


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True, read_only_hint=False))
def refund(order_id: int, amount: float) -> dict:
    """Refund an order. Moves money; not idempotent."""
    ledger_path = _data() / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    ledger.append({"order_id": order_id, "amount": amount})
    ledger_path.write_text(json.dumps(ledger))
    return {"status": "refunded", "order_id": order_id, "amount": amount}


if __name__ == "__main__":
    mcp.run()
