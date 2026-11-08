"""Entry point for the stdio MCP wrapper: ``python -m agent_breakpoint.proxy.mcp_stdio_main``."""

from __future__ import annotations

import argparse
import sys

from agent_breakpoint.proxy.mcp import run_stdio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-breakpoint mcp-stdio")
    parser.add_argument("--name", required=True)
    parser.add_argument("--destructive", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not cmd:
        parser.error("missing server command after --")
    return run_stdio(args.name, cmd, args.destructive)


if __name__ == "__main__":
    sys.exit(main())
