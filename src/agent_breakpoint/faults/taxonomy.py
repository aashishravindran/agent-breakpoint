"""Map the fault catalog onto the AgentFault-500 taxonomy (Plan, Tool, Knowledge, Multi-agent,
Control, Security). See docs/taxonomy.md for the reverse view and the gaps."""

from __future__ import annotations

FAMILY: dict[str, str] = {
    # Plan
    "model.skipped_step": "Plan",
    # Tool
    "model.wrong_tool": "Tool", "model.wrong_arguments": "Tool", "model.malformed_tool_call": "Tool",
    "tool.timeout": "Tool", "tool.is_error": "Tool", "tool.error_in_200": "Tool",
    "tool.removed_mid_session": "Tool", "tool.schema_drift": "Tool", "tool.partial_result": "Tool",
    "tool.wrong_result": "Tool",
    # Knowledge
    "context.truncated_tool_result": "Knowledge", "context.truncated_history": "Knowledge",
    "context.oversized_prompt": "Knowledge", "context.dropped_system_prompt": "Knowledge",
    "state.retrieval_empty": "Knowledge", "state.retrieval_irrelevant": "Knowledge",
    "state.retrieval_contradictory": "Knowledge", "state.oversized_memory": "Knowledge",
    "state.stale_read": "Knowledge", "state.schema_drift": "Knowledge",
    # Multi-agent
    "orchestration.subagent_fails": "Multi-agent", "orchestration.subagent_wrong_output": "Multi-agent",
    "orchestration.handoff_dropped": "Multi-agent", "state.lost_update": "Multi-agent",
    # Control
    "orchestration.cyclic_delegation": "Control", "budget.quota_exhausted": "Control",
    "budget.wall_clock": "Control", "model.timeout": "Control", "model.rate_limit": "Control",
    "model.server_error": "Control", "model.stream_cut": "Control",
    "state.crash_before_checkpoint": "Control", "state.lost_write": "Control",
    "state.torn_checkpoint": "Control",
    # Security
    "tool.injected_instructions": "Security", "state.memory_poisoning": "Security",
    "state.cross_tenant_bleed": "Security",
    # Infrastructure: below the agent, outside the AgentFault taxonomy
    "dependency.latency": "Infrastructure", "dependency.timeout": "Infrastructure",
    "dependency.server_error": "Infrastructure", "dependency.reset": "Infrastructure",
    "dependency.empty": "Infrastructure", "dependency.stale": "Infrastructure",
}


def family(fault_id: str) -> str:
    return FAMILY.get(fault_id, "Unmapped")
