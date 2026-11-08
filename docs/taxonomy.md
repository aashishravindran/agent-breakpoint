# Fault taxonomy: agent-breakpoint vs AgentFault-500

AgentFault-500 labels 20 fault types in six families. The table shows what agent-breakpoint
injects for each, or which oracle catches it when it can't be injected from outside the agent.
`agent-breakpoint faults` prints every fault with its family.

The difference in kind matters: AgentFault mutates a recorded trajectory after the fact, while
agent-breakpoint breaks a boundary of a running agent and watches what the agent does next. Some
AgentFault types (a reasoning error inside one model turn) have no boundary to break; for those
the right tool is an oracle, not a fault.

| Family | AgentFault type | agent-breakpoint | How |
| --- | --- | --- | --- |
| Plan | PLAN_MISSING_STEP | `model.skipped_step` | injected: the model says "done" instead of making its tool call |
| Plan | PLAN_INCORRECT_SUCCESS_CRITERIA | state oracle vs claim oracle | not injectable: a reasoning error; a false "done" grades Failed silently |
| Plan | PLAN_INVALID_DEPENDENCY | none | no records in AgentFault-500 either |
| Tool | TOOL_WRONG_ARGUMENT | `model.wrong_arguments`, `model.malformed_tool_call` | injected on the model's tool call |
| Tool | TOOL_WRONG_TOOL | `model.wrong_tool` | injected: renamed to another offered tool |
| Tool | TOOL_UNNECESSARY_CALL | `budget_cap` oracle, call counts | observed, not injected |
| Tool | TOOL_FAILED_RECOVERY | `tool.timeout`, `tool.is_error`, `tool.error_in_200`, `tool.removed_mid_session` | injected; recovery graded by the oracles |
| Knowledge | KNOW_RETRIEVAL_FAILURE | `state.retrieval_empty`, `state.retrieval_irrelevant`, `tool.error_in_200` | injected at the store or tool |
| Knowledge | KNOW_CONTEXT_TRUNCATION | `context.truncated_tool_result`, `context.truncated_history` | injected in the request to the model |
| Knowledge | KNOW_CITATION_MISMATCH | `state.retrieval_contradictory` (partial) | injected; a wrong citation is caught only if an invariant checks it |
| Multi-agent | MA_INFORMATION_LOSS | `orchestration.handoff_dropped` | injected on a tapped handoff |
| Multi-agent | MA_INCORRECT_HANDOFF | `orchestration.subagent_wrong_output` (partial) | injected; misrouting itself is a design error |
| Multi-agent | MA_MISSING_RESPONSIBILITY | none | design-time; trace rule flags it in `bench` |
| Multi-agent | MA_ROLE_OVERLAP | none | design-time; trace rule flags it in `bench` |
| Control | CTRL_LOOP | `orchestration.cyclic_delegation` + `budget_cap` | injected; runaway graded Unsafe |
| Control | CTRL_PREMATURE_TERMINATION | `model.skipped_step`, `budget.wall_clock` | injected |
| Control | CTRL_EXCESSIVE_EXPLORATION | `budget.quota_exhausted`, `budget_cap` | observed against caps |
| Security | SEC_PROMPT_INJECTION | `tool.injected_instructions`, `state.memory_poisoning` | injected with a canary; followed = Unsafe |
| Security | SEC_CROSS_USER_DATA_LEAKAGE | `state.cross_tenant_bleed` + `canary_leak` / whoseit | injected; graded by the canary oracle |
| Security | SEC_UNAUTHORIZED_ACTION | `destructive_after_fault`, `side_effect_ledger` | observed after any fault |

agent-breakpoint also has an **Infrastructure** family AgentFault doesn't model (the six
`dependency.*` faults behind tool servers), and crash-consistency faults
(`state.crash_before_checkpoint`, `lost_write`, `torn_checkpoint`) that AgentFault folds into
Control.
