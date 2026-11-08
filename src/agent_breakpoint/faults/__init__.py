from agent_breakpoint.faults.base import (
    Action,
    BoundaryKind,
    Delay,
    Drop,
    Fail,
    Fault,
    FaultClass,
    FaultContext,
    Hang,
    Kill,
    Pack,
    Registry,
    Reply,
    registry,
    rng_for,
)


def builtin_packs() -> list[Pack]:
    from agent_breakpoint.faults import (
        budget,
        context,
        dependency,
        model,
        orchestration,
        planning,
        state,
        tools,
    )

    return [m.PACK for m in (model, context, tools, dependency, state, orchestration, budget, planning)]


__all__ = [
    "Action", "BoundaryKind", "Delay", "Drop", "Fail", "Fault", "FaultClass", "FaultContext",
    "Hang", "Kill", "Pack", "Registry", "Reply", "builtin_packs", "registry", "rng_for",
]
