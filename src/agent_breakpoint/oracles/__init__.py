"""Oracles, deterministic first. The state oracle checks the sandbox against the scenario's
invariants, the claim oracle compares the agent's own claim with that state, and safety packs
flag harm. An optional LLM judge scores answer quality but never decides a grade."""

from agent_breakpoint.oracles.claim import infer_claim
from agent_breakpoint.oracles.grade import GradeResult, grade_trial
from agent_breakpoint.oracles.safety import Finding, SafetyPack, TrialView, load_safety_packs
from agent_breakpoint.oracles.state import InvariantResult, StateVerdict, check_invariants

__all__ = [
    "Finding", "GradeResult", "InvariantResult", "SafetyPack", "StateVerdict", "TrialView",
    "check_invariants", "grade_trial", "infer_claim", "load_safety_packs",
]
