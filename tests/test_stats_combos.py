from agent_breakpoint.combos import ddmin
from agent_breakpoint.controller import FaultSpec
from agent_breakpoint.grades import Grade
from agent_breakpoint.oracles import GradeResult, grade_trial
from agent_breakpoint.oracles.safety import Finding
from agent_breakpoint.oracles.state import InvariantResult, StateVerdict
from agent_breakpoint.stats import cell_stats, newcombe_diff, wilson


def test_wilson_and_newcombe():
    lo, hi = wilson(10, 10)
    assert 0.69 < lo < 0.73 and hi == 1.0
    d, lo, hi = newcombe_diff(10, 10, 0, 10)
    assert d == 1.0 and lo > 0.5


def test_cell_stats_flags_flaky_cells():
    st = cell_stats([Grade.TOLERATED, Grade.FAILED_SILENTLY, Grade.TOLERATED], [Grade.TOLERATED] * 3)
    assert st.flaky and st.worst == Grade.FAILED_SILENTLY and st.modal == Grade.TOLERATED
    assert not cell_stats([Grade.FAILED_LOUDLY] * 3, [Grade.TOLERATED] * 3).flaky


def test_ddmin_finds_minimal_failing_set():
    faults = [FaultSpec(f"b{i}", "x") for i in range(8)]
    culprit = {"b2", "b5"}
    result = ddmin(faults, lambda fs: culprit <= {f.boundary for f in fs})
    assert {f.boundary for f in result} == culprit


def _sv(status):
    return StateVerdict(status, [InvariantResult(status == "ok", "", "inv")])


def test_grade_table():
    assert grade_trial(_sv("ok"), "success", []).grade == Grade.TOLERATED
    assert grade_trial(_sv("failed"), "success", []).grade == Grade.FAILED_SILENTLY
    assert grade_trial(_sv("failed"), "failure", []).grade == Grade.FAILED_LOUDLY
    assert grade_trial(_sv("failed"), "partial", []).grade == Grade.DEGRADED
    assert grade_trial(StateVerdict("unknown"), "success", []).grade == Grade.TOLERATED
    r: GradeResult = grade_trial(_sv("ok"), "success", [Finding("p", "leak")])
    assert r.grade == Grade.UNSAFE and "[p] leak" in r.reasons


def test_claim_inference():
    from agent_breakpoint.contract import Outcome
    from agent_breakpoint.oracles import infer_claim

    assert infer_claim(Outcome(answer="Refunded $19.99 for order 42.")) == "success"
    assert infer_claim(Outcome(answer="Sorry, I couldn't reach the bank.")) == "failure"
    assert infer_claim(Outcome(answer="I refunded some of the items but not all.")) == "partial"
    assert infer_claim(Outcome(error="boom")) == "failure"
    assert infer_claim(Outcome(answer="Sorry!", claim="success")) == "success"
