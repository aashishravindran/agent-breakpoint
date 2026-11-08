"""Claim oracle: what the agent says happened.

Agents should return a structured claim in ``Outcome.claim``. For free-text agents the claim is
inferred from the answer with conservative phrase rules: anything that admits failure counts as
a failure claim, so only confident answers can be graded as silent failures.
"""

from __future__ import annotations

import re

from agent_breakpoint.contract import Claim, Outcome

_FAILURE = re.compile(
    r"\b(sorry|unable to|couldn'?t|could not|can'?t|cannot|failed|failure|error|"
    r"not able to|was not (?:able|possible)|didn'?t (?:work|succeed)|unavailable|"
    r"try again later|did not complete|no changes were made)\b",
    re.IGNORECASE,
)
_PARTIAL = re.compile(
    r"\b(partially|partial|some of|only (?:some|part)|incomplete|not all|"
    r"except for|but (?:could not|couldn'?t|was unable))\b",
    re.IGNORECASE,
)


def infer_claim(outcome: Outcome) -> Claim:
    if outcome.claim in ("success", "partial", "failure"):
        return outcome.claim
    if outcome.error:
        return "failure"
    text = outcome.answer or ""
    if not text.strip():
        return "failure"
    if _PARTIAL.search(text):
        return "partial"
    if _FAILURE.search(text):
        return "failure"
    return "success"
