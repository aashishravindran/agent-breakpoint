"""agent-breakpoint: break an agent system on purpose and grade how it copes.

The only names agent code should import are ``tap``, ``Outcome``, ``Task`` and ``Sandbox``.
``tap()`` is inert outside the harness, so it is safe to leave in production code.
"""

from agent_breakpoint.contract import Outcome, Sandbox, Task
from agent_breakpoint.grades import Grade
from agent_breakpoint.taps import tap

__all__ = ["Grade", "Outcome", "Sandbox", "Task", "tap"]
__version__ = "0.1.0"
