"""Dependency-free background-research statuses and budgets.

Kept apart from the rest of `juena.research` so the ORM model can derive its
index predicate from the same tuple the runner branches on, without the model
importing an agent runtime.
"""

from typing import Literal

ResearchStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

#: A job holds a slot while it is active. `ResearchJob`'s partial index derives
#: its predicate from this tuple, so this is the single definition.
ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")

#: Background jobs one user may have in flight at once. Each one is a full
#: specialist loop against the model provider, so this is a spend limit as much
#: as a concurrency limit.
DEFAULT_CONCURRENCY = 2

#: Wall-clock ceiling for one job. `SPECIALIST_MODEL_CALL_LIMIT` already bounds
#: the number of turns; this bounds the time, which a slow provider can stretch
#: without ever exceeding a turn count.
DEFAULT_TIMEOUT_SECONDS = 900

#: What a restarting process says about jobs it finds still marked active. They
#: cannot be resumed: the run lived in the previous process's event loop.
RESTART_MESSAGE = "The server restarted before this research finished"

TIMEOUT_MESSAGE = "This research ran past its time limit and was stopped"

BUSY_MESSAGE = (
    "You already have the maximum number of background research jobs running. "
    "Wait for one to finish, or use `task` to delegate this now instead."
)
