"""Postgres models this application owns, beside the three core creates.

`Base`, `utc_now`, `User`, `AuthSession` and `Chat` are juena-core's (00, decision
9, part 4). Only the tables no other consumer of core has live here: the SAML
login handshake, and background research jobs.

**A model class nobody imports is never created by `create_all`.** There is no
Alembic, so the failure is a missing table at first write rather than an error at
startup. `juena.server.service` imports this module for that side effect, with a
`# noqa: F401`, exactly as it imports the agent module for its registration.

`sandbox_jobs` is not here either: it is core's optional model, attached to the
same metadata by `configure_sandbox(... enabled=True)` and absent otherwise.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from juena_core.agents.findings import FindingsMap
from juena_core.server.database.models import Base, utc_now
from juena.research.constants import ACTIVE_STATUSES as ACTIVE_RESEARCH_STATUSES

__all__ = ["SamlLoginRequest", "ResearchJob"]


class SamlLoginRequest(Base):
    __tablename__ = "saml_login_requests"

    request_id: Mapped[str] = mapped_column(Text, primary_key=True)
    relay_state: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


_ACTIVE_RESEARCH_PREDICATE = text(
    "status IN ({})".format(", ".join(f"'{s}'" for s in ACTIVE_RESEARCH_STATUSES))
)


class ResearchJob(Base):
    """One specialist delegation running detached from the request that started it.

    The supervisor normally delegates inside its own tool call, which ties the
    specialist's lifetime to the HTTP connection. A row here is the alternative:
    the run happens in a background task and the answer waits until a later turn
    collects it, so closing the browser no longer abandons the work.

    `report` holds the finished `<specialist_report>` package verbatim -- the
    same text the `task` tool would have returned -- and `findings` the files the
    run wrote under `/findings/`, because a standalone invocation does not share
    graph state with the conversation that asked for it.
    """

    __tablename__ = "research_jobs"
    __table_args__ = (
        Index("ix_research_jobs_thread_status", "thread_id", "status"),
        Index(
            "ix_research_jobs_active_user",
            "user_id",
            postgresql_where=_ACTIVE_RESEARCH_PREDICATE,
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    specialist: Mapped[str] = mapped_column(String(64), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    # Frozen at admission: provider, model and the files the specialist receives.
    # Recovery must not reseed from a conversation that has moved on.
    run_input: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    report: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Deep Agents `FileData`, not plain strings: {path: {content, encoding, ...}}.
    # Only what the run created or changed -- never the snapshot it was seeded
    # with. JSON guarantees none of that shape, so anything read back out goes
    # through `juena_core.agents.findings.validate_findings` before it reaches state.
    findings: Mapped[FindingsMap] = mapped_column(JSON, nullable=False, default=dict)
    # The findings content this run started from, keyed by path. Collection
    # compares it against the conversation's current value to tell an ordinary
    # merge from a conflict.
    base_findings: Mapped[dict[str, str]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Set when a supervisor turn has actually read the result out. Without it the
    # same finished job is reported again on every following turn.
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
