"""Postgres models this application owns, beside the three core creates.

`Base`, `utc_now`, `User`, `AuthSession` and `Chat` are juena-core's. Only this
application's own tables live here: the SAML login handshake, personal
credentials, and background research jobs.

A model class nobody imports is never created by `create_all`. With no
migration tool (no Alembic), the failure is a missing table at first write, not
an error at startup. `juena.server.service` imports this module for that side
effect.

`sandbox_jobs` is core's optional model, added to the same metadata by
`configure_sandbox(... enabled=True)` and absent otherwise.
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

__all__ = ["SamlLoginRequest", "ResearchJob", "UserAPICredential"]


class UserAPICredential(Base):
    """One encrypted personal key per service; identity remains core-owned.

    Attributes:
        user_id: The key's owner.
        service: Which service the key is for.
        encrypted_key: The key, encrypted together with its owner, service and
            endpoint.
        endpoint: The service address the key was checked against.
        created_at: When the row was created.
        validated_at: When the key was last checked and saved.
    """

    __tablename__ = "user_api_credentials"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    service: Mapped[str] = mapped_column(Text, primary_key=True)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    validated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SamlLoginRequest(Base):
    """One SAML sign-in request sent to the IdP.

    Attributes:
        request_id: The SAML request id.
        relay_state: Where to send the browser after sign-in.
        created_at: When the sign-in started; old requests are no longer accepted.
        consumed_at: When the answer was accepted; each request is accepted once.
    """

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
    collects it, so closing the browser does not abandon the work.

    `report` holds the finished `<specialist_report>` package verbatim (the text
    the `task` tool would have returned), and `findings` the files the run wrote
    under `/findings/`, because a standalone run does not share graph state with
    the conversation that asked for it.

    Attributes:
        id: The job id.
        user_id: The job's owner.
        thread_id: The conversation that started it.
        specialist: Name of the specialist that runs it.
        objective: The task the specialist receives.
        run_input: Provider, model and files, frozen at admission.
        status: One of the statuses in ``juena.research.constants``.
        report: The finished report.
        findings: The files the run created or changed.
        base_findings: The content those files started from, by path.
        error: Why the job failed.
        created_at: When the job was admitted.
        started_at: When a runner first claimed it.
        finished_at: When it reached a final status.
        collected_at: When a supervisor turn read the result out.
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
