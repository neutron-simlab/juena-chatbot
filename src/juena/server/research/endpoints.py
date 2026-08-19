"""Read-only view of a conversation's background research.

Exists for the UI, not the agent: the supervisor collects results through its
own `check_research` tool. This is what lets the sidebar tell the user a result
is waiting without them having to send a message to find out.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from juena.research import jobs
from juena.server.auth import AuthenticatedUser, get_current_user
from juena.server.chat.repository import ChatNotFoundError, get_owned_chat
from juena.server.database import get_db_session

router = APIRouter(prefix="/research", tags=["research"])


@router.get("")
async def list_research(
    thread_id: str = Query(..., description="The conversation to report on"),
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Uncollected research on one thread, plus how many are ready.

    Ownership is checked against the chat, not against the job rows: a thread id
    the user does not own must 404 whether or not any research exists on it.
    """
    try:
        await get_owned_chat(session, user.id, thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc

    pending = await jobs.for_thread(thread_id, user.id)
    entries = [
        {
            "job_id": str(job.id),
            "specialist": job.specialist,
            "status": job.status,
            "objective": job.objective,
            "created_at": job.created_at.isoformat(),
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }
        for job in pending
    ]
    return {
        "jobs": entries,
        "running": sum(1 for e in entries if e["status"] in ("queued", "running")),
        "ready": sum(1 for e in entries if e["status"] in ("succeeded", "failed")),
    }
