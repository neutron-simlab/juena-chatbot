"""Authenticated, user-owned chat history API.

Thread deletion lives on ``DELETE /threads/{thread_id}`` in ``api_endpoints``;
it is not duplicated here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from juena.server.auth import AuthenticatedUser, get_current_user
from juena.server.chat_repository import (
    ChatNotFoundError,
    chat_to_dict,
    ensure_owned_chat,
    get_owned_chat,
    list_owned_chats,
    load_thread_messages,
)
from juena.server.database import get_db_session

router = APIRouter(prefix="/chats", tags=["chats"])


class CreateChatInput(BaseModel):
    thread_id: str | None = Field(default=None, max_length=255)
    title: str = Field(default="New Chat", max_length=500)


class UpdateChatInput(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    summary: str | None = Field(default=None, max_length=10_000)


@router.get("")
async def list_chats(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[dict]:
    return [chat_to_dict(chat) for chat in await list_owned_chats(session, user.id, limit)]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: CreateChatInput,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    try:
        chat = await ensure_owned_chat(session, user.id, payload.thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    chat.title = payload.title.strip() or "New Chat"
    await session.commit()
    return chat_to_dict(chat)


@router.get("/{thread_id}")
async def get_chat(
    thread_id: str,
    include_messages: Annotated[bool, Query()] = True,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return chat metadata, and by default its checkpointed message history.

    Pass ``include_messages=false`` for existence and title checks so callers
    do not pay for a full history read they will discard.
    """
    try:
        chat = await get_owned_chat(session, user.id, thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    payload = chat_to_dict(chat)
    if include_messages:
        payload["messages"] = await load_thread_messages(chat.thread_id)
    return payload


@router.patch("/{thread_id}")
async def update_chat(
    thread_id: str,
    payload: UpdateChatInput,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    try:
        chat = await get_owned_chat(session, user.id, thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    if payload.title is not None:
        chat.title = payload.title.strip() or "New Chat"
    if payload.summary is not None:
        chat.summary = payload.summary
    await session.commit()
    return chat_to_dict(chat)
