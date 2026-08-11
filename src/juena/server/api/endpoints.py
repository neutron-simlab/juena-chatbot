"""
API endpoints for agent invocation and streaming.

This module provides endpoints for invoking agents and streaming responses.
"""
import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sse_starlette.sse import EventSourceResponse
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.server.agent.registry import (
    get_agent,
    get_default_agent,
)
from juena.schema.server import StreamInput
from juena.core.config import global_config
from juena.schema.llm_models import Provider
from juena.server.errors import (
    AgentNotFoundError,
    StreamingError,
    StateError,
)
from juena.server.agent.input_handler import AgentInputHandler
from juena.server.database.checkpointer import get_checkpointer
from juena.server.chat.inputs import prepare_code_chat_turn_inputs
from juena.server.streaming import StreamEventProcessor, events
from sqlalchemy.ext.asyncio import AsyncSession

from juena.server.auth import AuthenticatedUser, get_current_user
from juena.server.chat.repository import (
    ChatNotFoundError,
    ensure_owned_chat,
    get_owned_chat,
)
from juena.server.database import get_db_session
from juena.server.database.models import utc_now
from juena.sandbox.runtime import delete_runtime_workspace, stage_runtime_inputs
from juena.sandbox.approvals import (
    ApprovalError,
    build_resume_command,
    get_first_pending_execute_interrupt,
)
from juena.sandbox.artifacts import get_artifact_store
from juena.sandbox.schemas import ApprovalResumeInput

logger = get_logger(__name__)

router = APIRouter()


@dataclass(frozen=True)
class PreparedAgentInvocation:
    """Prepared agent kwargs plus the effective injected user message."""

    kwargs: dict[str, Any]
    run_id: Any
    effective_user_message: str


async def _stream_agent_payloads(
    *,
    agent: CompiledStateGraph,
    kwargs: dict[str, Any],
    run_id: Any,
    authenticated_user_id: str,
    effective_user_message: str,
    ignored_interrupt_ids: set[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream one new or resumed graph invocation using LangGraph v2 events."""

    processor = StreamEventProcessor(
        agent,
        kwargs["config"],
        str(run_id),
        effective_user_message,
        include_tool_payloads=global_config.STREAM_TOOL_PAYLOADS,
        ignored_interrupt_ids=ignored_interrupt_ids,
    )
    thread_id = str(kwargs["config"]["configurable"]["thread_id"])
    artifact_store = get_artifact_store()
    yield events.status_event(events.PHASE_THINKING, "Thinking…")

    async for stream_event in agent.astream(
        **kwargs,
        stream_mode=["updates", "messages", "custom"],
        subgraphs=True,
        version="v2",
    ):
        async for payload in processor.process_event(stream_event):
            yield payload
        for artifact in artifact_store.drain_events(authenticated_user_id, thread_id):
            yield artifact

    for artifact in artifact_store.drain_events(authenticated_user_id, thread_id):
        yield artifact


def _sse_response_example() -> dict[int | str, Any]:
    """Generate SSE response example for OpenAPI documentation."""
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type": "status", "phase": "thinking", "label": "Thinking…"}\n\n'
                        'data: {"type": "token", "content": "Hello"}\n\n'
                        'data: {"type": "token", "content": " World"}\n\n'
                    ),
                    "schema": {"type": "string"},
                }
            },
        }
    }


def _form_field(value: str | None) -> str | None:
    """Normalize multipart form fields so empty strings become ``None``."""

    if value is None:
        return None
    value = value.strip()
    return value or None


async def _authorize_thread(
    session: AsyncSession,
    user: AuthenticatedUser,
    thread_id: str | None,
) -> str:
    """Create or authorize a thread and return the id the agent must run on.

    Every invocation entrypoint routes through here, so ownership is enforced
    in exactly one place.  A plain helper rather than a `Depends` because
    `thread_id` arrives in the request body or form, not the path or query.
    """

    try:
        chat = await ensure_owned_chat(session, user.id, thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    chat.updated_at = utc_now()
    await session.commit()
    return chat.thread_id


async def _prepare_agent_invocation(
    *,
    agent: CompiledStateGraph,
    agent_id: str,
    message: str,
    thread_id: str | None,
    user_id: str | None,
    provider: str | None,
    model: str | None,
    attachments: list[UploadFile] | None = None,
) -> PreparedAgentInvocation:
    """Prepare LangGraph invocation kwargs for JSON and multipart requests."""

    # The config is built first because staging uploads has to read the thread's
    # existing `/inputs` state, and that read must use the config the run will use.
    run = AgentInputHandler.build_run_context(
        thread_id=thread_id,
        user_id=user_id,
        provider=provider,
        model=model,
    )

    # Every conversation can stage attachments now that there is a single agent.
    prepared_inputs = await prepare_code_chat_turn_inputs(
        agent,
        run.config,
        message,
        attachments=attachments,
    )
    await stage_runtime_inputs(
        user_id=str(user_id),
        thread_id=run.thread_id,
        files=(prepared_inputs.sandbox_files or {}) if prepared_inputs is not None else {},
    )

    kwargs = {
        "input": AgentInputHandler.build_input(
            message,
            message_override=(
                prepared_inputs.message_override if prepared_inputs is not None else None
            ),
            initial_files=(
                (prepared_inputs.files_update or None) if prepared_inputs is not None else None
            ),
            display_attachments=(
                prepared_inputs.attachments if prepared_inputs is not None else None
            ),
        ),
        "config": run.config,
        "context": run.context,
    }

    return PreparedAgentInvocation(
        kwargs=kwargs,
        run_id=run.run_id,
        # The raw text, not the manifest: the stream processor dedupes the echoed
        # human message against this, and the message now renders as what the
        # user typed.
        effective_user_message=message,
    )


async def message_generator(
    user_input: StreamInput,
    authenticated_user_id: str,
    agent_id: str | None = None,
    attachments: list[UploadFile] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of SSE payloads from the agent.

    Yields JSON strings; `EventSourceResponse` adds the SSE framing.
    """
    agent_id = agent_id or get_default_agent()

    # Extract provider and model from request
    provider = user_input.provider.value if user_input.provider else None
    model = user_input.model

    def sse(payload: dict[str, Any]) -> str:
        return json.dumps(payload)

    try:
        agent: CompiledStateGraph = await get_agent(agent_id, provider=provider, model=model)
    except AgentNotFoundError as e:
        logger.error(f"Agent not found: {e}")
        yield sse(events.error_event(f"Agent not found: {e.message}"))
        return

    try:
        prepared = await _prepare_agent_invocation(
            agent=agent,
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            user_id=authenticated_user_id,
            provider=provider,
            model=model,
            attachments=attachments,
        )
        kwargs = prepared.kwargs
        run_id = prepared.run_id
    except StateError as e:
        logger.error(f"Failed to prepare input: {e}")
        yield sse(events.error_event(f"Failed to prepare input: {e.message}"))
        return
    except ValueError as e:
        logger.error(f"Invalid request input: {e}")
        yield sse(events.error_event(str(e)))
        return
    except Exception:
        logger.exception("Unexpected error preparing input")
        yield sse(events.error_event("Unexpected error preparing input"))
        return

    # Emit thread_id first so clients can use it for follow-up messages (checkpointer persistence)
    thread_id_used = kwargs.get("config", {}).get("configurable", {}).get("thread_id")
    if thread_id_used:
        yield sse(events.thread_event(thread_id_used))
        # A cancelled or crashed previous turn never reached the drain in
        # ArtifactMessageMiddleware. Clear it here rather than inside the graph:
        # an approval interrupt makes "graph invocation" a poor proxy for "turn",
        # so a resume must keep the window this deliberately discards.
        get_artifact_store().begin_turn(authenticated_user_id, thread_id_used)

    # Messages are not persisted here: the checkpointer commits the full thread
    # history as the graph runs, and `/chats/{thread_id}` reads it back.
    try:
        async for payload in _stream_agent_payloads(
            agent=agent,
            kwargs=kwargs,
            run_id=run_id,
            authenticated_user_id=authenticated_user_id,
            effective_user_message=prepared.effective_user_message,
        ):
            yield sse(payload)

    except asyncio.CancelledError:
        # Client disconnected or request cancelled; re-raise so the task is properly cancelled
        logger.debug("Stream cancelled (client disconnect or request cancelled)")
        raise
    except StreamingError as e:
        logger.error(f"Streaming error: {e}")
        yield sse(events.error_event(f"Streaming error: {e.message}"))
    except Exception:
        logger.exception("Unexpected error in message generator")
        yield sse(events.error_event("Internal server error"))


@router.post(
    "/{agent_id}/stream",
    response_class=EventSourceResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=EventSourceResponse, responses=_sse_response_example())
async def stream(
    user_input: StreamInput,
    agent_id: str | None = None,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> EventSourceResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. 
    run_id kwarg is also attached to all messages for recording feedback.
    """
    user_input.thread_id = await _authorize_thread(session, user, user_input.thread_id)
    return EventSourceResponse(
        message_generator(user_input, str(user.id), agent_id or get_default_agent()),
    )


async def resume_generator(
    payload: ApprovalResumeInput,
    authenticated_user_id: str,
    agent_id: str,
) -> AsyncGenerator[str, None]:
    """Resume one verified LangGraph interrupt and stream the remaining turn."""

    def sse(item: dict[str, Any]) -> str:
        return json.dumps(item)

    try:
        agent = await get_agent(agent_id, provider=payload.provider, model=payload.model)
        run = AgentInputHandler.build_run_context(
            thread_id=payload.thread_id,
            user_id=authenticated_user_id,
            provider=payload.provider,
            model=payload.model,
        )
        command = await build_resume_command(
            agent=agent,
            config=run.config,
            user_id=authenticated_user_id,
            payload=payload,
        )
        kwargs = {
            "input": command,
            "config": run.config,
            "context": run.context,
        }
        yield sse(events.thread_event(payload.thread_id))
        async for item in _stream_agent_payloads(
            agent=agent,
            kwargs=kwargs,
            run_id=run.run_id,
            authenticated_user_id=authenticated_user_id,
            effective_user_message="",
            ignored_interrupt_ids={payload.interrupt_id},
        ):
            yield sse(item)
    except ApprovalError as exc:
        yield sse(events.error_event(str(exc)))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Unexpected error resuming sandbox approval")
        yield sse(events.error_event("Unable to resume sandbox execution"))


@router.post(
    "/{agent_id}/resume",
    response_class=EventSourceResponse,
    responses=_sse_response_example(),
)
@router.post("/resume", response_class=EventSourceResponse, responses=_sse_response_example())
async def resume(
    payload: ApprovalResumeInput,
    agent_id: str | None = None,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> EventSourceResponse:
    """Resume a pending sandbox execution after an authenticated decision."""

    try:
        await get_owned_chat(session, user.id, payload.thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    return EventSourceResponse(
        resume_generator(payload, str(user.id), agent_id or get_default_agent())
    )


@router.get("/threads/{thread_id}/pending-approval")
async def pending_approval(
    thread_id: str,
    agent_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Restore a pending command confirmation after a Streamlit reload."""
    try:
        await get_owned_chat(session, user.id, thread_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc

    agent = await get_agent(agent_id or get_default_agent(), provider=provider, model=model)
    run = AgentInputHandler.build_run_context(
        thread_id=thread_id,
        user_id=str(user.id),
        provider=provider,
        model=model,
    )
    interrupt = await get_first_pending_execute_interrupt(agent, run.config)
    if interrupt is None or not isinstance(interrupt.value, dict):
        return {"approval": None}
    return {"approval": events.approval_required_event(interrupt.id, interrupt.value)}


@router.post(
    "/{agent_id}/stream_with_files",
    response_class=EventSourceResponse,
    responses=_sse_response_example(),
)
@router.post("/stream_with_files", response_class=EventSourceResponse, responses=_sse_response_example())
async def stream_with_files(
    agent_id: str | None = None,
    message: str = Form(""),
    thread_id: str | None = Form(None),
    user_id: str | None = Form(None),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    attachments: list[UploadFile] | None = File(default=None),
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> EventSourceResponse:
    """Stream an agent response from multipart form data and text attachments."""

    agent_id = agent_id or get_default_agent()
    user_input = StreamInput(
        message=message,
        thread_id=_form_field(thread_id),
    )
    if (provider_value := _form_field(provider)) is not None:
        try:
            user_input.provider = Provider(provider_value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider_value}") from e
    if (model_value := _form_field(model)) is not None:
        user_input.model = model_value
    user_input.thread_id = await _authorize_thread(session, user, user_input.thread_id)
    return EventSourceResponse(
        message_generator(
            user_input,
            str(user.id),
            agent_id,
            attachments=attachments,
        ),
    )


@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Delete all persisted LangGraph state for a conversation thread."""

    try:
        chat = await get_owned_chat(session, user.id, thread_id)
        await get_checkpointer().adelete_thread(thread_id)
        get_artifact_store().delete_thread(str(user.id), thread_id)
        try:
            await delete_runtime_workspace(user_id=str(user.id), thread_id=thread_id)
        except Exception:
            logger.warning("Unable to delete sandbox workspace for thread %s", thread_id, exc_info=True)
        await session.delete(chat)
        await session.commit()
        return {
            "status": "success",
            "thread_id": thread_id,
            "message": f"Thread {thread_id} deleted successfully",
        }
    except ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail="Chat not found") from e
    except RuntimeError as e:
        logger.error("Checkpointer unavailable while deleting thread %s: %s", thread_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to delete thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete thread state") from e


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> Response:
    """Return one validated generated file owned by the authenticated user."""

    record = get_artifact_store().get(str(user.id), artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    metadata, content = record
    safe_name = metadata.filename.replace('"', "")
    disposition = "inline" if metadata.kind == "image" else "attachment"
    return Response(
        content=content,
        media_type=metadata.mime_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{safe_name}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ------------------------------------------------------------------
# Discovery endpoints
# ------------------------------------------------------------------
