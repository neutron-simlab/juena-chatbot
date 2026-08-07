"""
API endpoints for agent invocation and streaming.

This module provides endpoints for invoking agents and streaming responses.
"""
import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.server.agent.registry import (
    get_agent,
    get_default_agent,
    list_registered_agents,
)
from juena.schema.server import ChatMessage, StreamInput, UserInput
from juena.core.config import global_config
from juena.schema.llm_models import Provider
from juena.server.utils import langchain_to_chat_message
from juena.server.errors import (
    AgentNotFoundError,
    StreamingError,
    StateError,
    ChatbotServerError
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

logger = get_logger(__name__)

router = APIRouter()


@dataclass(frozen=True)
class PreparedAgentInvocation:
    """Prepared agent kwargs plus the effective injected user message."""

    kwargs: dict[str, Any]
    run_id: Any
    effective_user_message: str


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

    # Messages are not persisted here: the checkpointer commits the full thread
    # history as the graph runs, and `/chats/{thread_id}` reads it back.
    try:
        processor = StreamEventProcessor(
            agent,
            kwargs["config"],
            str(run_id),
            prepared.effective_user_message,
            include_tool_payloads=global_config.STREAM_TOOL_PAYLOADS,
        )

        yield sse(events.status_event(events.PHASE_THINKING, "Thinking…"))

        # `updates` carries completed messages, `messages` carries answer
        # tokens. `custom` is not subscribed: nothing emits custom stream data.
        # subgraphs=True is what lets the processor tell coordinator output
        # apart from subagent activity.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages"], subgraphs=True
        ):
            async for payload in processor.process_event(stream_event):
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


async def _invoke_with_attachments(
    user_input: UserInput,
    *,
    authenticated_user_id: str,
    agent_id: str,
    attachments: list[UploadFile],
) -> ChatMessage:
    """Invoke an agent with multipart file attachments."""

    provider = user_input.provider.value if user_input.provider else None
    model = user_input.model

    try:
        agent: CompiledStateGraph = await get_agent(agent_id, provider=provider, model=model)
    except AgentNotFoundError as e:
        logger.error(f"Agent not found: {e}")
        raise HTTPException(status_code=404, detail=e.message) from e

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
        raise HTTPException(status_code=500, detail=f"Failed to prepare input: {e.message}") from e
    except ValueError as e:
        logger.error(f"Invalid request input: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error preparing input: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error preparing input") from e

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]

        if response_type == "values":
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            interrupt_value = response["__interrupt__"][0].value
            output = langchain_to_chat_message(
                AIMessage(content=interrupt_value if isinstance(interrupt_value, str) else str(interrupt_value))
            )
        else:
            logger.error(f"Unexpected response type: {response_type}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response type: {response_type}",
            )

        output.run_id = str(run_id)
        thread_id_used = kwargs.get("config", {}).get("configurable", {}).get("thread_id")
        if thread_id_used:
            output.thread_id = thread_id_used
        return output
    except HTTPException:
        raise
    except ChatbotServerError as e:
        logger.error(f"Server error during invocation: {e}")
        raise HTTPException(status_code=500, detail=e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error during invocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error during agent invocation") from e


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(
    user_input: UserInput,
    agent_id: str | None = None,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    The deprecated request user_id field is ignored; identity comes from the
    authenticated session.
    Provider and model can be specified in the request to use different LLMs.
    """
    agent_id = agent_id or get_default_agent()
    user_input.thread_id = await _authorize_thread(session, user, user_input.thread_id)

    # Extract provider and model from request
    provider = user_input.provider.value if user_input.provider else None
    model = user_input.model
    
    try:
        agent: CompiledStateGraph = await get_agent(agent_id, provider=provider, model=model)
    except AgentNotFoundError as e:
        logger.error(f"Agent not found: {e}")
        raise HTTPException(status_code=404, detail=e.message)
    
    try:

        prepared = await _prepare_agent_invocation(
            agent=agent,
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            user_id=str(user.id),
            provider=provider,
            model=model,
        )
        kwargs = prepared.kwargs
        run_id = prepared.run_id
    except StateError as e:
        logger.error(f"Failed to prepare input: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare input: {e.message}")
    except ValueError as e:
        logger.error(f"Invalid request input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error preparing input: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error preparing input")

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            interrupt_value = response["__interrupt__"][0].value
            output = langchain_to_chat_message(
                AIMessage(content=interrupt_value if isinstance(interrupt_value, str) else str(interrupt_value))
            )
        else:
            logger.error(f"Unexpected response type: {response_type}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response type: {response_type}"
            )

        output.run_id = str(run_id)
        thread_id_used = kwargs.get("config", {}).get("configurable", {}).get("thread_id")
        if thread_id_used:
            output.thread_id = thread_id_used
        return output
    except HTTPException:
        raise
    except ChatbotServerError as e:
        logger.error(f"Server error during invocation: {e}")
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during invocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error during agent invocation")


@router.post("/{agent_id}/invoke_with_files")
@router.post("/invoke_with_files")
async def invoke_with_files(
    agent_id: str | None = None,
    message: str = Form(""),
    thread_id: str | None = Form(None),
    user_id: str | None = Form(None),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    attachments: list[UploadFile] | None = File(default=None),
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ChatMessage:
    """Invoke an agent with multipart form data and text attachments."""

    agent_id = agent_id or get_default_agent()
    user_input = UserInput(
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
    if not attachments:
        return await invoke(user_input, agent_id=agent_id, user=user, session=session)

    user_input.thread_id = await _authorize_thread(session, user, user_input.thread_id)
    return await _invoke_with_attachments(
        user_input,
        authenticated_user_id=str(user.id),
        agent_id=agent_id,
        attachments=attachments,
    )


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
    The deprecated request user_id field is ignored; identity comes from the
    authenticated session.
    """
    user_input.thread_id = await _authorize_thread(session, user, user_input.thread_id)
    return EventSourceResponse(
        message_generator(user_input, str(user.id), agent_id or get_default_agent()),
    )


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


# ------------------------------------------------------------------
# Discovery endpoints
# ------------------------------------------------------------------

@router.get("/agents")
async def get_agents(_user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return all registered agents and the current default."""
    return {
        "agents": list_registered_agents(),
        "default": get_default_agent(),
    }


@router.get("/repositories")
async def get_repositories(
    _user: AuthenticatedUser = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Return metadata for all configured software repositories."""
    from juena.indexing.repo_manager import RepoManager

    mgr = RepoManager()
    return mgr.list_repo_metadata()
