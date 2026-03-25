"""
API endpoints for agent invocation and streaming.

This module provides endpoints for invoking agents, streaming responses,
and restarting agents with new configurations.
"""
import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.server.agent_registry import (
    DEFAULT_AGENT,
    get_agent,
    list_registered_agents,
    restart_agent,
)
from juena.schema.server import ChatMessage, StreamInput, UserInput
from juena.core.config import global_config
from juena.schema.llm_models import Provider, get_default_model_for_provider
from juena.server.utils import langchain_to_chat_message, set_thread_id_env
from juena.server.errors import (
    AgentNotFoundError,
    StreamingError,
    StateError,
    ChatbotServerError
)
from juena.server.agent_input_handler import AgentInputHandler
from juena.server.checkpointer import get_checkpointer
from juena.server.code_chat_inputs import prepare_code_chat_turn_inputs
from juena.server.streaming import StreamEventProcessor

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
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
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

    if attachments and agent_id != "code_chat_agent":
        raise ValueError("File attachments are only supported by code_chat_agent.")

    kwargs, run_id = await AgentInputHandler.prepare_input(
        message,
        thread_id=thread_id,
        user_id=user_id,
        provider=provider,
        model=model,
    )
    effective_message = message

    if agent_id == "code_chat_agent":
        prepared_inputs = await prepare_code_chat_turn_inputs(
            agent,
            kwargs["config"],
            message,
            attachments=attachments,
        )
        if prepared_inputs is not None:
            kwargs, run_id = await AgentInputHandler.prepare_input(
                message,
                thread_id=thread_id,
                user_id=user_id,
                run_id=run_id,
                provider=provider,
                model=model,
                message_override=prepared_inputs.message_override,
                initial_files=prepared_inputs.files_update or None,
            )
            effective_message = prepared_inputs.message_override

    return PreparedAgentInvocation(
        kwargs=kwargs,
        run_id=run_id,
        effective_user_message=effective_message,
    )


async def message_generator(
    user_input: StreamInput,
    agent_id: str = DEFAULT_AGENT,
    attachments: list[UploadFile] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    # Extract provider and model from request
    provider = user_input.provider.value if user_input.provider else None
    model = user_input.model
    
    try:
        agent: CompiledStateGraph = await get_agent(agent_id, provider=provider, model=model)
    except AgentNotFoundError as e:
        logger.error(f"Agent not found: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Agent not found: {e.message}'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    try:
        # Set thread_id as environment variable for tools that need it
        if user_input.thread_id:
            set_thread_id_env(user_input.thread_id)

        prepared = await _prepare_agent_invocation(
            agent=agent,
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            user_id=user_input.user_id,
            provider=provider,
            model=model,
            attachments=attachments,
        )
        kwargs = prepared.kwargs
        run_id = prepared.run_id
    except StateError as e:
        logger.error(f"Failed to prepare input: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to prepare input: {e.message}'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except ValueError as e:
        logger.error(f"Invalid request input: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        logger.error(f"Unexpected error preparing input: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error preparing input'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    # Emit thread_id first so clients can use it for follow-up messages (checkpointer persistence)
    thread_id_used = kwargs.get("config", {}).get("configurable", {}).get("thread_id")
    if thread_id_used:
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id_used})}\n\n"
    
    try:
        # Create stream event processor
        processor = StreamEventProcessor(
            agent,
            kwargs["config"],
            str(run_id),
            prepared.effective_user_message,
        )
        
        # Initialize streamed_message_ids from existing state to prevent duplicates
        # This ensures that when the graph resumes from a checkpoint, old messages
        # are not re-streamed to the client
        await processor._initialize_streamed_message_ids()
        
        # Process streamed events from the graph and yield messages over the SSE stream
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            async for sse_string in processor.process_event(stream_event):
                yield sse_string
        
    except asyncio.CancelledError:
        # Client disconnected or request cancelled; re-raise so the task is properly cancelled
        logger.debug("Stream cancelled (client disconnect or request cancelled)")
        raise
    except StreamingError as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming error: {e.message}'})}\n\n"
    except Exception as e:
        logger.error(f"Unexpected error in message generator: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def _invoke_with_attachments(
    user_input: UserInput,
    *,
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
        if user_input.thread_id:
            set_thread_id_env(user_input.thread_id)

        prepared = await _prepare_agent_invocation(
            agent=agent,
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            user_id=user_input.user_id,
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
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    Provider and model can be specified in the request to use different LLMs.
    """
    # Extract provider and model from request
    provider = user_input.provider.value if user_input.provider else None
    model = user_input.model
    
    try:
        agent: CompiledStateGraph = await get_agent(agent_id, provider=provider, model=model)
    except AgentNotFoundError as e:
        logger.error(f"Agent not found: {e}")
        raise HTTPException(status_code=404, detail=e.message)
    
    try:
        # Set thread_id as environment variable for tools that need it
        if user_input.thread_id:
            set_thread_id_env(user_input.thread_id)

        prepared = await _prepare_agent_invocation(
            agent=agent,
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            user_id=user_input.user_id,
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
    agent_id: str = DEFAULT_AGENT,
    message: str = Form(""),
    thread_id: str | None = Form(None),
    user_id: str | None = Form(None),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    attachments: list[UploadFile] | None = File(default=None),
) -> ChatMessage:
    """Invoke an agent with multipart form data and text attachments."""

    user_input = UserInput(
        message=message,
        thread_id=_form_field(thread_id),
        user_id=_form_field(user_id),
    )
    if (provider_value := _form_field(provider)) is not None:
        try:
            user_input.provider = Provider(provider_value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider_value}") from e
    if (model_value := _form_field(model)) is not None:
        user_input.model = model_value
    return await invoke(user_input, agent_id=agent_id) if not attachments else await _invoke_with_attachments(
        user_input,
        agent_id=agent_id,
        attachments=attachments,
    )


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. 
    run_id kwarg is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post(
    "/{agent_id}/stream_with_files",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream_with_files", response_class=StreamingResponse, responses=_sse_response_example())
async def stream_with_files(
    agent_id: str = DEFAULT_AGENT,
    message: str = Form(""),
    thread_id: str | None = Form(None),
    user_id: str | None = Form(None),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    attachments: list[UploadFile] | None = File(default=None),
) -> StreamingResponse:
    """Stream an agent response from multipart form data and text attachments."""

    user_input = StreamInput(
        message=message,
        thread_id=_form_field(thread_id),
        user_id=_form_field(user_id),
    )
    if (provider_value := _form_field(provider)) is not None:
        try:
            user_input.provider = Provider(provider_value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider_value}") from e
    if (model_value := _form_field(model)) is not None:
        user_input.model = model_value
    return StreamingResponse(
        message_generator(user_input, agent_id, attachments=attachments),
        media_type="text/event-stream",
    )


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, Any]:
    """Delete all persisted LangGraph state for a conversation thread."""

    try:
        await get_checkpointer().adelete_thread(thread_id)
        return {
            "status": "success",
            "thread_id": thread_id,
            "message": f"Thread {thread_id} deleted successfully",
        }
    except RuntimeError as e:
        logger.error("Checkpointer unavailable while deleting thread %s: %s", thread_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to delete thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete thread state") from e


@router.post("/{agent_id}/restart")
@router.post("/restart")
async def restart(
    agent_id: str = DEFAULT_AGENT,
    provider: str | None = Query(None, description="New LLM provider (openai or blablador)"),
    model: str | None = Query(None, description="New LLM model name"),
) -> dict[str, Any]:
    """
    Restart an agent with new provider/model configuration.
    
    This endpoint forces reinitialization of the agent graph with new LLM configuration,
    similar to refreshing the web page but keeping the new provider/model.
    
    Args:
        agent_id: Agent identifier (e.g., "my_agent")
        provider: New LLM provider (optional, uses current if not provided)
        model: New LLM model name (optional, uses current if not provided)
    
    Returns:
        Dictionary with restart status and agent info
    """
    try:
        agent = await restart_agent(agent_id, provider=provider, model=model)
        
        # Determine the actual provider/model used
        actual_provider = provider or global_config.DEFAULT_PROVIDER
        if model is None:
            try:
                provider_enum = Provider(actual_provider)
                actual_model = get_default_model_for_provider(provider_enum)
            except ValueError:
                actual_model = global_config.DEFAULT_MODEL
        else:
            actual_model = model
        
        return {
            "status": "success",
            "message": f"Agent {agent_id} restarted successfully",
            "provider": actual_provider,
            "model": actual_model,
            "agent_id": agent_id
        }
    except AgentNotFoundError as e:
        logger.error(f"Agent not found for restart: {e}")
        raise HTTPException(status_code=404, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to restart agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restart agent: {str(e)}")


# ------------------------------------------------------------------
# Discovery endpoints
# ------------------------------------------------------------------

@router.get("/agents")
async def get_agents() -> dict[str, Any]:
    """Return all registered agents and the current default."""
    return {
        "agents": list_registered_agents(),
        "default": DEFAULT_AGENT,
    }


@router.get("/repositories")
async def get_repositories() -> list[dict[str, Any]]:
    """Return metadata for all configured software repositories."""
    from juena.retrieval.repo_manager import RepoManager

    mgr = RepoManager()
    return mgr.list_repo_metadata()
