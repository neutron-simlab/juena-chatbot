"""
API endpoints for agent invocation and streaming.

This module provides endpoints for invoking agents, streaming responses,
and restarting agents with new configurations.
"""
import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.server.agent_registry import DEFAULT_AGENT, get_agent, restart_agent
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
from juena.server.streaming import StreamEventProcessor

logger = get_logger(__name__)

router = APIRouter()


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


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
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
        
        kwargs, run_id = await AgentInputHandler.prepare_input(
            user_input.message,
            agent,
            thread_id=user_input.thread_id,
            user_id=user_input.user_id,
            provider=provider,
            model=model,
        )
    except StateError as e:
        logger.error(f"Failed to prepare input: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to prepare input: {e.message}'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        logger.error(f"Unexpected error preparing input: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error preparing input'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    try:
        # Create stream event processor
        processor = StreamEventProcessor(
            agent,
            kwargs["config"],
            str(run_id),
            user_input.message
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
        
        kwargs, run_id = await AgentInputHandler.prepare_input(
            user_input.message,
            agent,
            thread_id=user_input.thread_id,
            user_id=user_input.user_id,
            provider=provider,
            model=model,
        )
    except StateError as e:
        logger.error(f"Failed to prepare input: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare input: {e.message}")
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
        return output
    except HTTPException:
        raise
    except ChatbotServerError as e:
        logger.error(f"Server error during invocation: {e}")
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during invocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error during agent invocation")


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
