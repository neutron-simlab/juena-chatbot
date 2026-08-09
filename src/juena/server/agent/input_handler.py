"""
Agent input preparation utilities for LangGraph agents.

This module provides utilities for preparing input for LangGraph agent execution.
LangGraph automatically resumes from checkpoint when invoked with the same thread_id.
"""

from typing import Any, NamedTuple, Sequence
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from juena.core.log import get_logger
from juena.server.agent.registry import _normalize_provider_model
from juena.server.chat.input_constants import (
    DISPLAY_ATTACHMENTS_KEY,
    DISPLAY_TEXT_KEY,
)
from juena.server.chat.input_types import UploadedAttachment
from juena.server.agent.runtime_model_middleware import RuntimeModelContext

logger = get_logger(__name__)


class AgentRunContext(NamedTuple):
    """The per-run identity of an invocation, independent of its input."""

    config: RunnableConfig
    context: RuntimeModelContext
    run_id: UUID
    thread_id: str


class AgentInputHandler:
    """Handles input preparation for LangGraph agents."""

    @staticmethod
    def build_run_context(
        thread_id: str | None = None,
        user_id: str | None = None,
        run_id: UUID | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AgentRunContext:
        """Build the config and runtime context for one agent run.

        Separate from ``build_input`` because staging uploaded files has to read
        the thread's existing state *before* the input is known, and that read
        needs the very same config the run will use.
        """
        run_id = run_id or uuid4()
        thread_id = thread_id or str(uuid4())
        if not user_id:
            raise ValueError("A trusted authenticated user_id is required")
        provider, model = _normalize_provider_model(provider, model)

        configurable: dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "provider": provider,
            "model": model,
        }
        return AgentRunContext(
            config=RunnableConfig(configurable=configurable, run_id=run_id),
            context=RuntimeModelContext(
                provider=provider,
                model=model,
                thread_id=thread_id,
                user_id=user_id,
            ),
            run_id=run_id,
            thread_id=thread_id,
        )

    @staticmethod
    def build_input(
        user_input: str,
        *,
        message_override: str | None = None,
        initial_files: dict[str, Any | None] | None = None,
        display_attachments: Sequence[UploadedAttachment] | None = None,
    ) -> dict[str, Any]:
        """Build the graph input for one human turn.

        Only graph state channels belong here. thread_id and user_id are not
        channels of the agent's state schema, so LangGraph drops them and logs
        "Input channel ... not found" for each one on every invocation. They
        already travel in config.configurable and RuntimeModelContext.

        When staged inputs replace the content, the model reads the manifest but
        the transcript must still show what the user typed. The checkpointer is
        the only store of message content, so the original text and the upload
        names ride along in ``additional_kwargs``, which LangChain serialises.
        ``langchain_to_chat_message`` unpacks them for both live streaming and
        history replay.
        """
        display_kwargs: dict[str, Any] = {}
        if message_override is not None and message_override != user_input:
            display_kwargs[DISPLAY_TEXT_KEY] = user_input
        if display_attachments:
            display_kwargs[DISPLAY_ATTACHMENTS_KEY] = [
                {"name": attachment.original_filename, "chars": attachment.char_count}
                for attachment in display_attachments
            ]

        input_data: dict[str, Any] = {
            "messages": [
                HumanMessage(
                    content=message_override or user_input,
                    additional_kwargs=display_kwargs,
                )
            ],
        }
        if initial_files is not None:
            input_data["files"] = initial_files
        return input_data
