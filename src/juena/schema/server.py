from typing import Any, Literal, NotRequired, Optional, Dict, List
from datetime import datetime

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from juena.schema.llm_models import Provider


# =================
# REQUEST/RESPONSE MODELS
# =================

class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["Hello, how can you help me?"],
    )
    thread_id: Optional[str] = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: Optional[str] = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["user_123"],
    )
    provider: Optional[Provider] = Field(
        description="LLM provider to use (openai or blablador).",
        default=None,
        examples=["openai", "blablador"],
    )
    model: Optional[str] = Field(
        description="LLM model name to use (provider-specific).",
        default=None,
        examples=["gpt-4o-mini", "GPT-OSS-120b"],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""
    pass  # Inherits all fields from UserInput


# =================
# MESSAGE MODELS
# =================

class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: Dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom", "system"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom", "system"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: List[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: Optional[str] = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: Optional[str] = Field(
        description="ID of this single invocation (one turn). Use for feedback/tracing (e.g. record_feedback(run_id=...)).",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    thread_id: Optional[str] = Field(
        description="ID of the conversation (whole chat). Use for follow-up messages to persist context.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: Dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: Dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )
    timestamp: Optional[datetime] = Field(
        description="Timestamp when the message was created.",
        default=None,
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


# =================
# HEALTH CHECK MODELS
# =================

class HealthStatus(BaseModel):
    """Health check response."""
    
    status: Literal["ok", "error"] = Field(
        description="Overall health status.",
    )
    timestamp: datetime = Field(
        description="Timestamp of the health check.",
        default_factory=datetime.now,
    )
    version: str = Field(
        description="Service version.",
        default="0.1.0",
    )
    uptime: Optional[float] = Field(
        description="Service uptime in seconds.",
        default=None,
    )
    details: Dict[str, Any] = Field(
        description="Additional health check details.",
        default={},
    )
