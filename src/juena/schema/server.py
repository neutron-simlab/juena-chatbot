from typing import Any, Literal, NotRequired, Optional, Dict, List
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from juena.schema.llm_models import AllModelEnum, BlabladorModelName, OpenAIModelName, Provider


# =================
# CORE API MODELS
# =================

class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["my_agent"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A helpful chatbot agent"],
    )
    capabilities: List[str] = Field(
        description="List of agent capabilities.",
        default=[],
        examples=[["chat", "tool_use", "streaming"]],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: List[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: List[str] = Field(
        description="List of available LLM model names.",
    )
    providers: List[Provider] = Field(
        description="List of available LLM providers.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["my_agent"],
    )
    default_model: str = Field(
        description="Default model used when none is specified.",
        examples=["gpt-4o-mini"],
    )
    default_provider: Provider = Field(
        description="Default provider used when none is specified.",
        default=Provider.OPENAI,
    )


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
# INTERRUPT MODELS
# =================

class ModuleInterruptInput(BaseModel):
    """Input for responding to a module interrupt."""
    
    interrupt_message: str = Field(
        description="User's response to the module interrupt.",
        examples=["I want to use default settings", "Configure custom parameters"],
    )
    message: str = Field(
        description="User's response to the module interrupt (alias for interrupt_message).",
        default="",
    )
    thread_id: str = Field(
        description="Thread ID of the conversation (shared by supervisor and modules).",
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


class ModuleInterruptResponse(BaseModel):
    """Response containing module interrupt information."""
    
    module_name: str = Field(
        description="Name of the module that requires input.",
        examples=["agent"],
    )
    interrupt_value: str = Field(
        description="The interrupt message/question from the module.",
        examples=["Please choose your configuration mode: Default Setup or Customize?"],
    )
    thread_id: str = Field(
        description="Thread ID of the conversation (shared by supervisor and modules).",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    interrupt_count: int = Field(
        description="Number of interrupts for this module (1, 2, 3, etc.).",
        examples=[1, 2, 3],
        default=1,
    )
    message: str = Field(
        description="Human-readable message about the interrupt.",
        examples=["Module 'agent' requires input: Please choose your configuration mode"],
    )
    created_at: datetime = Field(
        description="Timestamp when the interrupt was created.",
        default_factory=datetime.now,
    )


# =================
# FEEDBACK MODELS
# =================

class Feedback(BaseModel):
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: Dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


# =================
# HISTORY MODELS
# =================

class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    """Chat history response."""
    
    messages: List[ChatMessage] = Field(
        description="List of chat messages.",
    )
    thread_id: str = Field(
        description="Thread ID of the conversation.",
    )
    total_messages: int = Field(
        description="Total number of messages in the conversation.",
    )
    last_updated: Optional[datetime] = Field(
        description="Last time the conversation was updated.",
        default=None,
    )


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
