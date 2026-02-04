# juena-chatbot Template

A general-purpose chatbot template that provides a complete UI+Server+Client infrastructure for building LangGraph-based chatbots. Simply plug in your own LangGraph graph and start chatting!

## Features

- **LangGraph Integration**: Built-in support for any LangGraph agent
- **RESTful API Server**: FastAPI-based server with streaming support
- **Web Interface**: Streamlit-based chat UI with real-time streaming
- **Multiple LLM Providers**: Support for OpenAI and Blablador (OpenAI-compatible)
- **Real-time Streaming**: Server-Sent Events (SSE) for live conversation streaming
- **Thread Management**: Multi-turn conversation support with thread persistence
- **Pluggable Architecture**: Easy to customize and extend

## Architecture

```
┌─────────────┐
│  Streamlit  │  Web UI (Port 8501)
│     UI      │
└──────┬──────┘
       │ HTTP/SSE
       ▼
┌─────────────┐
│   FastAPI   │  API Server (Port 8000)
│   Server    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LangGraph  │  Your Custom Agent
│    Agent    │
└─────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

1. **Create `.env` file:**
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

2. **Build and start:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - API Server: http://localhost:8000
   - Web UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

See the Quick Start above for Docker usage.

### Option 2: Local Development

1. **Install Dependencies:**
   ```bash
   cd juena-chatbot
   pip install -e .
   ```

2. **Configure Environment:**
   Create a `.env` file:
   ```bash
   # Required: At least one LLM provider
   OPENAI_API_KEY=sk-your-openai-api-key-here
   DEFAULT_PROVIDER=openai

   # OR for Blablador
   # BLABLADOR_API_KEY=your-blablador-api-key-here
   # BLABLADOR_BASE_URL=https://api.helmholtz-blablador.fz-juelich.de/v1/
   # DEFAULT_PROVIDER=blablador
   ```

3. **Create Your Agent:**
   Create a file `my_agent.py`:
   ```python
   from langchain.agents import create_agent
   from juena.core.llms_providers import create_llm_with_fallback
   from juena.server.agent_registry import register_agent_factory

   async def create_my_agent(provider: str, model: str):
       llm = create_llm_with_fallback(provider=provider, model=model)
       
       react_agent = create_agent(
           model=llm,
           tools=[],  # Add your tools here
           system_prompt="You are a helpful assistant.",
           name="my_agent"
       )
       
       return (None, react_agent)

   register_agent_factory("my_agent", create_my_agent, set_as_default=True)
   ```

4. **Register Your Agent:**
   In `main.py`, import your agent before starting the server:
   ```python
   import my_agent  # This registers your agent
   ```

5. **Start the Server:**
   ```bash
   python main.py
   ```

6. **Start the UI:**
   In another terminal:
   ```bash
   streamlit run app/streamlit_app.py
   ```

   Access the UI at `http://localhost:8501`

## Creating Your Own Agent

### Introduction

This template uses LangChain's `create_agent` function to create react-agents. A react-agent follows the "Reasoning + Acting" pattern, where the agent:

1. **Reasons** - Uses an LLM to think through the problem and plan what tools to use
2. **Acts** - Executes an action using available tools
3. **Observes** - Gets feedback from the action
4. **Repeats** - Uses observations to inform the next reasoning step

The agent factory pattern allows you to plug in any LangGraph graph. The template handles all the infrastructure (server, client, UI, streaming) so you can focus on building your agent logic.

**Reference**: [LangChain Agents Documentation](https://docs.langchain.com/oss/python/langchain/agents)

### Understanding LangChain's `create_agent`

The `create_agent` function creates a react-agent graph:

```python
from langchain.agents import create_agent

react_agent = create_agent(
    model: BaseChatModel,           # The LLM to use
    tools: List[BaseTool],          # Tools the agent can use (optional)
    system_prompt: str,             # System prompt for the agent
    name: str,                      # Name for the agent
    middleware: Optional[List[AgentMiddleware]] = None  # Optional middleware
) -> CompiledStateGraph
```

**Parameters:**
- `model`: The language model (created using `create_llm_with_fallback`)
- `tools`: List of tools the agent can call (can be empty)
- `system_prompt`: Instructions for the agent's behavior
- `name`: Identifier for the agent
- `middleware`: Optional middleware for message processing

**Returns:** A `CompiledStateGraph` that can be used directly or wrapped in a larger LangGraph.

### Step-by-Step Agent Creation Guide

#### Step 1: Define Your State Schema

LangGraph requires a state schema. For simple react-agents created with `create_agent`, the state is handled automatically. However, if you need custom state, define it:

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class MyAgentState(TypedDict):
    messages: List[BaseMessage]  # Required: messages list
    # Add your custom state fields here
    user_preferences: dict
    current_task: str
```

**Note:** The `messages` list is required for LangGraph agents. You can add additional fields as needed.

#### Step 2: Create Tools (Optional)

Tools allow your agent to interact with external systems. Create tools using the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for information.
    
    Args:
        query: Search query string
        
    Returns:
        Search results as a string
    """
    # Your tool implementation
    return "Search results..."

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body
        
    Returns:
        Status message
    """
    # Your email sending logic
    return f"Email sent to {to}"

tools = [search_database, send_email]
```

**Tool Documentation:** The docstring is crucial - the LLM uses it to understand when and how to call the tool. Include:
- A clear description of what the tool does
- Args section listing all parameters
- Returns section describing the return value

#### Step 3: Create LLM Instance

Use the template's LLM provider utilities:

```python
from juena.core.llms_providers import create_llm_with_fallback

llm = create_llm_with_fallback(provider=provider, model=model)
```

This function:
- Creates an LLM instance for the specified provider/model
- Handles fallback to other available providers if the primary fails
- Uses configuration from your `.env` file

#### Step 4: Create System Prompt

Write a clear system prompt that guides your agent's behavior:

```python
system_prompt = """You are a helpful assistant that can search databases and send emails.

When you need information, use the search_database tool.
When users ask you to send emails, use the send_email tool.

Always be polite and helpful. If you don't know something, say so."""
```

**Best Practices:**
- Be specific about when to use each tool
- Set the agent's personality and tone
- Include examples if helpful
- Keep it concise but comprehensive

#### Step 5: Create React-Agent Using `create_agent`

Now create your react-agent:

```python
from langchain.agents import create_agent

# Create the react-agent
react_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    name="my_agent",
    middleware=None  # Optional middleware
)
```

**What happens:**
- `create_agent` returns a `CompiledStateGraph`
- The graph implements the react-agent loop: agent → tools → agent → ... until done
- The agent automatically handles tool calling, error handling, and completion

#### Step 6: Wrap in LangGraph StateGraph (If Needed)

For simple react-agents, you can use the result directly:

```python
# For simple react-agents, use directly
compiled_graph = react_agent
```

If you need custom nodes (preprocessing, postprocessing, conditional routing), wrap it:

```python
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

# Create wrapper graph if you need custom nodes
workflow = StateGraph(MyAgentState)
workflow.add_node("preprocess", preprocess_node)  # Your custom node
workflow.add_node("agent", react_agent)
workflow.add_node("postprocess", postprocess_node)  # Your custom node

workflow.add_edge(START, "preprocess")
workflow.add_edge("preprocess", "agent")
workflow.add_edge("agent", "postprocess")
workflow.add_edge("postprocess", END)

compiled_graph = workflow.compile(checkpointer=InMemorySaver())
```

#### Step 7: Create Agent Factory Function

Create an async factory function that the template will call:

```python
from typing import Tuple, Any
from langgraph.graph.state import CompiledStateGraph

async def create_my_agent(
    provider: str, 
    model: str
) -> Tuple[Any, CompiledStateGraph]:
    """
    Factory function to create your agent.
    
    Args:
        provider: LLM provider (e.g., 'openai', 'blablador')
        model: Model name (e.g., 'gpt-4o-mini')
    
    Returns:
        Tuple of (agent_instance, compiled_graph)
        - agent_instance: Can be None or your agent object (for accessing agent methods)
        - compiled_graph: The CompiledStateGraph to use
    """
    # Step 1: Create LLM
    llm = create_llm_with_fallback(provider=provider, model=model)
    
    # Step 2: Create tools (if any)
    tools = [search_database, send_email]  # Your tools here
    
    # Step 3: Create system prompt
    system_prompt = "You are a helpful assistant..."
    
    # Step 4: Create react-agent
    react_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        name="my_agent"
    )
    
    # Step 5: Return (agent_instance, compiled_graph)
    # For simple agents, agent_instance can be None
    return (None, react_agent)
```

**Important:** The factory function must be `async` and return a tuple of `(AgentInstance, CompiledStateGraph)`.

#### Step 8: Register Your Agent

Register your agent factory with the template:

```python
from juena.server.agent_registry import register_agent_factory

# Register your agent factory
register_agent_factory("my_agent", create_my_agent, set_as_default=True)
```

**Parameters:**
- `agent_id`: Unique identifier for your agent (used in API calls)
- `factory`: Your async factory function
- `set_as_default`: If True, sets this as the default agent

**When to register:** This must be done **before** starting the server. Common places:
- In `main.py` before `uvicorn.run()`
- In a separate `agents.py` file that you import in `main.py`
- In your agent file itself (as shown in the example)

#### Step 9: Start the Server

Update `main.py` to import your agent registration:

```python
# Import your agent (this registers it)
import my_agent  # or: from my_agent import create_my_agent; register_agent_factory("my_agent", create_my_agent)

from juena.server.service import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
```

### Complete Example: Simple Q&A Agent

Here's a complete, working example (`src/juena/agents/react_agent.py`):

```python
"""
Complete example: Simple Q&A Agent with LangChain create_agent
"""
from typing import TypedDict, List, Tuple, Any
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from juena.core.llms_providers import create_llm_with_fallback
from juena.server.agent_registry import register_agent_factory

# Step 1: Define state (optional for simple agents)
class QAAgentState(TypedDict):
    messages: List[BaseMessage]

# Step 2: Create tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    # In real implementation, call weather API
    return f"Weather in {location}: Sunny, 72°F"

tools = [get_weather]

# Step 3: Create agent factory
async def create_qa_agent(
    provider: str,
    model: str
) -> Tuple[Any, CompiledStateGraph]:
    """Create a Q&A agent with weather tool."""
    # Create LLM
    llm = create_llm_with_fallback(provider=provider, model=model)
    
    # Create system prompt
    system_prompt = """You are a helpful Q&A assistant.
    
    You can answer questions and get weather information.
    When users ask about weather, use the get_weather tool."""
    
    # Create react-agent
    react_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        name="qa_agent"
    )
    
    return (None, react_agent)

# Step 4: Register agent
register_agent_factory("qa_agent", create_qa_agent, set_as_default=True)
```

### Advanced Patterns

#### Pattern 1: Multi-Node Graph

Create a graph with multiple nodes for preprocessing and postprocessing:

```python
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

async def create_advanced_agent(provider: str, model: str):
    llm = create_llm_with_fallback(provider=provider, model=model)
    
    # Create react-agent
    react_agent = create_agent(
        model=llm,
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        name="agent"
    )
    
    # Create wrapper graph with custom nodes
    workflow = StateGraph(MyState)
    
    def preprocess_node(state):
        # Preprocess user input
        messages = state["messages"]
        # Add preprocessing logic
        return {"messages": messages}
    
    def postprocess_node(state):
        # Postprocess agent response
        messages = state["messages"]
        # Add postprocessing logic
        return {"messages": messages}
    
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("agent", react_agent)
    workflow.add_node("postprocess", postprocess_node)
    
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "agent")
    workflow.add_edge("agent", "postprocess")
    workflow.add_edge("postprocess", END)
    
    compiled = workflow.compile(checkpointer=InMemorySaver())
    return (None, compiled)
```

#### Pattern 2: Conditional Routing

Add conditional edges for dynamic routing:

```python
def route_after_agent(state):
    """Route based on agent output."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if agent needs user input
    if "needs_input" in last_message.content.lower():
        return "wait_for_input"
    else:
        return END

workflow.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "wait_for_input": "wait_for_input_node",
        END: END
    }
)
```

#### Pattern 3: Custom Middleware

Create middleware for logging or filtering:

```python
from langchain.agents import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    """Middleware that logs all messages."""
    def process_messages(self, messages):
        for msg in messages:
            print(f"Message: {msg.content}")
        return messages

# Use in create_agent
react_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt,
    name="my_agent",
    middleware=[LoggingMiddleware()]
)
```

#### Pattern 4: State Management

Manage custom state in your agent:

```python
class MyAgentState(TypedDict):
    messages: List[BaseMessage]
    user_preferences: dict
    conversation_count: int

def update_state_node(state):
    """Update custom state fields."""
    return {
        "conversation_count": state.get("conversation_count", 0) + 1
    }

workflow.add_node("update_state", update_state_node)
```

### Testing Your Agent

Test your agent locally before deploying:

```python
# test_agent.py
import asyncio
from my_agent import create_my_agent
from langchain_core.messages import HumanMessage

async def test():
    agent_instance, graph = await create_my_agent("openai", "gpt-4o-mini")
    
    # Test the graph
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hello!")]},
        config={"configurable": {"thread_id": "test-123"}}
    )
    
    print("Response:", result["messages"][-1].content)

asyncio.run(test())
```

### Troubleshooting

**Agent not responding:**
- Check that your agent is registered before starting the server
- Verify the agent_id matches what you registered
- Check server logs for errors

**Tools not being called:**
- Ensure tool docstrings are clear and descriptive
- Check that tools are properly decorated with `@tool`
- Verify the system prompt instructs the agent to use tools

**State not persisting:**
- Ensure you're using the same `thread_id` across requests
- Check that your graph is compiled with a checkpointer: `graph.compile(checkpointer=InMemorySaver())`

**Streaming not working:**
- Verify your LangGraph supports streaming (react-agents from `create_agent` do)
- Check that the server is running and accessible
- Look for errors in the browser console

### References

- **LangChain Agents Documentation**: https://docs.langchain.com/oss/python/langchain/agents
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **React Pattern Explanation**: https://langchain-tutorials.github.io/langchain-react-agent-pattern-2026/

## API Endpoints

### Agent Endpoints

- **POST `/{agent_id}/invoke`** or **POST `/invoke`** - Send message and get complete response
- **POST `/{agent_id}/stream`** or **POST `/stream`** - Real-time streaming response (SSE)
- **POST `/{agent_id}/restart`** or **POST `/restart`** - Restart agent with new configuration

### Health Check

- **GET `/health`** - Health check endpoint

See `/docs` for interactive API documentation.

## Project Structure

```
juena/
├── app/                    # Streamlit web interface
│   ├── streamlit_app.py    # Main Streamlit app
│   ├── chat_interface.py   # Chat UI logic
│   ├── sidebar.py          # Sidebar configuration
│   └── ui_components.py    # Reusable UI components
├── src/
│   └── juena/
│       ├── agents/         # Agent implementations
│       │   └── react_agent.py  # Example agent
│       ├── clients/        # HTTP client library
│       ├── server/         # FastAPI server
│       │   ├── service.py  # FastAPI app
│       │   ├── api_endpoints.py  # API routes
│       │   ├── agent_registry.py  # Agent factory registry
│       │   └── streaming/  # SSE streaming processors
│       ├── core/           # Core utilities
│       │   ├── config.py   # Configuration management
│       │   └── llms_providers.py  # LLM provider abstraction
│       └── schema/         # Pydantic schemas
├── main.py                # Server entry point
├── pyproject.toml         # Python project configuration
└── README.md              # This file
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-your-key-here
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini

# OR for Blablador
# BLABLADOR_API_KEY=your-key-here
# BLABLADOR_BASE_URL=https://api.helmholtz-blablador.fz-juelich.de/v1/
# DEFAULT_PROVIDER=blablador

# Optional: Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
UI_PORT=8501

# Optional: LangSmith tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=juena
```

## Customization

### Adding Custom Endpoints

Add custom endpoints in `src/juena/server/service.py`:

```python
from fastapi import APIRouter

custom_router = APIRouter()

@custom_router.get("/custom")
async def custom_endpoint():
    return {"message": "Custom endpoint"}

app.include_router(custom_router)
```

### Customizing UI

Modify Streamlit components in `app/`:
- `ui_components.py` - Message rendering, badges, colors
- `sidebar.py` - Configuration sidebar
- `chat_interface.py` - Chat logic and streaming

### Using Persistent Storage

Replace `InMemorySaver` with a persistent checkpointer:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
compiled = graph.compile(checkpointer=checkpointer)
```

## License

MIT License
