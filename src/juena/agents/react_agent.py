"""
Example: React Q&A Agent with LangChain create_agent

This example shows how to create a react chatbot agent using LangChain's
create_agent function and register it with JueNA.
"""
from typing import TypedDict, List, Tuple, Any
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from juena.core.llms_providers import create_llm_with_fallback
from juena.server.agent_registry import register_agent_factory
from juena.server.checkpointer import get_checkpointer

# Step 1: Define state
class ReactAgentState(TypedDict):
    messages: List[BaseMessage]


# Step 2: Create tools (optional)
@tool
def get_weather(location: str) -> str:
    """Get weather for a location.
    
    Args:
        location: The city or location name
        
    Returns:
        Weather information for the location
    """
    # In a real implementation, you would call a weather API here
    return f"Weather in {location}: Sunny, 72°F"


tools = [get_weather]


# Step 3: Create agent factory
async def create_react_agent(
    provider: str,
    model: str
) -> Tuple[Any, CompiledStateGraph]:
    """
    Create a react Q&A agent with weather and calculator tools.
    
    This factory function is called by the template to create your agent.
    It receives the provider and model from the configuration, and returns
    a tuple of (agent_instance, compiled_graph).
    
    Args:
        provider: LLM provider (e.g., 'openai', 'blablador')
        model: Model name (e.g., 'gpt-4o-mini')
    
    Returns:
        Tuple of (agent_instance, compiled_graph)
        - agent_instance: Can be None or your agent object
        - compiled_graph: The CompiledStateGraph to use
    """
    # Create LLM using template's utility function
    llm = create_llm_with_fallback(provider=provider, model=model)
    
    # Create system prompt
    system_prompt = """You are a helpful Q&A assistant.
    
    You can answer questions, get weather information, and perform calculations.
    
    When users ask about weather, use the get_weather tool.
    When users ask for calculations, use the calculate tool.
    
    Always be polite and helpful."""
    
    react_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=get_checkpointer(),
        name="react_agent"
    )
    
    # Return (agent_instance, compiled_graph)
    # For simple agents, agent_instance can be None
    return (None, react_agent)




# Step 4: Register agent factory
# This must be done before starting the server
# You can import this module in main.py or create a separate registration file
register_agent_factory("react_agent", create_react_agent, set_as_default=True)