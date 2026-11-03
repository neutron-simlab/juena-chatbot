"""Web search with the current caller's personal key."""

from langchain.tools import ToolRuntime, tool
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

TAVILY_MAX_RESULTS = 2
TAVILY_TOPIC = "general"
TAVILY_SEARCH_DEPTH = None
TAVILY_INCLUDE_ANSWER = False
TAVILY_INCLUDE_RAW_CONTENT = "markdown"
TAVILY_INCLUDE_IMAGES = False
TAVILY_INCLUDE_IMAGE_DESCRIPTIONS = False


def load_optional_tavily_tool():
    # The graph owns only this wrapper. Credentials and clients are per call.
    return tavily_search


@tool
async def tavily_search(query: str, runtime: ToolRuntime[RuntimeModelContext]) -> dict:
    """Search the web for current facts and external evidence."""
    from langchain_tavily import TavilySearch
    from juena.server.credentials import CredentialError, personal_key

    try:
        key = await personal_key(runtime.context.user_id, "tavily")
        client = TavilySearch(
            tavily_api_key=key, max_results=TAVILY_MAX_RESULTS, topic=TAVILY_TOPIC,
            search_depth=TAVILY_SEARCH_DEPTH, include_answer=TAVILY_INCLUDE_ANSWER,
            include_raw_content=TAVILY_INCLUDE_RAW_CONTENT, include_images=TAVILY_INCLUDE_IMAGES,
            include_image_descriptions=TAVILY_INCLUDE_IMAGE_DESCRIPTIONS,
        )
        result = await client.ainvoke({"query": query}, config={"callbacks": []})
        if isinstance(result, dict) and result.get("error"):
            return {"error": "Tavily could not complete the search. Check your key in My profile or retry later."}
        return result
    except CredentialError as error:
        return {"error": str(error)}
    except Exception:
        return {"error": "Tavily could not complete the search. Check your key in My profile or retry later."}
