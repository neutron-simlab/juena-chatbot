"""Web search and selective page reading with the current caller's personal key."""

from langchain.tools import ToolRuntime, tool
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

TAVILY_MAX_RESULTS = 2
TAVILY_TOPIC = "general"
TAVILY_SEARCH_DEPTH = None
TAVILY_INCLUDE_ANSWER = False
TAVILY_INCLUDE_RAW_CONTENT = False
TAVILY_INCLUDE_IMAGES = False
TAVILY_INCLUDE_IMAGE_DESCRIPTIONS = False


def load_optional_tavily_tool():
    # The graph owns only this wrapper. Credentials and clients are per call.
    return tavily_search


@tool
async def tavily_search(query: str, runtime: ToolRuntime[RuntimeModelContext]) -> dict:
    """Search the web for relevant excerpts and source URLs, without full pages."""
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


@tool
async def tavily_extract(
    url: str, runtime: ToolRuntime[RuntimeModelContext], query: str | None = None,
) -> str:
    """Read one selected web page for exact wording, numbers, or table headers.

    Supply query to retrieve relevant passages for prose. Omit query to read the
    full page when table headers, legends, or surrounding context are needed.
    """
    from langchain_tavily import TavilyExtract
    from juena.server.credentials import CredentialError, personal_key

    failure = "Tavily could not read this page. Check the URL and your key in My profile or retry later."
    try:
        key = await personal_key(runtime.context.user_id, "tavily")
        client = TavilyExtract(
            tavily_api_key=key, extract_depth="advanced", format="markdown",
            include_images=False, chunks_per_source=3 if query else None,
        )
        result = await client.ainvoke({"urls": [url], "query": query}, config={"callbacks": []})
        if not isinstance(result, dict) or result.get("error") or not result.get("results"):
            return failure
        page = result["results"][0]
        if not page.get("raw_content"):
            return failure
        # Keep real line breaks so offloaded tables can be read by line range.
        return f"Source: {page['url']}\n\n{page['raw_content']}"
    except CredentialError as error:
        return str(error)
    except Exception:
        return failure
