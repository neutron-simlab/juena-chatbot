"""How JüNA's own tools read in the activity log."""

from juena_core.server.streaming.tool_labels import describe_tool_call

from juena.server.tool_labels import JUENA_TOOL_LABELS
from juena.tools.repo_search import build_code_chat_tools
from juena.tools.research import build_research_tools


def test_every_label_names_a_tool_juena_builds() -> None:
    """A misspelt name would silently fall back to the generic line."""
    built = {tool.name for tool in build_code_chat_tools(None, None)}
    built |= {tool.name for tool in build_research_tools(["software-specialist"])}
    built.add("tavily_search")  # langchain-tavily's TavilySearch

    assert set(JUENA_TOOL_LABELS) <= built


def test_juena_tools_read_as_actions() -> None:
    def line(name: str, **args) -> str:
        return describe_tool_call(name, args, JUENA_TOOL_LABELS)

    assert line("search_code_semantic", repo_id="jscatter", query="sphere form factor") == (
        "Search the code of `jscatter`: “sphere form factor”"
    )
    assert line("search_docs_local", repo_id="all", query="install") == (
        "Search the docs of all repositories: “install”"
    )
    assert line(
        "start_research", specialist="neutron-research-specialist", objective="Find the flux"
    ) == "Start background research with the neutron research specialist: “Find the flux”"
    assert line("tavily_search", query="FRM II") == "Search the web: “FRM II”"
