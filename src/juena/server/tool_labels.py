"""How JüNA's own tools read in the activity log.

Core describes the tools it installs (files, `execute`, `task`, `ask_user`);
these are the ones this deployment adds. A tool missing here still shows, as
its name and first argument. See `juena_core.server.streaming.tool_labels`.
"""

from typing import Any

from juena_core.server.streaming.tool_labels import ToolLabel, code, describe, quote, short


def _repo(args: dict[str, Any]) -> str:
    # "all" is the repo id `search_code_semantic` takes for every repository.
    repo_id = short(args.get("repo_id"))
    return "all repositories" if repo_id.lower() in {"", "all"} else code(repo_id)


def _specialist(args: dict[str, Any]) -> str:
    return short(args.get("specialist") or "specialist").replace("-", " ")


JUENA_TOOL_LABELS: dict[str, ToolLabel] = {
    "search_code_semantic": lambda args: describe(
        f"Search the code of {_repo(args)}", quote(args.get("query")), sep=": "
    ),
    "search_docs_local": lambda args: describe(
        f"Search the docs of {_repo(args)}", quote(args.get("query")), sep=": "
    ),
    "list_indexed_repositories": lambda args: "List the indexed repositories",
    "start_research": lambda args: describe(
        f"Start background research with the {_specialist(args)}",
        quote(args.get("objective")),
        sep=": ",
    ),
    "check_research": lambda args: "Check on background research",
    "list_research": lambda args: "List background research",
    "tavily_search": lambda args: describe("Search the web", quote(args.get("query")), sep=": "),
}
