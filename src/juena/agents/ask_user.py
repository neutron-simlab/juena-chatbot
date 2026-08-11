"""A tool that pauses the graph to ask the user a question."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

from juena.schema.agents import AskUserSchema
from juena.schema.sandbox import CLARIFICATION_KIND

MAX_OPTIONS = 4

ASK_USER_DESCRIPTION = """Ask the user one question and wait for their answer.

The user is the domain expert; you are not. Use this whenever a choice would
change the result and you cannot resolve it from the objective, the staged
inputs, or a tool: an ambiguous request, a missing parameter, or a scientific
judgment call where guessing would produce a plausible but wrong answer. Ask
before spending execution budget on a guess.

Do not ask for anything already stated in the objective or retrievable with a
tool, and do not ask the same question twice. Give 2-4 concrete `options` when
you can name the likely choices; the user can always answer freely instead.

Returns the user's answer as a string."""


def _payload(asked_by: str, question: str, options: list[str] | None) -> dict[str, object]:
    cleaned = [item.strip() for item in (options or []) if item and item.strip()]
    return {
        "kind": CLARIFICATION_KIND,
        "asked_by": asked_by,
        "question": question.strip(),
        "options": cleaned[:MAX_OPTIONS],
    }


def build_ask_user_tool(asked_by: str) -> BaseTool:
    """Build an `ask_user` tool labelled with the agent that owns it.

    `interrupt` raises out of the node and is resumed by id, so this works the
    same from a specialist subgraph as from the supervisor -- the sandbox
    approval already takes that path. The answer becomes the tool result, so the
    asking agent keeps everything it had already established.
    """

    def ask_user(question: str, options: list[str] | None = None) -> str:
        answer = interrupt(_payload(asked_by, question, options))
        return str(answer) if answer is not None else ""

    return StructuredTool.from_function(
        name="ask_user",
        func=ask_user,
        description=ASK_USER_DESCRIPTION,
        args_schema=AskUserSchema,
        infer_schema=False,
    )
