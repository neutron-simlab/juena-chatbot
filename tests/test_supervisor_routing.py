"""Scripted supervisor calls verify task routing and report synthesis plumbing."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from juena.agents.juena_agent import SUPERVISOR_TASK_DESCRIPTION
from juena.agents.specialists.neutron_research.agent import (
    NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
    NEUTRON_RESEARCH_SPECIALIST_NAME,
)
from juena.agents.specialists.software.agent import (
    SOFTWARE_SPECIALIST_DESCRIPTION,
    SOFTWARE_SPECIALIST_NAME,
)


class _ToolModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> "_ToolModel":
        return self


def _report(finding: str) -> str:
    return (
        f"## Finding\n{finding}\n"
        "## Evidence\nScripted evidence\n"
        "## Actions\nNone\n"
        "## Artifacts\nNone\n"
        "## Limitations\nScripted test report\n"
    )


def _build_supervisor(
    responses: list[AIMessage],
    captured: dict[str, list[str]],
) -> Any:
    def specialist(name: str, finding: str) -> dict[str, Any]:
        def run(state: dict[str, Any]) -> dict[str, list[AIMessage]]:
            request = next(message for message in state["messages"] if isinstance(message, HumanMessage))
            captured.setdefault(name, []).append(request.text)
            return {"messages": [AIMessage(content=_report(finding))]}

        description = (
            SOFTWARE_SPECIALIST_DESCRIPTION
            if name == SOFTWARE_SPECIALIST_NAME
            else NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION
        )
        return {
            "name": name,
            "description": description,
            "runnable": RunnableLambda(run),
        }

    backend = StateBackend()
    return create_agent(
        model=_ToolModel(responses=responses),
        tools=[],
        system_prompt="Delegate substantive domain work and synthesize the reports.",
        middleware=[
            SubAgentMiddleware(
                backend=backend,
                subagents=[
                    specialist(SOFTWARE_SPECIALIST_NAME, "Repository behavior confirmed."),
                    specialist(
                        NEUTRON_RESEARCH_SPECIALIST_NAME,
                        "Neutron-science explanation confirmed.",
                    ),
                ],
                system_prompt=None,
                task_description=SUPERVISOR_TASK_DESCRIPTION,
            )
        ],
        checkpointer=InMemorySaver(),
    )


@pytest.mark.parametrize(
    ("user_request", "target", "task_description", "final_answer"),
    [
        (
            "Where is repository indexing configured?",
            SOFTWARE_SPECIALIST_NAME,
            "Inspect repository indexing configuration and cite paths.",
            "Repository indexing is configured in the reported modules.",
        ),
        (
            "Explain neutron reflectometry.",
            NEUTRON_RESEARCH_SPECIALIST_NAME,
            "Explain neutron reflectometry and identify the main limitations.",
            "Neutron reflectometry measures depth-dependent interface structure.",
        ),
    ],
)
def test_supervisor_uses_one_matching_specialist_by_default(
    user_request: str,
    target: str,
    task_description: str,
    final_answer: str,
) -> None:
    captured: dict[str, list[str]] = {}
    supervisor = _build_supervisor(
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "task",
                    "args": {
                        "description": task_description,
                        "subagent_type": target,
                    },
                    "id": "delegate-one",
                }],
            ),
            AIMessage(content=final_answer),
        ],
        captured,
    )

    result = supervisor.invoke(
        {"messages": [("user", user_request)]},
        config={"configurable": {"thread_id": f"single-{target}"}},
    )

    assert captured == {target: [task_description]}
    assert result["messages"][-1].text == final_answer
    report_message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "## Finding" in report_message.text


def test_supervisor_fans_out_only_for_independent_mixed_request() -> None:
    captured: dict[str, list[str]] = {}
    supervisor = _build_supervisor(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Inspect the repository API and cite exact paths.",
                            "subagent_type": SOFTWARE_SPECIALIST_NAME,
                        },
                        "id": "delegate-software",
                    },
                    {
                        "name": "task",
                        "args": {
                            "description": "Explain the independent neutron-science concept.",
                            "subagent_type": NEUTRON_RESEARCH_SPECIALIST_NAME,
                        },
                        "id": "delegate-neutron",
                    },
                ],
            ),
            AIMessage(content="The repository and neutron findings are reconciled here."),
        ],
        captured,
    )

    result = supervisor.invoke(
        {"messages": [("user", "Review the API and separately explain neutron spin echo.")]},
        config={"configurable": {"thread_id": "mixed-request"}},
    )

    assert set(captured) == {
        SOFTWARE_SPECIALIST_NAME,
        NEUTRON_RESEARCH_SPECIALIST_NAME,
    }
    assert result["messages"][-1].text == (
        "The repository and neutron findings are reconciled here."
    )
    assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == 2


def test_delegated_task_contains_only_relevant_memory_preference() -> None:
    captured: dict[str, list[str]] = {}
    delegated = (
        "Inspect the repository. Relevant saved preference: prefer Python examples. "
        "Return exact paths and limitations."
    )
    supervisor = _build_supervisor(
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "task",
                    "args": {
                        "description": delegated,
                        "subagent_type": SOFTWARE_SPECIALIST_NAME,
                    },
                    "id": "delegate-with-preference",
                }],
            ),
            AIMessage(content="The answer uses Python examples."),
        ],
        captured,
    )

    supervisor.invoke(
        {
            "messages": [(
                "user",
                "Inspect the repository. Saved notes also contain an unrelated dark-mode preference.",
            )]
        },
        config={"configurable": {"thread_id": "memory-filter"}},
    )

    task = captured[SOFTWARE_SPECIALIST_NAME][0]
    assert "prefer Python examples" in task
    assert "dark-mode" not in task
    assert "complete memory" not in task
