"""Asking the user is a tool: it pauses the graph and resumes with their answer."""

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from juena_core.agents.ask_user import CLARIFICATION_KIND, _payload, build_ask_user_tool


class _ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_the_payload_names_the_asker_and_bounds_the_options() -> None:
    value = _payload(
        "software-specialist",
        "  Which background subtraction?  ",
        ["Solvent-only", "  ", "Empty cell", "None", "A fourth", "A fifth"],
    )

    assert value == {
        "kind": CLARIFICATION_KIND,
        "asked_by": "software-specialist",
        "question": "Which background subtraction?",
        # Blanks dropped, then capped so the card stays a card.
        "options": ["Solvent-only", "Empty cell", "None", "A fourth"],
    }


def test_a_blank_question_is_rejected_before_it_reaches_the_user() -> None:
    tool = build_ask_user_tool("juena")

    with pytest.raises(ValueError):
        tool.args_schema(question="   ")


def test_the_answer_comes_back_as_the_tool_result() -> None:
    agent = create_agent(
        model=_ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "Which q-range?", "options": ["Low", "High"]},
                            "id": "ask-1",
                        }
                    ],
                ),
                AIMessage(content="Using the low-q range as you asked."),
            ]
        ),
        tools=[build_ask_user_tool("software-specialist")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "ask-thread"}}

    paused = agent.invoke({"messages": [("user", "Fit this data.")]}, config=config)
    interrupts = paused["__interrupt__"]
    assert interrupts[0].value["question"] == "Which q-range?"

    resumed = agent.invoke(
        Command(resume={interrupts[0].id: "Low, 0.01 to 0.1 per angstrom"}),
        config=config,
    )

    answer = next(
        message
        for message in resumed["messages"]
        if getattr(message, "tool_call_id", None) == "ask-1"
    )
    assert answer.text == "Low, 0.01 to 0.1 per angstrom"
    assert resumed["messages"][-1].text == "Using the low-q range as you asked."
