"""Tests for server message conversion helpers."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import ChatMessage as LangchainChatMessage

from juena_core.server.utils import langchain_to_chat_message


def test_tool_message_conversion_adds_default_tool_ui_metadata() -> None:
    message = ToolMessage(content='{"ok": true}', tool_call_id="call-1")

    chat_message = langchain_to_chat_message(message)

    assert chat_message.type == "tool"
    assert chat_message.tool_call_id == "call-1"
    assert chat_message.custom_data == {
        "tool_kind": "regular_tool_result",
        "display_mode": "collapsed_by_default",
    }


def test_message_id_is_preserved() -> None:
    """The client dedups streamed messages against checkpointed history by id,
    so dropping it here makes the two impossible to correlate."""
    assert langchain_to_chat_message(AIMessage(content="hi", id="run-abc")).id == "run-abc"
    assert langchain_to_chat_message(HumanMessage(content="hi", id="h-1")).id == "h-1"
    assert langchain_to_chat_message(AIMessage(content="hi")).id is None


def test_reasoning_blocks_are_surfaced_not_dropped_into_content() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "internal deliberation"},
            {"type": "text", "text": "The answer is 42."},
        ],
        response_metadata={"model_provider": "openai"},
    )

    chat_message = langchain_to_chat_message(message)

    assert chat_message.content == "The answer is 42."
    assert chat_message.custom_data["reasoning"] == "internal deliberation"


def test_untyped_content_blocks_do_not_raise() -> None:
    """The old hand-rolled flattener indexed content_item["type"] unguarded."""
    message = AIMessage(content=[{"text": "no type key"}, {"type": "text", "text": "ok"}])

    assert langchain_to_chat_message(message).content == "ok"


def test_custom_message_accepts_dict_and_rejects_str() -> None:
    payload = {"tool_kind": "custom_widget"}
    converted = langchain_to_chat_message(
        LangchainChatMessage(role="custom", content=[payload])
    )
    assert converted.type == "custom"
    assert converted.custom_data == payload

    # A str payload used to be indexed as content[0], yielding a single char.
    with pytest.raises(ValueError, match="must be a dict"):
        langchain_to_chat_message(LangchainChatMessage(role="custom", content="oops"))


def test_staged_human_message_replays_as_the_users_own_text() -> None:
    """A turn with uploads sends the `/inputs` manifest to the model, but the
    checkpointer is the only store of message content -- so without the
    preserved original, reloading the chat showed agent scaffolding inside the
    user's own bubble."""
    typed = "Why does my reduction script crash on the second run?"
    manifest = (
        "Persistent uploaded files for this chat are available under `/inputs/uploads/`.\n"
        f"\nUser request: {typed}\n\n- /inputs/uploads/reduce.py"
    )
    message = HumanMessage(
        content=manifest,
        additional_kwargs={
            "juena_display_text": typed,
            "juena_attachments": [{"name": "reduce.py", "chars": 812}],
        },
    )

    chat_message = langchain_to_chat_message(message)

    assert chat_message.content == typed
    assert chat_message.custom_data["attachments"] == [{"name": "reduce.py", "chars": 812}]


def test_plain_human_message_is_unaffected() -> None:
    chat_message = langchain_to_chat_message(HumanMessage(content="hello"))

    assert chat_message.content == "hello"
    assert "attachments" not in chat_message.custom_data
