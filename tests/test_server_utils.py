"""Tests for server message conversion helpers."""

from langchain_core.messages import ToolMessage

from juena.server.utils import langchain_to_chat_message


def test_tool_message_conversion_adds_default_tool_ui_metadata() -> None:
    message = ToolMessage(content='{"ok": true}', tool_call_id="call-1")

    chat_message = langchain_to_chat_message(message)

    assert chat_message.type == "tool"
    assert chat_message.tool_call_id == "call-1"
    assert chat_message.custom_data == {
        "tool_kind": "regular_tool_result",
        "display_mode": "collapsed_by_default",
    }
