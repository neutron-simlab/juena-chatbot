"""Tests for UI helper behavior."""

from app.ui_components import should_collapse_tool_payload


def test_should_collapse_tool_payload_for_explicit_collapsed_mode() -> None:
    assert should_collapse_tool_payload(
        {"tool_kind": "regular_tool_result", "display_mode": "collapsed_by_default"}
    ) is True


def test_should_collapse_tool_payload_for_legacy_tool_message() -> None:
    assert should_collapse_tool_payload({}) is True
    assert should_collapse_tool_payload(None) is True


def test_should_collapse_tool_payload_respects_inline_override() -> None:
    assert should_collapse_tool_payload({"display_mode": "inline"}) is False
