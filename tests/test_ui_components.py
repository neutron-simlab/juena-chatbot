"""Tests for UI helper behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

from juena.schema.server import ChatMessage

import app.ui_components as ui_components


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_should_collapse_tool_payload_for_explicit_collapsed_mode() -> None:
    assert ui_components.should_collapse_tool_payload(
        {"tool_kind": "regular_tool_result", "display_mode": "collapsed_by_default"}
    ) is True


def test_should_collapse_tool_payload_for_legacy_tool_message() -> None:
    assert ui_components.should_collapse_tool_payload({}) is True
    assert ui_components.should_collapse_tool_payload(None) is True


def test_should_collapse_tool_payload_respects_inline_override() -> None:
    assert ui_components.should_collapse_tool_payload({"display_mode": "inline"}) is False


def test_render_message_human_without_math_uses_write(monkeypatch) -> None:
    write_mock = Mock()
    markdown_mock = Mock()
    fake_st = SimpleNamespace(
        chat_message=lambda _role: _DummyContext(),
        write=write_mock,
        markdown=markdown_mock,
        json=Mock(),
    )
    monkeypatch.setattr(ui_components, "st", fake_st)

    ui_components.render_message(ChatMessage(type="human", content="plain text"))

    write_mock.assert_called_once_with("plain text")
    markdown_mock.assert_not_called()


def test_render_message_human_with_math_uses_markdown(monkeypatch) -> None:
    write_mock = Mock()
    markdown_mock = Mock()
    fake_st = SimpleNamespace(
        chat_message=lambda _role: _DummyContext(),
        write=write_mock,
        markdown=markdown_mock,
        json=Mock(),
    )
    monkeypatch.setattr(ui_components, "st", fake_st)

    ui_components.render_message(ChatMessage(type="human", content=r"Equation: \(a+b\)"))

    write_mock.assert_not_called()
    markdown_mock.assert_called_once_with("Equation: $a+b$")


def test_render_message_human_with_broken_latex_still_uses_markdown(monkeypatch) -> None:
    write_mock = Mock()
    markdown_mock = Mock()
    fake_st = SimpleNamespace(
        chat_message=lambda _role: _DummyContext(),
        write=write_mock,
        markdown=markdown_mock,
        json=Mock(),
    )
    monkeypatch.setattr(ui_components, "st", fake_st)

    ui_components.render_message(
        ChatMessage(type="human", content=r"1. \frac{4\pi R^{3}}{3},\frac{j_{1}(qR)}{qR}$")
    )

    write_mock.assert_not_called()
    markdown_mock.assert_called_once_with(r"1. $\frac{4\pi R^{3}}{3},\frac{j_{1}(qR)}{qR}$")


def test_render_message_human_code_only_math_stays_literal(monkeypatch) -> None:
    write_mock = Mock()
    markdown_mock = Mock()
    fake_st = SimpleNamespace(
        chat_message=lambda _role: _DummyContext(),
        write=write_mock,
        markdown=markdown_mock,
        json=Mock(),
    )
    monkeypatch.setattr(ui_components, "st", fake_st)

    ui_components.render_message(ChatMessage(type="human", content=r"`\(a+b\)`"))

    write_mock.assert_called_once_with(r"`\(a+b\)`")
    markdown_mock.assert_not_called()
