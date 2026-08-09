"""Tests for UI helper behavior."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from juena.schema.server import ChatMessage

import app.ui_components as ui_components


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), "white").save(buffer, format="PNG")
    return buffer.getvalue()


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


def test_sanitize_assistant_content_removes_complete_inline_data_image() -> None:
    content = "Plot:\n\n![Sine wave](data:image/png;base64,aGVsbG8=)\n\nDone."

    assert ui_components.sanitize_assistant_content(content) == "Plot:\n\nDone."


def test_sanitize_assistant_content_removes_truncated_inline_data_image() -> None:
    content = "Plot:\n\n![Sine wave](data:image/png;base64,aGVsbG8=\n\nDone."

    assert ui_components.sanitize_assistant_content(content) == "Plot:\n\nDone."


def test_render_streaming_token_hides_inline_data_image() -> None:
    placeholder = Mock()

    ui_components.render_streaming_token(
        "Plot:\n\n![Sine wave](data:image/png;base64,aGVsbG8=)",
        placeholder,
    )

    placeholder.markdown.assert_called_once_with("Plot:\n\n▌")


def test_render_artifacts_displays_plot_and_downloads(monkeypatch) -> None:
    html_mock = Mock()
    download_mock = Mock()
    expand_mock = Mock(return_value=False)
    fake_st = SimpleNamespace(
        session_state=SimpleNamespace(client=Mock()),
        html=html_mock,
        button=expand_mock,
        container=lambda **_kwargs: _DummyContext(),
        download_button=download_mock,
        caption=Mock(),
    )
    monkeypatch.setattr(ui_components, "st", fake_st)
    png = _png()

    ui_components.render_artifacts(
        {
            "artifacts": [
                {
                    "artifact_id": "plot-1",
                    "filename": "plot.png",
                    "caption": "Sine wave",
                    "kind": "image",
                    "mime_type": "image/png",
                    "content_base64": base64.b64encode(png).decode("ascii"),
                },
                {
                    "artifact_id": "script-1",
                    "filename": "analysis.py",
                    "caption": "Analysis script",
                    "kind": "file",
                    "mime_type": "text/x-python",
                    "content_base64": base64.b64encode(b"print('ok')\n").decode("ascii"),
                },
            ]
        }
    )

    rendered_plot = html_mock.call_args.args[0]
    assert "data:image/png;base64," in rendered_plot
    assert f"max-width:{ui_components.PLOT_PREVIEW_WIDTH_PX}px" in rendered_plot
    assert "Sine wave" in rendered_plot
    assert expand_mock.call_args.kwargs["icon"] == ":material/open_in_full:"
    assert [call.args[0] for call in download_mock.call_args_list] == [
        "Download plot.png",
        "Download analysis.py",
    ]


def test_render_artifact_expand_opens_plot_dialog(monkeypatch) -> None:
    png = _png()
    dialog_mock = Mock()
    monkeypatch.setattr(ui_components, "_render_plot_dialog", dialog_mock)
    monkeypatch.setattr(
        ui_components,
        "st",
        SimpleNamespace(
            session_state=SimpleNamespace(client=Mock()),
            html=Mock(),
            button=Mock(return_value=True),
            container=lambda **_kwargs: _DummyContext(),
            download_button=Mock(),
            caption=Mock(),
        ),
    )

    ui_components.render_artifacts(
        {
            "artifacts": [
                {
                    "artifact_id": "plot-1",
                    "filename": "plot.png",
                    "caption": "Sine wave",
                    "kind": "image",
                    "mime_type": "image/png",
                    "content_base64": base64.b64encode(png).decode("ascii"),
                }
            ]
        }
    )

    dialog_mock.assert_called_once_with(png, "Sine wave")


def test_render_png_escapes_caption_html(monkeypatch) -> None:
    html_mock = Mock()
    monkeypatch.setattr(ui_components, "st", SimpleNamespace(html=html_mock))

    ui_components._render_png(_png(), '<script>alert("x")</script>', max_width="720px")

    rendered = html_mock.call_args.args[0]
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_render_artifact_history_fetches_through_authenticated_client(monkeypatch) -> None:
    client = Mock()
    client.get_artifact.return_value = b"result\n"
    download_mock = Mock()
    monkeypatch.setattr(
        ui_components,
        "st",
        SimpleNamespace(
            session_state=SimpleNamespace(client=client),
            image=Mock(),
            download_button=download_mock,
            caption=Mock(),
        ),
    )

    ui_components.render_artifacts(
        {
            "artifacts": [
                {
                    "artifact_id": "file-1",
                    "filename": "result.txt",
                    "kind": "file",
                    "mime_type": "text/plain",
                }
            ]
        }
    )

    client.get_artifact.assert_called_once_with("file-1")
    assert download_mock.call_args.kwargs["data"] == b"result\n"
