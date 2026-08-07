"""Tests for code-chat staged input helpers."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from deepagents.backends.utils import file_data_to_string
from starlette.datastructures import Headers, UploadFile

from juena.server.chat import input_utils as code_chat_utils
from juena.server.chat import inputs as code_chat_inputs


def _upload(filename: str, content: bytes, content_type: str = "text/plain") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


@pytest.mark.asyncio
async def test_normalize_uploaded_attachments_stages_text_files_with_deduped_names() -> None:
    attachments, files_update = await code_chat_inputs.normalize_uploaded_attachments(
        [
            _upload("../../example.py", b"print('one')\n", "text/x-python"),
            _upload("example.py", b"print('two')\n", "text/x-python"),
        ]
    )

    assert [attachment.staged_path for attachment in attachments] == [
        "/inputs/uploads/example.py",
        "/inputs/uploads/example_2.py",
    ]
    assert sorted(files_update) == [
        "/inputs/uploads/example.py",
        "/inputs/uploads/example_2.py",
    ]


@pytest.mark.asyncio
async def test_normalize_uploaded_attachments_dedupes_against_existing_thread_uploads() -> None:
    attachments, _files_update = await code_chat_inputs.normalize_uploaded_attachments(
        [_upload("example.py", b"print('three')\n", "text/x-python")],
        existing_upload_paths=[
            "/inputs/uploads/example.py",
            "/inputs/uploads/example_2.py",
        ],
    )

    assert [attachment.staged_path for attachment in attachments] == [
        "/inputs/uploads/example_3.py",
    ]


@pytest.mark.asyncio
async def test_normalize_uploaded_attachments_rejects_unsupported_or_binary_files() -> None:
    with pytest.raises(ValueError, match="unsupported extension"):
        await code_chat_inputs.normalize_uploaded_attachments([_upload("notes.pdf", b"%PDF-1.7")])

    with pytest.raises(ValueError, match="valid UTF-8 text"):
        await code_chat_inputs.normalize_uploaded_attachments([_upload("bad.py", b"print('x')\x00")])


@pytest.mark.asyncio
async def test_prepare_code_chat_turn_inputs_preserves_thread_uploads_and_clears_scratch() -> None:
    class FakeAgent:
        async def aget_state(self, config):  # noqa: ANN001
            return SimpleNamespace(
                values={
                    "files": {
                        "/inputs/old.txt": {
                            "content": "old",
                            "encoding": "utf-8",
                            "created_at": "c",
                            "modified_at": "m",
                        },
                        "/inputs/uploads/kept.py": {
                            "content": "print('kept')",
                            "encoding": "utf-8",
                            "created_at": "c",
                            "modified_at": "m",
                        },
                        "/unrelated.txt": {
                            "content": "keep",
                            "encoding": "utf-8",
                            "created_at": "c",
                            "modified_at": "m",
                        },
                    }
                }
            )

    prepared = await code_chat_inputs.prepare_code_chat_turn_inputs(
        FakeAgent(),  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
        message=(
            "Why is this failing?\n\n"
            "```python\n"
            "def boom():\n"
            "    return missing_name\n"
            "```\n\n"
            "```\n"
            "Traceback (most recent call last):\n"
            "NameError: name 'missing_name' is not defined\n"
            "```\n"
        ),
        attachments=[_upload("extra.log", b"ERROR: something happened\n")],
    )

    assert prepared is not None
    assert prepared.files_update["/inputs/old.txt"] is None
    assert "/inputs/uploads/kept.py" not in prepared.files_update
    assert "/inputs/current_message.txt" in prepared.files_update
    assert "/inputs/current_code.py" in prepared.files_update
    assert "/inputs/current_error.txt" in prepared.files_update
    assert "/inputs/uploads/extra.log" in prepared.files_update
    assert "/inputs/uploads_manifest.md" in prepared.files_update
    manifest_text = file_data_to_string(
        prepared.files_update["/inputs/uploads_manifest.md"]
    )
    assert "/inputs/uploads/kept.py" in manifest_text
    assert "/inputs/uploads/extra.log" in manifest_text
    assert "Inspect `/inputs` before answering." in prepared.message_override
    assert "Persistent uploaded files for this chat are available under `/inputs/uploads/`." in prepared.message_override
    assert "User request: Why is this failing?" in prepared.message_override


@pytest.mark.asyncio
async def test_prepare_code_chat_turn_inputs_reminds_about_existing_thread_uploads_on_follow_up() -> None:
    class FakeAgent:
        async def aget_state(self, config):  # noqa: ANN001
            return SimpleNamespace(
                values={
                    "files": {
                        "/inputs/uploads/snippet.py": {
                            "content": "print('hello')",
                            "encoding": "utf-8",
                            "created_at": "c",
                            "modified_at": "m",
                        },
                        "/inputs/uploads_manifest.md": {
                            "content": (
                                "# Persistent chat uploads\n\n"
                                "1. `/inputs/uploads/snippet.py` (from snippet.py, 14 chars)"
                            ),
                            "encoding": "utf-8",
                            "created_at": "c",
                            "modified_at": "m",
                        },
                    }
                }
            )

    prepared = await code_chat_inputs.prepare_code_chat_turn_inputs(
        FakeAgent(),  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
        message="What does the uploaded file do?",
    )

    assert prepared is not None
    assert prepared.files_update == {}
    assert "/inputs/uploads_manifest.md" in prepared.message_override
    assert "Persistent uploaded files for this chat are available under `/inputs/uploads/`." in prepared.message_override
    assert "Current turn files:" not in prepared.message_override


@pytest.mark.parametrize(
    "message",
    [
        # Prose that used to be staged as a traceback because ERROR_LINE_RE ended
        # in a bare `Error:` alternative.
        "What does the Error: message in the manual mean?",
        # Prose that used to be staged as code because any line containing one of
        # `[{}();]` counted as code-like.
        "Can you (please) explain how the detector works?",
        "How do I normalise SANS data?",
        # An unfenced traceback is no longer staged either -- the accepted cost of
        # trusting only what the user marked deliberately. It still reaches the
        # agent verbatim in the message.
        'Traceback (most recent call last):\n  File "run.py", line 12\nValueError: bad',
    ],
)
def test_unfenced_text_is_never_staged(message: str) -> None:
    assert code_chat_utils.extract_pasted_code_context(message) is None


def test_fenced_code_is_staged_with_its_language() -> None:
    context = code_chat_utils.extract_pasted_code_context(
        "why does this fail?\n\n```python\ndef f(x):\n    return x / 0\n```"
    )

    assert context is not None
    assert context.contains_code and not context.contains_error
    assert code_chat_utils.staged_code_path(context) == "/inputs/current_code.py"
    assert context.user_goal == "why does this fail?"


def test_fenced_traceback_is_staged_as_an_error() -> None:
    context = code_chat_utils.extract_pasted_code_context(
        'help!\n\n```\nTraceback (most recent call last):\n'
        '  File "run.py", line 12\n    x = 1\nValueError: bad\n```'
    )

    assert context is not None
    assert context.contains_error and not context.contains_code
    assert context.user_goal == "help!"


def test_fenced_js_stack_trace_is_detected() -> None:
    """The `^`-anchored alternatives only matched at string start before
    re.MULTILINE was set, so JS/Java traces went unrecognised."""
    context = code_chat_utils.extract_pasted_code_context(
        "```\n  at foo (bar.js:10)\n  at baz (bar.js:22)\n```"
    )

    assert context is not None
    assert context.contains_error
