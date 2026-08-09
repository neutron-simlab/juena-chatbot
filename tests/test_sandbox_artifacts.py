"""Artifact validation, ownership, persistence, and download metadata."""

import io
from uuid import uuid4

import pytest
from fastapi import HTTPException
from PIL import Image

from juena.server.api.endpoints import get_artifact
from juena.server.auth import AuthenticatedUser
from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")


def test_stores_inline_png_and_downloadable_script_for_owner(tmp_path) -> None:
    store = _store(tmp_path)
    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="sine plot.png",
        content=_png(),
    )
    script = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="analysis.py",
        content=b"print('done')\n",
    )

    assert (plot.kind, plot.mime_type, plot.width, plot.height) == (
        "image",
        "image/png",
        12,
        8,
    )
    assert (script.kind, script.mime_type) == ("file", "text/x-python")
    assert store.get("user-a", script.artifact_id)[1] == b"print('done')\n"
    assert store.get("user-b", script.artifact_id) is None

    refs = store.claim_for_message("user-a", "thread-a")
    assert [item["artifact_id"] for item in refs] == [plot.artifact_id, script.artifact_id]
    assert store.claim_for_message("user-a", "thread-a") == []


@pytest.mark.parametrize("filename", ["result.html", "plot.svg", "program.exe", "data.zip"])
def test_rejects_active_or_unapproved_download_types(tmp_path, filename: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _store(tmp_path).register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename=filename,
            content=b"not trusted",
        )


def test_only_inline_plot_event_contains_bytes(tmp_path) -> None:
    store = _store(tmp_path)
    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="plot.png",
        content=_png(),
    )
    script = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="analysis.py",
        content=b"print('done')\n",
    )

    plot_event, script_event = store.drain_events("user-a", "thread-a")
    assert plot_event["artifact_id"] == plot.artifact_id
    assert plot_event["content_base64"]
    assert script_event["artifact_id"] == script.artifact_id
    assert "content_base64" not in script_event
    persisted = store.claim_for_message("user-a", "thread-a")
    assert all("content_base64" not in item for item in persisted)


def test_approved_command_marker_is_exact_and_single_use(tmp_path) -> None:
    store = _store(tmp_path)
    store.mark_approved_command("user-a", "thread-a", "interrupt-1", "python plot.py")

    assert store.consume_approved_command("user-a", "thread-a", "python other.py") is None
    assert (
        store.consume_approved_command("user-a", "thread-a", "python plot.py")
        == "interrupt-1"
    )
    assert store.consume_approved_command("user-a", "thread-a", "python plot.py") is None


def test_thread_deletion_removes_files(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="result.txt",
        content=b"result\n",
    )

    store.delete_thread("user-a", "thread-a")

    assert store.get("user-a", ref.artifact_id) is None


@pytest.mark.asyncio
async def test_artifact_endpoint_enforces_authenticated_owner(tmp_path) -> None:
    owner_id = uuid4()
    other_id = uuid4()
    store = _store(tmp_path)
    set_artifact_store_for_tests(store)
    ref = store.register_artifact(
        user_id=str(owner_id),
        thread_id="thread-a",
        run_id=None,
        filename="analysis.py",
        content=b"print('ok')\n",
    )
    owner = AuthenticatedUser(owner_id, "owner", None, None)
    other = AuthenticatedUser(other_id, "other", None, None)

    try:
        response = await get_artifact(ref.artifact_id, user=owner)
        with pytest.raises(HTTPException) as exc_info:
            await get_artifact(ref.artifact_id, user=other)
    finally:
        set_artifact_store_for_tests(None)

    assert response.body == b"print('ok')\n"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert exc_info.value.status_code == 404
