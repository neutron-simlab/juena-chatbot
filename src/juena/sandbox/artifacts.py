"""Validated PNG persistence and per-turn artifact event queues."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from PIL import Image

from juena.core.config import global_config
from juena.sandbox.constants import (
    DOWNLOAD_TYPES,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_BYTES_PER_TURN,
    MAX_FILE_ARTIFACT_BYTES,
    MAX_IMAGE_ARTIFACTS_PER_TURN,
    MAX_RECORD_ARTIFACTS_PER_TURN,
)
from juena.sandbox.schemas import ArtifactRef

MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PIXELS = 16_000_000
ARTIFACT_MESSAGE_KEY = "juena_artifacts"
MAX_UNDELIVERED_NOTICES_PER_TURN = 8

# What a file is for, which decides whose budget it spends. A "record" is the
# application's own transcript of an approved command; a "result" is something
# the user asked for. Mixing the two lets a long debugging session crowd out the
# figure that was the point of the turn.
ArtifactCategory = Literal["result", "record"]

_TEXT_EXTENSIONS = frozenset(
    {".csv", ".ipynb", ".json", ".log", ".md", ".py", ".sh", ".tsv", ".txt"}
)


@dataclass(slots=True)
class _StoredArtifact:
    artifact_id: str
    user_id: str
    thread_id: str
    run_id: str | None
    filename: str
    mime_type: str
    kind: Literal["image", "file"]
    size: int
    width: int | None
    height: int | None
    caption: str
    created_at: str

    def public_ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.artifact_id,
            filename=self.filename,
            mime_type=self.mime_type,
            kind=self.kind,
            size=self.size,
            width=self.width,
            height=self.height,
            caption=self.caption,
            created_at=datetime.fromisoformat(self.created_at),
        )


def undelivered_note(dropped: list[tuple[str, str]]) -> str:
    """The block telling the model a file exists but the user cannot see it.

    Shared so a file refused during a command and one refused during a write
    are described to the model in the same words.
    """
    lines = "\n".join(f"- {name}: {reason}" for name, reason in dropped)
    return (
        "Files written to /workspace/outputs but NOT delivered to the user:\n"
        + lines
        + "\nThe user cannot see these files. No file path, URL, or sandbox: link "
        "will show them. Do not write a Markdown image or a path for them. Say in "
        "your report that they could not be delivered, give the reason above, and "
        "offer to produce them again in a new message."
    )


@dataclass(slots=True)
class _PendingTurn:
    """One turn's artifact accounting, held only in memory.

    The budget is a cost model rather than a file count: an image is re-encoded
    into the SSE stream and painted inline, while a downloadable file costs one
    button. Nothing here is persisted -- the caps constrain what a single answer
    may carry, and a restart legitimately starts that over.
    """

    ids: list[str] = field(default_factory=list)
    images: int = 0
    result_bytes: int = 0
    records: int = 0
    # Files written straight into /workspace/outputs that could not be
    # registered. `write_file` has nowhere to report this, so it waits here for
    # the next approved execution or the specialist's final report.
    undelivered: list[tuple[str, str]] = field(default_factory=list)
    undelivered_omitted: int = 0

    def admit(self, category: ArtifactCategory, kind: str, size: int) -> None:
        """Raise when the turn has no room for this file. Never mutates."""
        if category == "record":
            if self.records >= MAX_RECORD_ARTIFACTS_PER_TURN:
                raise ValueError("This turn already has the maximum number of execution records")
            return
        if kind == "image" and self.images >= MAX_IMAGE_ARTIFACTS_PER_TURN:
            raise ValueError(
                f"This turn already has the maximum of {MAX_IMAGE_ARTIFACTS_PER_TURN} plots"
            )
        if self.result_bytes + size > MAX_ARTIFACT_BYTES_PER_TURN:
            budget_mb = MAX_ARTIFACT_BYTES_PER_TURN // 1024**2
            raise ValueError(f"This turn has reached its {budget_mb} MB file budget")

    def add(self, artifact_id: str, category: ArtifactCategory, kind: str, size: int) -> None:
        self.ids.append(artifact_id)
        if category == "record":
            self.records += 1
            return
        self.result_bytes += size
        if kind == "image":
            self.images += 1


class ArtifactStore:
    """Filesystem-backed artifacts with in-process live-stream queues."""

    def __init__(self, root: Path, audit_file: Path) -> None:
        self.root = root
        self.audit_file = audit_file
        self._lock = threading.RLock()
        self._pending_message: dict[tuple[str, str], _PendingTurn] = {}
        self._pending_events: dict[tuple[str, str], list[str]] = {}

    @staticmethod
    def _user_key(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    def _user_root(self, user_id: str) -> Path:
        return self.root / self._user_key(user_id)

    @staticmethod
    def _safe_filename(filename: str) -> str:
        leaf = Path(filename).name.strip()
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
        return (cleaned or "result.txt")[:240]

    @staticmethod
    def _validate_png(content: bytes) -> tuple[int, int]:
        if not content or len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("Generated PNG is empty or exceeds the 5 MB limit")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format != "PNG":
                    raise ValueError("Generated artifact is not a PNG")
                width, height = image.size
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError("Generated artifact is not a valid PNG") from exc
        if (
            width <= 0
            or height <= 0
            or width > MAX_IMAGE_EDGE
            or height > MAX_IMAGE_EDGE
            or width * height > MAX_IMAGE_PIXELS
        ):
            raise ValueError("Generated PNG dimensions exceed the configured limit")
        return width, height

    @classmethod
    def _validate_artifact(
        cls,
        filename: str,
        content: bytes,
    ) -> tuple[str, str, int | None, int | None]:
        extension = Path(filename).suffix.lower()
        if extension == ".png":
            width, height = cls._validate_png(content)
            return "image/png", "image", width, height
        mime_type = DOWNLOAD_TYPES.get(extension)
        if mime_type is None:
            raise ValueError("Generated file type is not allowed")
        if not content or len(content) > MAX_FILE_ARTIFACT_BYTES:
            raise ValueError("Generated file is empty or exceeds the 10 MB limit")
        if extension in _TEXT_EXTENSIONS:
            try:
                decoded = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Generated text file is not valid UTF-8") from exc
            if "\x00" in decoded:
                raise ValueError("Generated text file contains binary data")
        elif extension == ".pdf" and not content.startswith(b"%PDF-"):
            raise ValueError("Generated PDF has an invalid header")
        elif extension == ".npy" and not content.startswith(b"\x93NUMPY"):
            raise ValueError("Generated NumPy file has an invalid header")
        elif extension == ".npz" and not content.startswith(b"PK\x03\x04"):
            raise ValueError("Generated NumPy archive has an invalid header")
        return mime_type, "file", None, None

    def _metadata_path(self, user_id: str, artifact_id: str) -> Path:
        return self._user_root(user_id) / f"{artifact_id}.json"

    def _content_path(self, user_id: str, artifact_id: str) -> Path:
        return self._user_root(user_id) / f"{artifact_id}.bin"

    def register_artifact(
        self,
        *,
        user_id: str,
        thread_id: str,
        run_id: str | None,
        filename: str,
        content: bytes,
        caption: str | None = None,
        category: ArtifactCategory = "result",
    ) -> ArtifactRef:
        artifact_id = str(uuid4())
        safe_filename = self._safe_filename(filename)
        mime_type, kind, width, height = self._validate_artifact(safe_filename, content)
        created_at = datetime.now(timezone.utc).isoformat()
        stored = _StoredArtifact(
            artifact_id=artifact_id,
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
            filename=safe_filename,
            mime_type=mime_type,
            kind=kind,
            size=len(content),
            width=width,
            height=height,
            caption=(caption or Path(safe_filename).stem.replace("_", " ")).strip(),
            created_at=created_at,
        )

        with self._lock:
            key = (user_id, thread_id)
            pending = self._pending_message.setdefault(key, _PendingTurn())
            pending.admit(category, kind, len(content))
            user_root = self._user_root(user_id)
            user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            content_path = self._content_path(user_id, artifact_id)
            metadata_path = self._metadata_path(user_id, artifact_id)
            content_tmp = content_path.with_suffix(".bin.tmp")
            metadata_tmp = metadata_path.with_suffix(".json.tmp")
            content_tmp.write_bytes(content)
            metadata_tmp.write_text(json.dumps(asdict(stored), sort_keys=True), encoding="utf-8")
            os.replace(content_tmp, content_path)
            os.replace(metadata_tmp, metadata_path)
            pending.add(artifact_id, category, kind, len(content))
            self._pending_events.setdefault(key, []).append(artifact_id)
        return stored.public_ref()

    def _load(self, user_id: str, artifact_id: str) -> _StoredArtifact | None:
        path = self._metadata_path(user_id, artifact_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stored = _StoredArtifact(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if stored.user_id != user_id or stored.artifact_id != artifact_id:
            return None
        return stored

    def get(self, user_id: str, artifact_id: str) -> tuple[ArtifactRef, bytes] | None:
        with self._lock:
            stored = self._load(user_id, artifact_id)
            if stored is None:
                return None
            try:
                content = self._content_path(user_id, artifact_id).read_bytes()
            except OSError:
                return None
        return stored.public_ref(), content

    def claim_for_message(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            pending = self._pending_message.pop((user_id, thread_id), None)
            ids = pending.ids if pending is not None else []
            refs = [stored.public_ref() for item in ids if (stored := self._load(user_id, item))]
        return [ref.model_dump(mode="json") for ref in refs]

    def note_undelivered(
        self,
        user_id: str,
        thread_id: str,
        filename: str,
        reason: str,
    ) -> None:
        """Queue a file that reached the workspace but will not reach the user."""
        with self._lock:
            key = (user_id, thread_id)
            pending = self._pending_message.setdefault(key, _PendingTurn())
            if len(pending.undelivered) < MAX_UNDELIVERED_NOTICES_PER_TURN:
                pending.undelivered.append((filename, reason))
            else:
                pending.undelivered_omitted += 1

    def drain_undelivered(self, user_id: str, thread_id: str) -> list[tuple[str, str]]:
        """Take queued notices for an execution result or final specialist report."""
        with self._lock:
            pending = self._pending_message.get((user_id, thread_id))
            if pending is None or not pending.undelivered:
                return []
            queued = list(pending.undelivered)
            if pending.undelivered_omitted:
                count = pending.undelivered_omitted
                noun = "file" if count == 1 else "files"
                queued.append(
                    (
                        f"{count} additional output {noun}",
                        "Details omitted because this report is bounded",
                    )
                )
            pending.undelivered.clear()
            pending.undelivered_omitted = 0
            return queued

    def begin_turn(self, user_id: str, thread_id: str) -> None:
        """Discard a window left behind by an abandoned turn.

        The window is normally drained when the supervisor writes its final
        message. A cancelled stream or a crashed run never gets there, and the
        leftovers would both eat the next turn's budget and attach themselves to
        the next turn's answer.
        """
        with self._lock:
            self._pending_message.pop((user_id, thread_id), None)
            self._pending_events.pop((user_id, thread_id), None)

    def drain_events(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            ids = self._pending_events.pop((user_id, thread_id), [])
            records = [self.get(user_id, item) for item in ids]
        events: list[dict[str, Any]] = []
        for record in records:
            if record is None:
                continue
            ref, content = record
            event = {"type": "artifact", **ref.model_dump(mode="json")}
            if ref.kind == "image":
                event["content_base64"] = base64.b64encode(content).decode("ascii")
            events.append(event)
        return events

    def delete_thread(self, user_id: str, thread_id: str) -> None:
        with self._lock:
            user_root = self._user_root(user_id)
            for metadata_path in user_root.glob("*.json") if user_root.is_dir() else ():
                stored = self._load(user_id, metadata_path.stem)
                if stored is None or stored.thread_id != thread_id:
                    continue
                self._content_path(user_id, stored.artifact_id).unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            self._pending_message.pop((user_id, thread_id), None)
            self._pending_events.pop((user_id, thread_id), None)

    def audit(self, record: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self._lock:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self.audit_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


_artifact_store: ArtifactStore | None = None
_artifact_store_lock = threading.Lock()


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        with _artifact_store_lock:
            if _artifact_store is None:
                _artifact_store = ArtifactStore(
                    Path(global_config.SANDBOX_ARTIFACT_ROOT),
                    Path(global_config.SANDBOX_AUDIT_FILE),
                )
    return _artifact_store


def set_artifact_store_for_tests(store: ArtifactStore | None) -> None:
    global _artifact_store
    _artifact_store = store
