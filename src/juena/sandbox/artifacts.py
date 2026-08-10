"""Validated PNG persistence and per-turn artifact event queues."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from PIL import Image

from juena.core.config import global_config
from juena.sandbox.constants import (
    DOWNLOAD_TYPES,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACTS_PER_EXECUTION,
    MAX_FILE_ARTIFACT_BYTES,
)
from juena.sandbox.schemas import ArtifactRef

MAX_ARTIFACTS_PER_TURN = 12
MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PIXELS = 16_000_000
ARTIFACT_MESSAGE_KEY = "juena_artifacts"

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


class ArtifactStore:
    """Filesystem-backed artifacts with in-process live-stream queues."""

    def __init__(self, root: Path, audit_file: Path) -> None:
        self.root = root
        self.audit_file = audit_file
        self._lock = threading.RLock()
        self._pending_message: dict[tuple[str, str], list[str]] = {}
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
            pending_ids = self._pending_message.get(key, [])
            if len(pending_ids) >= MAX_ARTIFACTS_PER_TURN:
                raise ValueError("This turn already has the maximum number of generated files")
            if kind == "image":
                pending_images = sum(
                    1
                    for item in pending_ids
                    if (existing := self._load(user_id, item)) is not None
                    and existing.kind == "image"
                )
                if pending_images >= MAX_ARTIFACTS_PER_EXECUTION:
                    raise ValueError("This turn already has the maximum number of plots")
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
            self._pending_message.setdefault(key, []).append(artifact_id)
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
            ids = self._pending_message.pop((user_id, thread_id), [])
            refs = [stored.public_ref() for item in ids if (stored := self._load(user_id, item))]
        return [ref.model_dump(mode="json") for ref in refs]

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
