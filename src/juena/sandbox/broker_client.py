"""Authenticated client for the separately deployed sandbox broker."""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
from typing import Iterable

import httpx

from juena.core.config import global_config
from juena.sandbox.constants import BROKER_AUTH_HEADER
from juena.sandbox.schemas import (
    BrokerFile,
    BrokerFileResult,
    DownloadFilesRequest,
    ExecuteRequest,
    ExecuteResult,
    StageInputsRequest,
    UploadFilesRequest,
)


class SandboxBrokerError(RuntimeError):
    """Raised when the isolated execution broker cannot complete a request."""


def workspace_id_for(user_id: str, thread_id: str, token: str) -> str:
    """Return an opaque tenant/thread key without exposing either identifier."""

    payload = f"{user_id}\0{thread_id}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), payload, hashlib.sha256).hexdigest()


class SandboxBrokerClient:
    def __init__(self, base_url: str, token: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={BROKER_AUTH_HEADER: token},
            timeout=httpx.Timeout(timeout),
        )

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxBrokerError("Sandbox broker request failed") from exc

    def execute(self, request: ExecuteRequest) -> ExecuteResult:
        return ExecuteResult.model_validate(
            self._post("/v1/execute", request.model_dump(mode="json"))
        )

    def assert_healthy(self) -> None:
        """Fail startup when the configured isolated broker is unreachable."""
        try:
            response = self._client.get("/health")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxBrokerError("Sandbox broker health check failed") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise SandboxBrokerError("Sandbox broker health check returned an invalid response")

    def upload_files(
        self, workspace_id: str, files: Iterable[tuple[str, bytes]]
    ) -> list[BrokerFileResult]:
        request = UploadFilesRequest(
            workspace_id=workspace_id,
            files=[
                BrokerFile(path=path, content_base64=base64.b64encode(content).decode("ascii"))
                for path, content in files
            ],
        )
        payload = self._post("/v1/files/upload", request.model_dump(mode="json"))
        return [BrokerFileResult.model_validate(item) for item in payload.get("files", [])]

    def download_files(self, workspace_id: str, paths: list[str]) -> list[BrokerFileResult]:
        request = DownloadFilesRequest(workspace_id=workspace_id, paths=paths)
        payload = self._post("/v1/files/download", request.model_dump(mode="json"))
        return [BrokerFileResult.model_validate(item) for item in payload.get("files", [])]

    async def stage_inputs(self, workspace_id: str, files: list[tuple[str, bytes]]) -> None:
        request = StageInputsRequest(
            workspace_id=workspace_id,
            files=[
                BrokerFile(path=path, content_base64=base64.b64encode(content).decode("ascii"))
                for path, content in files
            ],
        )
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={BROKER_AUTH_HEADER: self.token},
                timeout=httpx.Timeout(self.timeout),
            ) as client:
                response = await client.post(
                    "/v1/inputs/stage", json=request.model_dump(mode="json")
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SandboxBrokerError("Unable to stage sandbox inputs") from exc

    async def delete_workspace(self, workspace_id: str) -> None:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={BROKER_AUTH_HEADER: self.token},
                timeout=httpx.Timeout(self.timeout),
            ) as client:
                response = await client.delete(f"/v1/workspaces/{workspace_id}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SandboxBrokerError("Unable to delete sandbox workspace") from exc

    def close(self) -> None:
        self._client.close()


_client: SandboxBrokerClient | None = None
_client_lock = threading.Lock()


def get_broker_client() -> SandboxBrokerClient:
    global _client
    if not global_config.SANDBOX_ENABLED:
        raise SandboxBrokerError("Sandbox execution is disabled")
    if not global_config.SANDBOX_BROKER_URL or not global_config.SANDBOX_BROKER_TOKEN:
        raise SandboxBrokerError("Sandbox broker is not configured")
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = SandboxBrokerClient(
                    global_config.SANDBOX_BROKER_URL,
                    global_config.SANDBOX_BROKER_TOKEN,
                    global_config.SANDBOX_BROKER_TIMEOUT_SECONDS,
                )
    return _client


def set_broker_client_for_tests(client: SandboxBrokerClient | None) -> None:
    global _client
    _client = client


def close_broker_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
