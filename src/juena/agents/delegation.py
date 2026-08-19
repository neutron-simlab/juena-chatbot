"""The boundary a specialist is invoked across.

`SubAgentMiddleware` hands a subagent the parent's whole state minus its private
keys, and copies the subagent's whole state back the same way. That is a sensible
default for an agent framework and the wrong one here, for two reasons that only
show up in JüNA's shape:

*Going in.* The supervisor's `files` channel is not only findings. Summarization
parks evicted conversation turns under `/conversation_history/` through the same
channel, so a specialist -- which is told it cannot see the conversation -- was
being handed the part of it the supervisor had already thrown away.

*Coming back.* Everything the run left in `files` returns, including findings it
merely inherited on the way in. Re-writing an unchanged file is harmless until
something else edits it in the meantime.

Neither is fixed by asking the specialist not to look, and neither should depend
on remembering to re-check after the next upstream release adds a state field. So
each specialist is registered through this adapter, which names what may cross in
each direction. `SubAgentMiddleware` keeps its own semantics; the boundary is
decided here, before invocation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

from juena.agents.findings import crossing_files, findings_delta


class SpecialistDelegate(Runnable[dict[str, Any], dict[str, Any]]):
    """One specialist, invoked with an allowlist rather than a snapshot."""

    def __init__(self, runnable: Runnable, *, name: str) -> None:
        self._runnable = runnable
        self.name = name

    def _inbound(self, state: Any) -> dict[str, Any]:
        """Exactly the objective and the files the contract names.

        Built from scratch rather than filtered, so a state field added upstream
        is absent by default instead of crossing until someone notices.
        """
        values = state if isinstance(state, dict) else {}
        return {
            "messages": values.get("messages") or [],
            "files": crossing_files(values.get("files")),
        }

    def _outbound(self, sent: dict[str, Any], result: Any) -> dict[str, Any]:
        """The report, and the findings this run actually wrote."""
        values = result if isinstance(result, dict) else {}
        return {
            "messages": values.get("messages") or [],
            "files": findings_delta(sent["files"], values.get("files")),
        }

    def invoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sent = self._inbound(input)
        return self._outbound(sent, self._runnable.invoke(sent, config, **kwargs))

    async def ainvoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sent = self._inbound(input)
        return self._outbound(sent, await self._runnable.ainvoke(sent, config, **kwargs))


def with_delegation_boundary(specialists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap each compiled specialist for registration with `SubAgentMiddleware`."""

    return [
        {**spec, "runnable": SpecialistDelegate(spec["runnable"], name=spec["name"])}
        for spec in specialists
    ]
