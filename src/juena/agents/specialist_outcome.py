"""Compose the research package a specialist hands back to the supervisor."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage

from juena.agents.specialist_contracts import ResultArtifactEvidence, SpecialistReport
from juena.sandbox.artifacts import get_artifact_store
from juena.sandbox.evidence import SandboxExecutionEvidence
from juena.sandbox.runtime import _context_value

REPORT_OPEN = "<specialist_report>"
REPORT_CLOSE = "</specialist_report>"
VERIFIED_OPEN = "<verified_by_server>"
VERIFIED_CLOSE = "</verified_by_server>"

NO_REPORT = (
    "The specialist returned no structured report. Its run ended early -- most "
    "often because the model-call budget was exhausted -- so nothing it said "
    "mid-run can be treated as a result."
)
NO_SUCCESSFUL_RUN = (
    "Every sandbox command in this run failed. No calculation, fit, or plot was "
    "produced."
)
UNVERIFIED_DIRECTIVE = (
    "Tell the user this attempt failed and say why. Use ask_user to offer concrete "
    "next steps. Do not report results, values, plots, or file names -- none exist."
)


class SpecialistOutcomeState(AgentState):
    """Specialist-private baseline used to isolate this run's own artifacts."""

    specialist_initial_artifact_ids: NotRequired[Annotated[list[str], PrivateStateAttr]]


def _identity(runtime: Any) -> tuple[str, str] | None:
    context = getattr(runtime, "context", None)
    user_id = _context_value(context, "user_id")
    thread_id = _context_value(context, "thread_id")
    if not user_id or not thread_id:
        return None
    return user_id, thread_id


def _executions(state: dict[str, Any]) -> list[SandboxExecutionEvidence]:
    """Read back the typed evidence recorded at the sandbox backend boundary."""

    events: list[SandboxExecutionEvidence] = []
    for value in state.get("sandbox_execution_events", []):
        try:
            events.append(SandboxExecutionEvidence.model_validate(value))
        except ValueError:
            continue
    return events


def _bullets(label: str, items: list[str]) -> list[str]:
    return [f"{label}:", *[f"- {item}" for item in items]] if items else []


def _report_block(report: SpecialistReport | None) -> str:
    if report is None:
        return f"{REPORT_OPEN}\nNone returned.\n{REPORT_CLOSE}"
    lines = [f"Status: {report.status}", "", f"Finding: {report.finding}"]
    for label, items in (
        ("Evidence", report.evidence),
        ("Actions", report.actions),
        ("Limitations", report.limitations),
    ):
        if items:
            lines.extend(["", *_bullets(label, items)])
    body = "\n".join(lines)
    return f"{REPORT_OPEN}\n{body}\n{REPORT_CLOSE}"


def _execution_summary(executions: list[SandboxExecutionEvidence]) -> str:
    if not executions:
        return "Executions: none."
    detail = ", ".join(
        f"{event.status}"
        + ("" if event.exit_code is None else f" (exit {event.exit_code})")
        for event in executions
    )
    return f"Executions: {len(executions)} -- {detail}."


def _artifact_summary(artifacts: list[ResultArtifactEvidence]) -> str:
    if not artifacts:
        return "Result artifacts delivered: none."
    detail = "; ".join(
        f"{item.filename} ({item.kind}, id {item.artifact_id})" for item in artifacts
    )
    return f"Result artifacts delivered: {detail}."


def _verified_block(
    *,
    specialist: str,
    executions: list[SandboxExecutionEvidence],
    artifacts: list[ResultArtifactEvidence],
    undelivered: list[tuple[str, str]],
    failures: list[str],
) -> str:
    lines = [f"Specialist: {specialist}"]
    if failures:
        lines.append("STATUS: UNVERIFIED")
        lines.extend(f"- {reason}" for reason in failures)
    else:
        lines.append("STATUS: VERIFIED")
    lines.append(_execution_summary(executions))
    lines.append(_artifact_summary(artifacts))
    if undelivered:
        lines.append("Produced but NOT delivered to the user:")
        lines.extend(f"- {filename}: {reason}" for filename, reason in undelivered)
    if failures:
        lines.append(UNVERIFIED_DIRECTIVE)
    body = "\n".join(lines)
    return f"{VERIFIED_OPEN}\n{body}\n{VERIFIED_CLOSE}"


class SpecialistOutcomeMiddleware(AgentMiddleware[SpecialistOutcomeState]):
    """Pair the specialist's own report with server facts it cannot influence.

    Deep Agents forwards a subagent's `structured_response` verbatim, and falls
    back to the last non-empty `AIMessage` when there is none -- which turns an
    interrupted run into a confident-looking result. Composing the hand-off here
    closes that path: the supervisor always receives an explicit verdict, and the
    execution and artifact lines are read from the sandbox evidence and the
    artifact store rather than from anything the specialist wrote.
    """

    state_schema = SpecialistOutcomeState

    def __init__(self, *, specialist_name: str) -> None:
        self._specialist_name = specialist_name

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        identity = _identity(runtime)
        if identity is None:
            return {"specialist_initial_artifact_ids": []}
        refs = get_artifact_store().peek_result_refs(*identity)
        return {"specialist_initial_artifact_ids": [ref.artifact_id for ref in refs]}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return self.before_agent(state, runtime)

    def _own_artifacts(
        self,
        state: dict[str, Any],
        identity: tuple[str, str] | None,
    ) -> tuple[list[ResultArtifactEvidence], list[tuple[str, str]]]:
        """Result files this run added, plus notices for files that never arrived."""

        if identity is None:
            return [], []
        baseline = set(state.get("specialist_initial_artifact_ids", []))
        store = get_artifact_store()
        artifacts = [
            ResultArtifactEvidence(
                artifact_id=ref.artifact_id,
                filename=ref.filename,
                kind=ref.kind,
                mime_type=ref.mime_type,
                caption=ref.caption,
            )
            for ref in store.peek_result_refs(*identity)
            if ref.artifact_id not in baseline
        ]
        # Peek, never drain: the root response boundary owns delivery, and a
        # specialist that consumed these notices would hide them from the user.
        return artifacts, store.peek_undelivered(*identity)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        try:
            report = SpecialistReport.model_validate(state.get("structured_response"))
        except ValueError:
            report = None

        executions = _executions(state)
        artifacts, undelivered = self._own_artifacts(state, _identity(runtime))

        failures: list[str] = []
        if report is None:
            failures.append(NO_REPORT)
        if executions and not any(event.succeeded for event in executions):
            failures.append(NO_SUCCESSFUL_RUN)

        if report is not None and undelivered:
            report = report.model_copy(
                update={
                    "limitations": [
                        *report.limitations,
                        *[f"{name} was not delivered: {why}" for name, why in undelivered],
                    ]
                }
            )

        package = "\n\n".join(
            [
                _report_block(report),
                _verified_block(
                    specialist=self._specialist_name,
                    executions=executions,
                    artifacts=artifacts,
                    undelivered=undelivered,
                    failures=failures,
                ),
            ]
        )
        # Clearing `structured_response` routes the package through Deep Agents'
        # last-AIMessage path, so the supervisor reads prose instead of a JSON
        # envelope that would sit in checkpointed history for every later turn.
        return {
            "structured_response": None,
            "messages": [AIMessage(content=package)],
        }

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return self.after_agent(state, runtime)
