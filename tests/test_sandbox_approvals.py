"""Server-side validation for sandbox interrupt decisions."""

from types import SimpleNamespace

import pytest
from langgraph.types import Interrupt

from juena.sandbox.approvals import ApprovalError, build_resume_command
from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.schemas import ApprovalResumeInput, ClarificationResumeInput
from juena.sandbox.policy import requires_execution_approval


def _interrupt(interrupt_id: str = "interrupt-1") -> Interrupt:
    return Interrupt(
        id=interrupt_id,
        value={
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "python plot.py", "timeout": 600},
                    "description": "Create the requested plot",
                }
            ],
            "review_configs": [
                {"allowed_decisions": ["approve", "edit", "reject"]}
            ],
        },
    )


class _Agent:
    def __init__(self, interrupts):
        self.snapshot = SimpleNamespace(
            tasks=[SimpleNamespace(interrupts=tuple(interrupts))]
        )

    async def aget_state(self, config):
        return self.snapshot


@pytest.fixture
def artifact_store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    yield store
    set_artifact_store_for_tests(None)


@pytest.mark.asyncio
async def test_approve_resumes_exact_interrupt(artifact_store) -> None:
    command = await build_resume_command(
        agent=_Agent([_interrupt()]),
        config={},
        user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="thread-a",
            interrupt_id="interrupt-1",
            decision="approve",
        ),
    )

    assert command.resume == {
        "interrupt-1": {"decisions": [{"type": "approve"}]}
    }


@pytest.mark.asyncio
async def test_edit_changes_only_command_text(artifact_store) -> None:
    command = await build_resume_command(
        agent=_Agent([_interrupt()]),
        config={},
        user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="thread-a",
            interrupt_id="interrupt-1",
            decision="edit",
            edited_command="python revised.py",
        ),
    )

    [decision] = command.resume["interrupt-1"]["decisions"]
    assert decision["edited_action"] == {
        "name": "execute",
        "args": {"command": "python revised.py", "timeout": 600},
    }


@pytest.mark.asyncio
async def test_reject_sends_a_reject_decision(artifact_store) -> None:
    command = await build_resume_command(
        agent=_Agent([_interrupt()]),
        config={},
        user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="thread-a",
            interrupt_id="interrupt-1",
            decision="reject",
        ),
    )

    [decision] = command.resume["interrupt-1"]["decisions"]
    assert decision["type"] == "reject"


@pytest.mark.asyncio
async def test_stale_or_cross_thread_interrupt_is_rejected(artifact_store) -> None:
    with pytest.raises(ApprovalError, match="stale"):
        await build_resume_command(
            agent=_Agent([_interrupt("different")]),
            config={},
            user_id="user-a",
            payload=ApprovalResumeInput(
                thread_id="thread-a",
                interrupt_id="interrupt-1",
                decision="approve",
            ),
        )


@pytest.mark.asyncio
async def test_non_execute_interrupt_is_rejected(artifact_store) -> None:
    interrupt = _interrupt()
    interrupt.value["action_requests"][0]["name"] = "write_file"
    with pytest.raises(ApprovalError, match="not a pending sandbox command"):
        await build_resume_command(
            agent=_Agent([interrupt]),
            config={},
            user_id="user-a",
            payload=ApprovalResumeInput(
                thread_id="thread-a",
                interrupt_id="interrupt-1",
                decision="approve",
            ),
        )


def test_redundant_artifact_export_does_not_request_human_approval() -> None:
    request = SimpleNamespace(
        tool_call={
            "name": "execute",
            "args": {
                "command": (
                    "python -c \"import base64; "
                    "base64.b64encode(open('/workspace/outputs/plot.png','rb').read())\""
                )
            },
        }
    )

    assert requires_execution_approval(request) is False
    request.tool_call["args"]["command"] = "python /workspace/plot.py"
    assert requires_execution_approval(request) is True


def _question(interrupt_id: str = "interrupt-2") -> Interrupt:
    return Interrupt(
        id=interrupt_id,
        value={
            "kind": "clarification",
            "asked_by": "software-specialist",
            "question": "Which background subtraction should I use?",
            "options": ["Solvent-only", "Empty cell"],
        },
    )


@pytest.mark.asyncio
async def test_answer_resumes_the_asking_agent_with_the_users_own_words(
    artifact_store,
) -> None:
    command = await build_resume_command(
        agent=_Agent([_question()]),
        config={},
        user_id="user-a",
        payload=ClarificationResumeInput(
            kind="clarification",
            thread_id="thread-a",
            interrupt_id="interrupt-2",
            answer="  Empty cell  ",
        ),
    )

    assert command.resume == {"interrupt-2": "Empty cell"}


@pytest.mark.asyncio
async def test_an_answer_cannot_approve_a_pending_sandbox_command(artifact_store) -> None:
    with pytest.raises(ApprovalError, match="not a pending question"):
        await build_resume_command(
            agent=_Agent([_interrupt()]),
            config={},
            user_id="user-a",
            payload=ClarificationResumeInput(
                kind="clarification",
                thread_id="thread-a",
                interrupt_id="interrupt-1",
                answer="approve it",
            ),
        )
