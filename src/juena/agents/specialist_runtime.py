"""Small shared runtime helpers for first-class JueNA specialists."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from juena.agents.backends import (
    FINDINGS_PREFIX,
    FindingsStateBackend,
    ReadOnlyFilesystemBackend,
    ReadOnlyInputsStateBackend,
)
from juena.agents.loop_guard import RepeatedToolCallMiddleware
from juena.agents.specialist_outcome import SpecialistOutcomeMiddleware
from juena.core.config import global_config
from juena.core.llms_providers import build_chat_model, get_available_providers, get_default_model
from juena.core.log import get_logger
from juena.sandbox.middleware import SandboxExecutionMiddleware
from juena.sandbox.policy import requires_execution_approval

logger = get_logger(__name__)

SUPERVISOR_MODEL_CALL_LIMIT = 50
SUPERVISOR_TOOL_CALL_LIMIT = 100
SPECIALIST_MODEL_CALL_LIMIT = 60
SPECIALIST_TOOL_CALL_LIMIT = 150
#: Background work gets a wider budget, because the reason the interactive one is
#: tight does not apply: nobody is watching a spinner. The objectives sent here
#: are the long ones by definition, and a run that exhausts its calls returns no
#: structured report at all -- so the whole wait produces `STATUS: UNVERIFIED`
#: and the user gets nothing. Wall-clock is the real bound on a background job
#: (`RESEARCH_TIMEOUT_SECONDS`); this only stops a loop from spinning forever.
BACKGROUND_MODEL_CALL_LIMIT = 150
BACKGROUND_TOOL_CALL_LIMIT = 400
#: Questions allowed per graph *invocation*, not per user turn: answering one
#: resumes the graph, which starts a new invocation and resets the count. So this
#: bounds one uninterrupted stretch of asking, and the prompt rule against
#: re-asking an answered question is what covers the rest.
ASK_USER_CALL_LIMIT = 3
MODEL_MAX_RETRIES = 3
#: Pinned rather than inherited: `ToolRetryMiddleware` defaults to 2 today, and a
#: silent upstream change to that default would silently change JueNA's policy.
TOOL_MAX_RETRIES = 2

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Appended to a specialist's prompt when it is compiled for background work.
#: The capabilities it describes are genuinely absent from that instance -- the
#: tools are not bound and the sandbox backend is not mounted -- so this tells
#: the model what it will find rather than asking it to refrain.
UNATTENDED_NOTICE = (
    "You are running unattended, as a background job the user started and then "
    "carried on with their conversation. Nobody is watching this run: `ask_user` "
    "is not available to you here, and neither is sandboxed execution. Decide "
    "with the evidence you can gather, and put whatever you could not settle in "
    "`limitations` -- an honest gap there is worth far more than a guess, "
    "because the supervisor can act on it and cannot act on a guess.\n\n"
    "Your report is the deliverable. The `/findings/` rule is unchanged here: "
    "write a file only if this objective asked you to, and if it did, the file "
    "itself comes back with the report for whoever is delegated to next."
)


class PromptResourceError(RuntimeError):
    """Raised when a required packaged prompt cannot be loaded."""


def load_markdown(package: str, filename: str) -> str:
    """Load one required Markdown resource and fail loudly when packaging is broken."""

    try:
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise PromptResourceError(f"Unable to load prompt resource {package}:{filename}") from exc
    if not text.strip():
        raise PromptResourceError(f"Prompt resource {package}:{filename} is empty")
    return text.rstrip() + "\n"


def _valid_skill_file(skill_file: Path) -> bool:
    """Perform the minimum Agent Skills validation needed before enabling middleware."""

    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.startswith("---\n"):
        return False
    try:
        _marker, frontmatter, body = text.split("---", 2)
        metadata = yaml.safe_load(frontmatter)
    except (ValueError, yaml.YAMLError):
        return False
    if not isinstance(metadata, dict):
        return False
    name = metadata.get("name")
    description = metadata.get("description")
    return bool(
        isinstance(name, str)
        and name == skill_file.parent.name
        and len(name) <= 64
        and _SKILL_NAME_RE.fullmatch(name)
        and isinstance(description, str)
        and description.strip()
        and len(description) <= 1024
        and body.strip()
    )


def has_authored_skills(skills_dir: Path | None) -> bool:
    """Return whether a specialist owns at least one valid ``SKILL.md``."""

    if skills_dir is None or not skills_dir.is_dir():
        return False
    return any(
        child.is_dir() and _valid_skill_file(child / "SKILL.md")
        for child in skills_dir.iterdir()
    )


def build_specialist_backend(
    *,
    repo_cache_root: Path | None = None,
    skills_dir: Path | None = None,
) -> CompositeBackend:
    """Build isolated staged-input, repository, and optional skill routes."""

    routes: dict[str, Any] = {FINDINGS_PREFIX: FindingsStateBackend()}
    if repo_cache_root is not None:
        routes["/repos/"] = ReadOnlyFilesystemBackend(
            root_dir=repo_cache_root,
            virtual_mode=True,
            label="read-only repository",
        )
    if has_authored_skills(skills_dir):
        routes["/skills/"] = ReadOnlyFilesystemBackend(
            root_dir=skills_dir,
            virtual_mode=True,
            label="read-only skill",
        )
    return CompositeBackend(default=ReadOnlyInputsStateBackend(), routes=routes)


def resilience_middleware(
    fallback_models: list[Any],
    *,
    model_call_limit: int,
    tool_call_limit: int,
    ask_user_bound: bool = True,
) -> list[Any]:
    """Build retry, fallback, and budget middleware in nesting order.

    **Every budget here is per graph invocation.** A resume after an `ask_user`
    answer or a sandbox approval is a new invocation, so these counts start again.
    No `thread_limit` is set: the stock one spans every later turn of the same
    chat, which would eventually stop a healthy long conversation. Bounding a
    *user turn* needs a turn identity JueNA does not have yet.

    Only `ModelCallLimitMiddleware` terminates. The tool limiters stay on
    `exit_behavior="continue"`, which is a mechanical cap -- the over-budget call
    does not run -- rather than a terminal one: `exit_behavior="end"` raises
    `NotImplementedError` when the offending response carries other pending tool
    calls, so a parallel batch that crossed the budget would crash the graph
    instead of ending it. The repeated-call guard closes the observed loop, and
    the model-call limit remains the final bound.
    """

    middleware: list[Any] = []
    if fallback_models:
        middleware.append(ModelFallbackMiddleware(*fallback_models))
    middleware.extend([
        ModelRetryMiddleware(max_retries=MODEL_MAX_RETRIES, on_failure="error"),
        ToolRetryMiddleware(max_retries=TOOL_MAX_RETRIES),
        ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="continue"),
    ])
    if ask_user_bound:
        # Asking is cheap for the model and expensive for the user, so one
        # uninterrupted stretch of asking is capped. Omitted where `ask_user` is
        # not bound at all: a budget on an absent tool can never fire.
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name="ask_user",
                run_limit=ASK_USER_CALL_LIMIT,
                exit_behavior="continue",
            )
        )
    return middleware


def build_fallback_models() -> list[Any]:
    """Build deployment-level fallbacks independently of the first request model."""

    fallback_provider = (global_config.FALLBACK_PROVIDER or "").strip().lower()
    if not fallback_provider:
        return []
    if not get_available_providers().get(fallback_provider, False):
        logger.warning(
            "FALLBACK_PROVIDER '%s' is not configured; running without a model fallback",
            fallback_provider,
        )
        return []
    fallback_model = get_default_model(fallback_provider)
    if not fallback_model:
        return []
    logger.info("Model fallback enabled: %s/%s", fallback_provider, fallback_model)
    return [build_chat_model(provider=fallback_provider, model=fallback_model)]


def build_specialist_middleware(
    *,
    backend: BackendProtocol,
    summarizer_model: Any,
    fallback_models: list[Any],
    filesystem_tool_descriptions: dict[str, str],
    specialist_name: str,
    # Omitted by a specialist that authors no skills, rather than pointed at a
    # directory that does not exist.
    skills_dir: Path | None = None,
    skills_label: str = "",
    enable_execution_approval: bool = False,
    unattended: bool = False,
) -> list[Any]:
    """Build the common middleware stack while keeping agent choices local."""

    # A specialist receives one already-scoped objective from the supervisor.
    # Do not expose write_todos here: it adds planning churn without contributing
    # execution or report evidence, and models can loop on identical todo updates.
    middleware: list[Any] = [
        SpecialistOutcomeMiddleware(specialist_name=specialist_name),
        RepeatedToolCallMiddleware(),
    ]
    if has_authored_skills(skills_dir):
        middleware.append(
            SkillsMiddleware(
                backend=backend,
                sources=[("/skills/", skills_label)],
            )
        )
    filesystem_middleware = FilesystemMiddleware(
        backend=backend,
        custom_tool_descriptions=filesystem_tool_descriptions,
        max_execute_timeout=global_config.SANDBOX_EXECUTION_TIMEOUT_SECONDS,
    )
    summarization_middleware = create_summarization_middleware(summarizer_model, backend)
    middleware.extend([
        filesystem_middleware,
        summarization_middleware,
        PatchToolCallsMiddleware(),
        *resilience_middleware(
            fallback_models,
            model_call_limit=(
                BACKGROUND_MODEL_CALL_LIMIT if unattended else SPECIALIST_MODEL_CALL_LIMIT
            ),
            tool_call_limit=(
                BACKGROUND_TOOL_CALL_LIMIT if unattended else SPECIALIST_TOOL_CALL_LIMIT
            ),
            ask_user_bound=not unattended,
        ),
    ])
    if enable_execution_approval:
        # Records what the human approved, reading the command straight from the
        # tool call. Runs that never needed approval get no approval records.
        middleware.append(SandboxExecutionMiddleware())
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "execute": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": (
                            "Review the generated command before it runs in the "
                            "isolated sandbox."
                        ),
                        "when": requires_execution_approval,
                    }
                },
            )
        )
    return middleware
