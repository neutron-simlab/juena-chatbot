"""The JüNA FastAPI application: core's factory, plus what is JüNA's alone.

The routes, the streaming vocabulary, the lifespans and the identity seam are
`juena_core.server.service.create_app`'s. What is passed in here is everything
core cannot know: SAML, background research, the Podman sandbox, and the one
paragraph of prompt text that closes a staged-input manifest.
"""

import warnings
from urllib.parse import urlsplit
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_core._api import LangChainBetaWarning

from juena_core.filesystem import get_filesystem
from juena_core.log import get_logger
from juena_core.sandbox.approvals import SandboxResumeInput, register_sandbox_interrupt
from juena_core.sandbox.runtime import (
    delete_runtime_workspace,
    sandbox_lifespan,
    stage_runtime_inputs,
)
from juena_core.server.service import StreamPolicy, ThreadWorkspace, create_app

from juena.core.config import global_config
from juena.research import runner as research_runner
from juena.server.auth.endpoints import router as auth_router
from juena.server.auth.principal import session_principal
from juena.server.auth.saml import saml_lifespan
from juena.server.research.endpoints import router as research_router
from juena.server.tracing import langfuse_lifespan, start_langfuse

# Imported for its side effect: the module self-registers the agent factory.
# Importing *this* module has to be enough, because the process serving the API
# is not always `main.py` -- production runs `uvicorn juena.server.service:app`,
# and the deployment plan splits the API and the UI into separate containers.
# Registering only from `main.py` leaves those processes with an empty registry,
# so every invocation would 404.
import juena.agents.juena_agent  # noqa: F401  register the JüNA supervisor

# The same trap, one layer down: a model class nobody imports is never created
# by `create_all`, and that fails at the first write rather than at startup.
import juena.server.database.models  # noqa: F401  add this app's two tables

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = get_logger(__name__)
logger.info("Service logging initialized")

# Rendering an approval card and resuming from one have a single owner and a
# single lifetime, so they register together -- and before the app is built,
# because an unregistered kind can neither be shown nor answered.
register_sandbox_interrupt()

# Before the app is built: the first run must already carry the handler.
if global_config.LANGFUSE_TRACING:
    start_langfuse()


@asynccontextmanager
async def _optional_saml_lifespan() -> AsyncGenerator[None, None]:
    """Start the SAML SP unless the development bypass replaces it.

    Skipping it is what lets a local checkout run without SP certificates,
    `xmlsec1`, or network access to the FZJ IdP.
    """
    if global_config.AUTH_DEV_BYPASS:
        logger.warning(
            "AUTH_DEV_BYPASS is enabled: SAML is not configured and /auth/dev-login "
            "issues sessions without verifying identity. Development only."
        )
        yield
        return
    async with saml_lifespan():
        yield


app = create_app(
    principal=session_principal,
    filesystem=get_filesystem(),
    browser_origins=tuple(dict.fromkeys(
        f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
        for url in (global_config.UI_PUBLIC_URL, global_config.JUENA_PUBLIC_URL)
    )),
    # Core's clarification arm plus the sandbox approval arm; the union exists
    # because `/resume` must accept exactly the kinds this deployment raises.
    resume_input=SandboxResumeInput,
    # Graph state is enough for a model reading `/inputs` through a file tool.
    # It is not enough for a container process that opens them, so the same
    # files are copied into the thread's sandbox mount.
    workspace=ThreadWorkspace(
        stage=stage_runtime_inputs,
        delete=delete_runtime_workspace,
    ),
    # Without the allowlist the sandbox's status events stop reaching the UI;
    # without the silent tool, `execute` stdout starts reaching the transcript.
    stream_policy=StreamPolicy(
        custom_event_types=frozenset({"sandbox_status"}),
        silent_tools=frozenset({"execute"}),
    ),
    # This block is prepended to the *supervisor's* turn, and the supervisor
    # has neither `/repos` nor Context7 -- it cannot read `/inputs` either.
    # So the instruction is about what to put in an objective, not what to do.
    closing_note=(
        "Conversation files are shared under `/workspace`; uploads are in `/workspace/uploads`. "
        "Pasted material is under `/inputs`. No attachment or pin is needed. Name "
        "the paths a specialist needs in the objective you send it."
    ),
    title="juena",
    version="1.0.0",
    extra_routers=(auth_router, research_router),
    # Entered in this order inside the database lifespan, and unwound in
    # reverse before it. Research starts last so recovery can build agents only
    # after their resources are ready; it stops before those resources close.
    # Langfuse enters first so it flushes last.
    extra_lifespans=(
        langfuse_lifespan,
        _optional_saml_lifespan,
        sandbox_lifespan,
        research_runner.lifespan,
    ),
)
