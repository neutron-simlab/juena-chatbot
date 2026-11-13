"""The JüNA FastAPI application: core's factory, plus what is JüNA's alone.

`juena_core.server.service.create_app` provides the routes, the streaming, the
startup and shutdown work, and the hook for signing users in. This module passes
in what core cannot know: SAML, background research, the Podman sandbox, and
the paragraph of prompt text that ends the list of staged input files.
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
from juena.core.personal_models import credential_error_message
from juena.research import runner as research_runner
from juena.server.auth.endpoints import router as auth_router
from juena.server.auth.cors import SamlLogoutCORSMiddleware
from juena.server.credential_endpoints import router as credential_router, require_chat_credentials
from juena.server.auth.principal import session_principal
from juena.server.auth.saml import saml_lifespan
from juena.server.tool_labels import JUENA_TOOL_LABELS
from juena.server.research.endpoints import router as research_router
from juena.server.tracing import langfuse_lifespan, start_langfuse
from juena.uploads import UPLOAD_POLICY

# Imported for its side effect: the module registers the agent. Import it here,
# not only in `main.py`: production runs `uvicorn juena.server.service:app`,
# which never runs `main.py`, and without the agent every run answers 404.
import juena.agents.juena_agent  # noqa: F401  register the JüNA supervisor

# Also for its side effect: a model class nobody imports is never created by
# `create_all`, and that fails at the first write rather than at startup.
import juena.server.database.models  # noqa: F401  add this app's tables

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = get_logger(__name__)
logger.info("Service logging initialized")

# Registers both how the approval card is shown and how it is answered. Do it
# before the app is built: an unregistered kind can neither be shown nor answered.
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
    invocation_guard=require_chat_credentials,
    invocation_error=credential_error_message,
    filesystem=get_filesystem(),
    upload_policy=UPLOAD_POLICY,
    run_controls=True,
    browser_origins=tuple(dict.fromkeys(
        f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
        for url in (global_config.UI_PUBLIC_URL, global_config.JUENA_PUBLIC_URL)
    )),
    # Core refuses a browser request that changes something unless it comes
    # from the origins above. IFFLogin's page posts to these two from its own
    # site, and neither trusts the session cookie: each is authorized by the
    # SAML signature it carries.
    origin_exempt_paths=("/auth/acs", "/auth/sls"),
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
    # without the silent tool, `execute` stdout reaches the transcript whenever
    # tool payloads are switched on. The labels word this deployment's own
    # tools in the activity log.
    stream_policy=StreamPolicy(
        custom_event_types=frozenset({"sandbox_status"}),
        silent_tools=frozenset({"execute", "analyze_tiff"}),
        tool_labels=JUENA_TOOL_LABELS,
    ),
    # Prepended to the supervisor's turn. The supervisor has neither `/repos`
    # nor Context7 and cannot read `/inputs` either, so the note says what to
    # put in an objective, not what to do.
    closing_note=(
        "Conversation files are shared under `/workspace`; uploads are in `/workspace/uploads`. "
        "Pasted material is under `/inputs`. No attachment or pin is needed. Name "
        "the paths a specialist needs in the objective you send it."
    ),
    title="juena",
    version="1.0.0",
    extra_routers=(auth_router, credential_router, research_router),
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

app.add_middleware(SamlLogoutCORSMiddleware, idp_metadata_url=global_config.SAML_IDP_METADATA_URL)
