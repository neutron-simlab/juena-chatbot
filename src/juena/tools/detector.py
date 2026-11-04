"""Structured detector analysis through the existing isolated execution worker."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import shlex
from typing import Annotated, Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import Field

from juena_core.runtime_context import _context_value
from juena_core.sandbox.evidence import capture_sandbox_execution
from juena_core.sandbox.runtime import RuntimePodmanSandboxBackend, sandbox_enabled
from juena_core.filesystem.storage import assert_safe_path
from juena_core.schema.interrupts import ExecutionEvidence
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext


@tool
async def analyze_tiff(
    operation: Literal["inspect", "histogram", "roi", "compare"],
    paths: Annotated[list[str], Field(min_length=1, max_length=5)],
    runtime: ToolRuntime[RuntimeModelContext],
    pages: list[int] | None = None,
    roi: list[int] | None = None,
    scale: Literal["linear", "log"] = "linear",
) -> Command | ToolMessage:
    """Inspect detector TIFFs, histogram pixels, measure a region, or compare frames.

    Paths are absolute conversation /workspace paths. Pages are zero-based: supply
    one index (default 0) or one per path. ROI bounds are x0,y0,x1,y1 with a top-left
    origin and exclusive upper bounds. Compare exactly two equal-sized planes,
    calculating second minus first. Statistics retain raw zero/negative values;
    scale affects previews only. Read the TIFF skill for scientific conventions.

    Runs automatically in the isolated worker and delivers one PNG, a JSON report,
    and a reproduction script. Use its returned numerical evidence; do not generate
    scripts, read TIFF bytes with text tools, or re-read collected outputs.
    """
    if not sandbox_enabled():
        return ToolMessage(content="TIFF inspection is unavailable because execution is disabled. The uploaded original is retained.", tool_call_id=runtime.tool_call_id, status="error")
    for path in paths:
        assert_safe_path(path)
        if not path.startswith("/workspace/") or PurePosixPath(path).suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("TIFF inputs must be .tif/.tiff files under /workspace")
    run_id = _context_value(runtime.context, "run_id")
    if not run_id:
        raise RuntimeError("TIFF analysis requires an authenticated graph run_id")
    payload = {"operation": operation, "inputs": paths, "pages": pages, "roi": roi, "scale": scale}
    command = shlex.join([
        "python", "-c",
        "import json,sys; from analysis.tiff import execute_request; execute_request(json.loads(sys.argv[1]))",
        json.dumps(payload),
    ])
    with capture_sandbox_execution(run_id) as events:
        result = await RuntimePodmanSandboxBackend().aexecute(command)
    if not events:
        events = [ExecutionEvidence(graph_run_id=run_id, command=command, status="tool_error", exit_code=None)]
    return Command(update={
        "messages": [ToolMessage(
            content=result.output, tool_call_id=runtime.tool_call_id,
            status="success" if events[-1].succeeded else "error",
        )],
        "execution_events": [event.model_dump(mode="json") for event in events],
    })
