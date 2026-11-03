"""Create the optional queue or prove a real worker can return a PNG.

The launcher pipes this file into the running app. Smoke tests use a disposable
workspace, never an existing conversation, and retain failed results for diagnosis.
"""

import argparse
import time
from uuid import uuid4

PLOT_SCRIPT = '''
import os
import socket
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
assert os.getuid() != 0, "execution must be non-root"
assert not Path("/var/run/docker.sock").exists()
assert not Path("/etc/juena").exists()
assert not os.getenv("DATABASE_URL")
assert not os.getenv("BLABLADOR_API_KEY")
with socket.socket() as probe:
    probe.settimeout(1)
    try:
        connected = probe.connect_ex(("1.1.1.1", 443)) == 0
    except OSError:
        connected = False
    assert not connected, "execution must have no external network"
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
plt.plot(x, np.sin(x))
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.savefig("/workspace/outputs/sandbox-smoke.png")
print("sandbox plot and isolation checks passed")
'''


def check_worker(jobs, store, *, pause=time.sleep) -> bool:
    """Submit through the same queue as chat; require both PNG bytes and evidence."""
    from PIL import Image

    identity = "smoke-" + uuid4().hex
    jobs.open()
    try:
        for _ in range(30):
            if jobs.worker_is_alive():
                break
            pause(1)
        else:
            print("Sandbox worker did not connect within 30 seconds.")
            return False
        store.ensure(identity, identity)
        store.ensure_outputs(identity)
        script = store.workspace_dir(identity) / ".scratch/smoke.py"
        script.write_text(PLOT_SCRIPT)
        script.chmod(0o660)
        result = jobs.execute(tenant_id=identity, workspace_id=identity,
                              command="python /workspace/.scratch/smoke.py",
                              timeout_seconds=60, max_output_bytes=10000)
        image = store.workspace_dir(identity) / "outputs/sandbox-smoke.png"
        if result.status != "completed" or result.exit_code != 0 or not image.is_file():
            print(f"Sandbox smoke test failed: {result.status}\n{result.output}")
            return False
        try:
            with Image.open(image) as png:
                if png.format != "PNG":
                    raise ValueError("not a PNG")
                png.verify()
        except (OSError, ValueError, SyntaxError):
            print("Sandbox returned an invalid PNG.")
            return False
        if "sandbox-smoke.png" not in result.artifact_names:
            print("Worker produced a PNG but did not report it as a collected artifact.")
            return False
        print("Sandbox passed: queued job executed as non-root, network blocked, PNG collected.")
        # A completed job means the worker removed its container before marking
        # the row terminal. Failed/time-limited jobs retain their workspace.
        store.delete(identity, identity)
        return True
    finally:
        jobs.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--require-enabled", action="store_true")
    args = parser.parse_args()
    from juena.core.config import global_config as config
    from juena_core.filesystem.storage import WorkspaceStore
    from juena_core.sandbox.jobs import SandboxJobs, database_url

    if args.require_enabled and not config.SANDBOX_ENABLED:
        print("Execution is still disabled in the running application.")
        return 1
    if args.initialize:
        from sqlalchemy import create_engine
        from juena_core.sandbox.models import SandboxJob

        url = database_url(config.DATABASE_URL).replace("postgresql://", "postgresql+psycopg://", 1)
        engine = create_engine(url)
        try:
            SandboxJob.__table__.create(engine, checkfirst=True)
        finally:
            engine.dispose()
        print("Sandbox job table is ready.")
        return 0

    jobs = SandboxJobs(config.DATABASE_URL, concurrency=config.SANDBOX_CONCURRENCY)
    store = WorkspaceStore(config.SANDBOX_WORKSPACE_ROOT,
                           limit_bytes=config.SANDBOX_WORKSPACE_LIMIT_BYTES,
                           idle_ttl_seconds=config.SANDBOX_IDLE_TTL_SECONDS)
    return int(not check_worker(jobs, store))


if __name__ == "__main__":
    raise SystemExit(main())
