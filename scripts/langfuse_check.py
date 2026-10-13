"""Prove the self-hosted Langfuse really stores traces: `juena langfuse-check`.

A green health check only says the web server is up. This sends one test
observation with the chatbot's own key pair and waits until Langfuse can read
it back -- which needs the keys to match, the API to accept the batch and the
worker to have written it into ClickHouse.

Usage: python scripts/langfuse_check.py [--if-configured] URL

The keys are read the way the application reads them: python-dotenv on
JUENA_ENV_PATH or this repository's .env, with the process environment taking
precedence. So quotes and inline comments in .env mean here what they mean to
the application. The keys are never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

READ_BACK_SECONDS = 60
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument(
        "--if-configured",
        action="store_true",
        help="succeed quietly when .env has no Langfuse keys yet",
    )
    args = parser.parse_args()

    # The same rule as juena.core.config.
    load_dotenv(os.getenv("JUENA_ENV_PATH") or REPO_ROOT / ".env", override=False)
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") or ""
    if not public_key or not secret_key:
        if args.if_configured:
            print("No Langfuse keys in .env yet, so tracing is not checked.")
            print("Copy the tracing block from env.example, then 'juena langfuse-check'.")
            return 0
        print(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in .env "
            "(see env.example).",
            file=sys.stderr,
        )
        return 1

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=args.base_url,
        tracing_enabled=True,
    )
    try:
        try:
            client.auth_check()
        except Exception:
            print(
                f"Langfuse at {args.base_url} rejected the key pair in .env.\n"
                "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY must equal "
                "LANGFUSE_INIT_PROJECT_PUBLIC_KEY / _SECRET_KEY in "
                "deploy/langfuse/.env -- and those only take effect on the "
                "first start against empty volumes.",
                file=sys.stderr,
            )
            return 1

        with client.start_as_current_observation(
            name="juena-langfuse-check", as_type="span"
        ) as span:
            trace_id = span.trace_id
        client.flush()

        deadline = time.monotonic() + READ_BACK_SECONDS
        while time.monotonic() < deadline:
            if client.api.observations.get_many(trace_id=trace_id).data:
                print(f"Langfuse stores traces: test trace {trace_id} is readable.")
                return 0
            time.sleep(2)

        print(
            f"Langfuse accepted the keys but the test trace {trace_id} was not "
            f"readable within {READ_BACK_SECONDS} s: the worker is probably not "
            "ingesting. See 'juena langfuse-logs'.",
            file=sys.stderr,
        )
        return 1
    finally:
        client.shutdown()


if __name__ == "__main__":
    sys.exit(main())
