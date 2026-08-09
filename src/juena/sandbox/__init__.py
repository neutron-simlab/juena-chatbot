"""Tenant-scoped sandbox execution and generated-artifact support."""


def sandbox_enabled() -> bool:
    """Load application configuration only when the application asks for it."""
    from juena.sandbox.runtime import sandbox_enabled as runtime_sandbox_enabled

    return runtime_sandbox_enabled()

__all__ = ["sandbox_enabled"]
