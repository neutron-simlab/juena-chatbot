"""Deferred runtime lookup keeps secrets out of shared model configuration."""


async def blablador_key() -> str:
    """Read the personal Blablador key of the user the current run belongs to.

    Returns:
        The decrypted key.

    Raises:
        CredentialError: There is no signed-in run, or the user has no usable key.
    """
    from langgraph.runtime import get_runtime
    from juena.server.credentials import CredentialError, personal_key

    try:
        context = get_runtime().context
    except RuntimeError:
        raise CredentialError("Blablador requires an authenticated JüNA run.") from None
    user_id = context.get("user_id") if isinstance(context, dict) else getattr(context, "user_id", None)
    return await personal_key(user_id, "blablador")


def credential_error_message(error: Exception) -> str | None:
    """User-facing failures without upstream bodies, URLs, or credentials.

    Args:
        error: The exception a run raised.

    Returns:
        A safe message for a credential or Blablador failure, or None for any
        other error.
    """
    from openai import APIError, APIStatusError
    from juena.server.credentials import CredentialError

    cause = error
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, CredentialError):
            return str(cause)
        cause = cause.__cause__
    if isinstance(error, APIStatusError):
        if error.status_code in (401, 403):
            return "Blablador rejected your key. Check or replace it in My profile."
        if error.status_code == 429:
            return "Blablador is rate limiting your requests. Try again shortly."
    if isinstance(error, APIError):
        return "Blablador could not complete the request. Try again shortly."
    return None
