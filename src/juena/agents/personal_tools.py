"""Expose external tools only when the current user has a readable credential."""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from juena.server.credentials import CredentialError, personal_key


class PersonalToolsMiddleware(AgentMiddleware):
    """Hide a service's tools unless the current user has a readable key for it.

    Also tells the model, in the system message, whether the service is
    available.
    """

    def __init__(self, service: str, tool_names: tuple[str, ...]):
        """Initialize the middleware.

        Args:
            service: Personal credential service, e.g. ``"tavily"``.
            tool_names: Names of the tools that need that service's key.
        """
        self.service = service
        self.tool_names = frozenset(tool_names)

    async def awrap_model_call(self, request, handler):
        """Check the user's key, then call the model with or without the tools.

        Args:
            request: The model request.
            handler: Calls the model with a request.

        Returns:
            The model's response.
        """
        context = request.runtime.context
        user_id = context.get("user_id") if isinstance(context, dict) else getattr(context, "user_id", None)
        try:
            await personal_key(user_id, self.service)
            available = True
        except CredentialError:
            available = False
        tools = request.tools if available else [
            t for t in request.tools
            if (t.get("name") if isinstance(t, dict) else t.name) not in self.tool_names
        ]
        status = (f"{self.service.capitalize()} is available using this user's personal key."
                  if available else f"{self.service.capitalize()} is unavailable. Do not claim external verification or invent citations. The user can add a key in My profile.")
        content = list(request.system_message.content_blocks) if request.system_message else []
        content.append({"type": "text", "text": status})
        return await handler(request.override(tools=tools, system_message=SystemMessage(content=content)))
