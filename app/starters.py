"""Starter topics offered on an empty chat."""

from dataclasses import dataclass

from juena.core.config import global_config


@dataclass(frozen=True)
class StarterPrompt:
    """A chip label and the question it sends."""

    label: str
    prompt: str
    requires_sandbox: bool = False


STARTER_PROMPTS: tuple[StarterPrompt, ...] = (
    StarterPrompt(
        label="General neutron science",
        prompt=(
            "Give me a brief introduction to neutron science and the kinds of "
            "neutron-science questions you can help me answer."
        ),
    ),
    StarterPrompt(
        label="Neutron software and codes",
        prompt=(
            "List all neutron-related software repositories currently indexed in JüNA, "
            "briefly explain what each one is for."
        ),
    ),
)

PLOT_STARTER = StarterPrompt(
    label="Plot with Python",
    prompt=(
        "Create and execute a Python script in the sandbox now using NumPy and "
        "Matplotlib. Plot a sine wave for x from 0 to 2π, label both axes, add "
        "a title and grid, and save both the reusable script and PNG under "
        "`/workspace/outputs` so they appear in the chat as downloads. Do not "
        "only show me code and do not encode the image as text."
    ),
    requires_sandbox=True,
)


def build_starter_prompts() -> list[StarterPrompt]:
    """Return immediate-send prompts available in this deployment."""
    prompts = list(STARTER_PROMPTS)
    if global_config.SANDBOX_ENABLED or global_config.SANDBOX_STARTER_PREVIEW:
        prompts.append(PLOT_STARTER)
    return prompts
