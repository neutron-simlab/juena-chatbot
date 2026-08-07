"""Starter topics offered on an empty chat."""

from dataclasses import dataclass

@dataclass(frozen=True)
class StarterPrompt:
    """A chip label and the question it sends."""

    label: str
    prompt: str


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
            "briefly explain what each one is for, and ask which repository I would "
            "like to explore."
        ),
    ),
)


def build_starter_prompts() -> list[StarterPrompt]:
    """Return the two product-defined immediate-send prompts."""
    return list(STARTER_PROMPTS)
