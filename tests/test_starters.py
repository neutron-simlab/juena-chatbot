"""Tests for the two product-defined starter topics."""

from app.starters import build_starter_prompts


def test_starters_are_exactly_the_two_neutron_topics() -> None:
    starters = build_starter_prompts()

    assert [starter.label for starter in starters] == [
        "General neutron science",
        "Neutron software and codes",
    ]
    assert starters[0].prompt == (
        "Give me a brief introduction to neutron science and the kinds of "
        "neutron-science questions you can help me answer."
    )
    assert starters[1].prompt == (
        "List all neutron-related software repositories currently indexed in JüNA, "
        "briefly explain what each one is for, and ask which repository I would "
        "like to explore."
    )


def test_each_build_returns_a_fresh_list() -> None:
    first = build_starter_prompts()
    second = build_starter_prompts()

    assert first == second
    assert first is not second
