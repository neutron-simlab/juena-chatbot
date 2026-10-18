"""Tests for the two product-defined starter topics."""

from app.starters import PLOT_STARTER, build_starter_prompts, global_config


def test_starters_are_exactly_the_two_neutron_topics_when_preview_is_off(monkeypatch) -> None:
    monkeypatch.setattr(global_config, "SANDBOX_ENABLED", False)
    monkeypatch.setattr(global_config, "SANDBOX_STARTER_PREVIEW", False)
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
        "briefly explain what each one is for."
    )


def test_each_build_returns_a_fresh_list() -> None:
    first = build_starter_prompts()
    second = build_starter_prompts()

    assert first == second
    assert first is not second


def test_plot_starter_appears_only_when_sandbox_is_enabled(monkeypatch) -> None:
    monkeypatch.setattr(global_config, "SANDBOX_STARTER_PREVIEW", False)
    monkeypatch.setattr(global_config, "SANDBOX_ENABLED", True)

    starters = build_starter_prompts()

    assert starters[-1] == PLOT_STARTER
    assert starters[-1].label == "Plot with Python"
    assert "Matplotlib" in starters[-1].prompt
    assert "execute" in starters[-1].prompt
    assert "/workspace/outputs" in starters[-1].prompt
    assert "do not encode the image" in starters[-1].prompt


def test_plot_starter_can_be_previewed_in_development(monkeypatch) -> None:
    monkeypatch.setattr(global_config, "SANDBOX_ENABLED", False)
    monkeypatch.setattr(global_config, "SANDBOX_STARTER_PREVIEW", True)

    assert build_starter_prompts()[-1] == PLOT_STARTER
