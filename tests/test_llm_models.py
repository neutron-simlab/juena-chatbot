"""Tests for supported LLM model catalogs."""

from juena_core.schema.llm_models import (
    BlabladorModelName,
    Provider,
    get_blablador_model_display_name,
    get_default_model_for_provider,
    get_models_for_provider,
)


def test_blablador_models_include_minimax() -> None:
    models = get_models_for_provider(Provider.BLABLADOR)

    assert BlabladorModelName.GPT_OSS.value in models
    assert BlabladorModelName.MINIMAX_M27.value in models
    assert BlabladorModelName.QWEN35_122B.value in models


def test_blablador_display_name_for_minimax() -> None:
    assert (
        get_blablador_model_display_name(BlabladorModelName.MINIMAX_M27.value)
        == "MiniMax-M2.7"
    )


def test_blablador_display_name_for_qwen35_122b() -> None:
    assert (
        get_blablador_model_display_name(BlabladorModelName.QWEN35_122B.value)
        == "QWEN3.5-122B"
    )


def test_blablador_display_names_hide_provider_ranking_prefixes() -> None:
    labels = [
        get_blablador_model_display_name(model.value)
        for model in BlabladorModelName
    ]

    assert labels == ["GPT-OSS-120b", "MiniMax-M2.7", "QWEN3.5-122B"]
    assert all(not label[:1].isdigit() for label in labels)


def test_blablador_default_model_remains_gpt_oss() -> None:
    assert get_default_model_for_provider(Provider.BLABLADOR) == BlabladorModelName.GPT_OSS.value
