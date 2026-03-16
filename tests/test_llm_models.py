"""Tests for supported LLM model catalogs."""

from juena.schema.llm_models import (
    BlabladorModelName,
    Provider,
    get_blablador_model_display_name,
    get_default_model_for_provider,
    get_models_for_provider,
)


def test_blablador_models_include_minimax() -> None:
    models = get_models_for_provider(Provider.BLABLADOR)

    assert BlabladorModelName.GPT_OSS.value in models
    assert BlabladorModelName.MINIMAX_M25.value in models


def test_blablador_display_name_for_minimax() -> None:
    assert (
        get_blablador_model_display_name(BlabladorModelName.MINIMAX_M25.value)
        == "MiniMax-M2.5"
    )


def test_blablador_default_model_remains_gpt_oss() -> None:
    assert get_default_model_for_provider(Provider.BLABLADOR) == BlabladorModelName.GPT_OSS.value
