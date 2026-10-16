"""Verify specialist model selection and usable long-context budgets."""

from types import SimpleNamespace

import pytest
from deepagents.middleware.summarization import SummarizationMiddleware

from juena.agents.specialists.neutron_research import agent as neutron
from juena.agents.specialists.software import agent as software
from juena_core.llms_providers import build_chat_model
from juena_core.schema.llm_models import BlabladorModelName


@pytest.mark.parametrize("unattended", [False, True])
@pytest.mark.parametrize("specialist", [software, neutron])
def test_specialists_use_juwels_qwen_with_matching_compaction(
    monkeypatch, specialist, unattended
) -> None:
    captured = {}

    def capture_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(specialist, "create_agent", capture_agent)
    summarizer = build_chat_model(
        provider="blablador",
        model=BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value,
        max_tokens=32_768,
    )
    common = {
        "summarizer_model": summarizer,
        "fallback_models": [],
        "unattended": unattended,
    }
    if specialist is software:
        software.build_software_specialist(
            repo_manager=SimpleNamespace(cache_dir=None),
            vector_index=None,
            has_repo_research=False,
            context7_tools=(),
            **common,
        )
    else:
        neutron.build_neutron_research_specialist(tavily_tool=None, **common)

    model = captured["model"]
    assert model.model_name == (
        "90 - Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster - "
        "An excellent model from September 2026"
    )
    assert model.max_tokens == 32_768
    assert model.profile["max_input_tokens"] == 907_232
    compaction = next(
        item for item in captured["middleware"]
        if isinstance(item, SummarizationMiddleware)
    )
    assert compaction.model is summarizer
    # The normal 85% trigger now retains long histories, while reserving output
    # and headroom for another tool result before reaching the served 940k limit.
    assert not compaction._should_summarize([], 700_000)
    assert compaction._should_summarize([], 800_000)


@pytest.mark.parametrize("specialist", [software, neutron])
def test_specialists_stream_with_one_retry_layer_and_a_real_fallback(monkeypatch, specialist) -> None:
    """Streaming bounds the gap between tokens; ModelRetryMiddleware owns retries;
    a fallback that calls the specialist's own model is dropped."""
    from langchain.agents.middleware import ModelFallbackMiddleware

    captured = {}
    monkeypatch.setattr(specialist, "create_agent", lambda **kwargs: captured.update(kwargs) or object())
    same_model_other_settings = build_chat_model(
        provider="blablador", model=BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value
    )
    other_model = build_chat_model(provider="blablador", model=BlabladorModelName.GPT_OSS.value)
    common = {
        "summarizer_model": other_model,
        "fallback_models": [same_model_other_settings, other_model],
    }
    if specialist is software:
        software.build_software_specialist(
            repo_manager=SimpleNamespace(cache_dir=None),
            vector_index=None,
            has_repo_research=False,
            context7_tools=(),
            **common,
        )
    else:
        neutron.build_neutron_research_specialist(tavily_tool=None, **common)

    model = captured["model"]
    assert model.streaming is True
    assert model.stream_usage is True
    assert model.max_retries == 0
    fallback = next(m for m in captured["middleware"] if isinstance(m, ModelFallbackMiddleware))
    assert fallback.models == [other_model]
