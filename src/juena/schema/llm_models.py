from enum import StrEnum
from typing import TypeAlias, Dict, List


class Provider(StrEnum):
    """Supported LLM providers in the current system"""
    OPENAI = "openai"
    BLABLADOR = "blablador"


class OpenAIModelName(StrEnum):
    """OpenAI chat models intentionally offered by this deployment."""
    
    GPT_4O_MINI = "gpt-4o-mini"


class BlabladorModelName(StrEnum):
    """Blablador model names (OpenAI-compatible API).
    The UI offers the curated Blablador model options used by this project.
    """

    GPT_OSS = "01 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
    MINIMAX_M27 = "01 - MiniMax-M2.7 - our best model as of April, 2026"
    QWEN35_122B = "02 - Qwen3.5-122B-A10B-FP8, general purpose large model"


#: Total context each Blablador model serves -- input plus output -- in tokens.
#: Read from the provider itself (`GET /v1/models` reports `max_model_len`) on
#: 2026-08-19, not guessed. LangChain has no profile entry for these names, and
#: without one the summarization middleware falls back to a fixed 170,000-token
#: threshold, which is above the window it exists to protect. Re-check with:
#:   curl -H "Authorization: Bearer $BLABLADOR_API_KEY" $BLABLADOR_BASE_URL/models
BLABLADOR_MAX_MODEL_LEN: Dict[str, int] = {
    BlabladorModelName.GPT_OSS.value: 131_072,
    BlabladorModelName.MINIMAX_M27.value: 131_072,
    BlabladorModelName.QWEN35_122B.value: 262_144,
}


BLABLADOR_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    BlabladorModelName.GPT_OSS.value: "GPT-OSS-120b",
    BlabladorModelName.MINIMAX_M27.value: "MiniMax-M2.7",
    BlabladorModelName.QWEN35_122B.value: "QWEN3.5-122B",
}


def get_blablador_model_display_name(model_id: str) -> str:
    """Return the UI display name for a Blablador model id."""
    return BLABLADOR_MODEL_DISPLAY_NAMES.get(model_id, model_id)


AllModelEnum: TypeAlias = OpenAIModelName | BlabladorModelName


PROVIDER_MODELS: Dict[Provider, List[str]] = {
    Provider.OPENAI: [model.value for model in OpenAIModelName],
    Provider.BLABLADOR: [model.value for model in BlabladorModelName],
}


def get_models_for_provider(provider: Provider) -> List[str]:
    """Get list of available models for a specific provider"""
    return PROVIDER_MODELS.get(provider, [])


def get_default_model_for_provider(provider: Provider) -> str:
    """Get the default model for a specific provider"""
    defaults = {
        Provider.OPENAI: OpenAIModelName.GPT_4O_MINI.value,
        Provider.BLABLADOR: BlabladorModelName.GPT_OSS.value,
    }
    return defaults.get(provider, "")

