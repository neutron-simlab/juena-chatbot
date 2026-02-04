from enum import StrEnum
from typing import TypeAlias, Dict, List


class Provider(StrEnum):
    """Supported LLM providers in the current system"""
    OPENAI = "openai"
    BLABLADOR = "blablador"
    # Future providers can be added here
    # ANTHROPIC = "anthropic"
    # GOOGLE = "google"
    # AZURE_OPENAI = "azure_openai"


class OpenAIModelName(StrEnum):
    """OpenAI model names - https://platform.openai.com/docs/models/gpt-4o"""
    
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_35_TURBO = "gpt-3.5-turbo"


class BlabladorModelName(StrEnum):
    """Blablador model names (OpenAI-compatible API).
    Only GPT-OSS-120b is offered in the UI (alias: GPT-OSS-120b).
    """

    GPT_OSS = "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"


# Display name shown in UI for Blablador models (model id -> label)
BLABLADOR_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    BlabladorModelName.GPT_OSS.value: "GPT-OSS-120b",
}


def get_blablador_model_display_name(model_id: str) -> str:
    """Return the UI display name for a Blablador model id."""
    return BLABLADOR_MODEL_DISPLAY_NAMES.get(model_id, model_id)


# Type alias for all supported models
AllModelEnum: TypeAlias = OpenAIModelName | BlabladorModelName


# Provider to model mapping
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


def is_valid_model_for_provider(provider: Provider, model: str) -> bool:
    """Check if a model is valid for a specific provider"""
    return model in get_models_for_provider(provider)


def get_provider_for_model(model: str) -> Provider | None:
    """Get the provider that supports a specific model"""
    for provider, models in PROVIDER_MODELS.items():
        if model in models:
            return provider
    return None
