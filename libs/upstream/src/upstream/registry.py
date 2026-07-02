from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_MODELS_CONFIGS = [
    Path.home() / ".haiku" / "models.json",
    Path(".haiku") / "models.json",
    Path("models.json"),
]
PROVIDER_ALIASES = {"openai": "openai-compatible"}


class ModelCost(BaseModel):
    input: float = 0
    output: float = 0
    cacheRead: float = 0
    cacheWrite: float = 0


class ModelInfo(BaseModel):
    id: str
    name: str
    api: str
    provider: str
    baseUrl: str | None = None
    reasoning: bool | None = None
    input: list[str] = Field(default_factory=lambda: ["text"])
    cost: ModelCost = Field(default_factory=ModelCost)
    contextWindow: int | None = None
    maxTokens: int | None = None


class ProviderInfo(BaseModel):
    id: str
    name: str = ""
    api: str = "openai-completions"
    baseUrl: str | None = None
    apiKeyEnv: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.id)


class ResolvedModel(BaseModel):
    provider: ProviderInfo
    model: ModelInfo


class ModelRegistry:
    def __init__(
        self,
        *,
        providers: dict[str, ProviderInfo] | None = None,
        models: dict[str, ModelInfo] | None = None,
    ) -> None:
        self.providers = providers or {}
        self.models = models or {}

    @classmethod
    def builtins(cls) -> ModelRegistry:
        registry = cls()
        registry.add_provider(
            ProviderInfo(
                id="openai-compatible",
                name="OpenAI Compatible",
                api="openai-completions",
                baseUrl="https://api.openai.com/v1",
                apiKeyEnv="OPENAI_API_KEY",
            )
        )
        for model_id, context_window, max_tokens in [
            ("gpt-4o-mini", 128000, 16384),
            ("gpt-4o", 128000, 16384),
            ("gpt-4.1-mini", 1047576, 32768),
            ("gpt-4.1", 1047576, 32768),
        ]:
            registry.add_model(
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    api="openai-completions",
                    provider="openai-compatible",
                    baseUrl="https://api.openai.com/v1",
                    input=["text", "image"],
                    contextWindow=context_window,
                    maxTokens=max_tokens,
                )
            )

        registry.add_provider(
            ProviderInfo(
                id="openai-codex",
                name="OpenAI Codex (ChatGPT OAuth)",
                api="openai-codex",
                baseUrl="https://chatgpt.com/backend-api",
            )
        )
        registry.add_model(
            ModelInfo(
                id="gpt-5.5",
                name="gpt-5.5",
                api="openai-codex",
                provider="openai-codex",
                baseUrl="https://chatgpt.com/backend-api",
                input=["text", "image"],
                contextWindow=272000,
                maxTokens=128000,
            )
        )
        return registry

    @classmethod
    def load(cls, paths: list[Path] | None = None) -> ModelRegistry:
        registry = cls.builtins()
        config_paths = paths if paths is not None else DEFAULT_MODELS_CONFIGS
        for path in config_paths:
            if path.exists():
                registry.apply_json(path)
        return registry

    def add_provider(self, provider: ProviderInfo) -> None:
        self.providers[_normalize_provider(provider.id)] = provider

    def add_model(self, model: ModelInfo) -> None:
        provider_id = _normalize_provider(model.provider)
        self.models[_model_key(provider_id, model.id)] = model.model_copy(
            update={"provider": provider_id}
        )

    def apply_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for provider_name, provider_data in (data.get("providers") or {}).items():
            provider_data = dict(provider_data)
            models = provider_data.pop("models", [])
            model_overrides = provider_data.pop("modelOverrides", {})
            provider = ProviderInfo.model_validate({"id": provider_name, **provider_data})
            self.add_provider(provider)
            for model_data in models:
                merged = {
                    "api": provider.api,
                    "provider": provider.id,
                    "baseUrl": provider.baseUrl,
                    **model_data,
                }
                self.add_model(ModelInfo.model_validate(merged))
            for model_id, override in model_overrides.items():
                existing = self.models.get(_model_key(provider.id, model_id))
                if existing is None:
                    raise ValueError(
                        f"Cannot override unknown model: {provider_name}/{model_id}"
                    )
                self.add_model(existing.model_copy(update=override))

    def list_models(self) -> list[ModelInfo]:
        return sorted(self.models.values(), key=lambda model: (model.provider, model.id))

    def resolve(
        self,
        model_ref: str,
        *,
        provider_ref: str | None = None,
    ) -> ResolvedModel:
        prefixed_provider: str | None = None
        model_id = model_ref
        if "/" in model_ref:
            prefixed_provider, model_id = _split_model_ref(model_ref)

        provider_id = _normalize_provider(provider_ref) if provider_ref else prefixed_provider
        matches = [
            model
            for model in self.models.values()
            if model.id == model_id and (provider_id is None or model.provider == provider_id)
        ]
        if not matches:
            if provider_id is not None and provider_id not in self.providers:
                raise ValueError(f"Unknown provider: {provider_id}")
            if provider_id is not None:
                raise ValueError(
                    f"Model '{model_id}' is not available for provider '{provider_id}'."
                )
            raise ValueError(f"Unknown model: {model_ref}")
        if len(matches) > 1:
            providers = ", ".join(sorted(model.provider for model in matches))
            raise ValueError(
                f"Ambiguous model '{model_id}'. Use provider/model. Matches: {providers}"
            )
        model = matches[0]
        provider = self.providers[model.provider]
        return ResolvedModel(provider=provider, model=model)

    def resolve_api_key(
        self,
        resolved: ResolvedModel,
        *,
        explicit_api_key: str | None = None,
    ) -> str | None:
        if explicit_api_key:
            return explicit_api_key
        if resolved.provider.apiKeyEnv:
            return os.getenv(resolved.provider.apiKeyEnv)
        return None


def _normalize_provider(provider: str) -> str:
    return PROVIDER_ALIASES.get(provider, provider)


def _split_model_ref(value: str) -> tuple[str, str]:
    provider, model = value.split("/", 1)
    if not provider or not model:
        raise ValueError(f"Invalid model reference: {value}")
    return _normalize_provider(provider), model


def _model_key(provider: str, model: str) -> str:
    return f"{provider}/{model}"
