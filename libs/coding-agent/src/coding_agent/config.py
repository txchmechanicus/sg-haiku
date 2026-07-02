from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from upstream.auth import AuthStorage, resolve_config_value
from upstream.providers import MockProvider, ModelProvider, OpenAICompatibleProvider
from upstream.registry import ModelRegistry


@dataclass(frozen=True)
class ProviderConfig:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    models_config_paths: list[Path] | None = None
    auth_file: Path | None = None

    def build(self) -> ModelProvider:
        if self.model is None:
            return MockProvider()
        registry = ModelRegistry.load(self.models_config_paths)
        resolved = registry.resolve(self.model, provider_ref=self.provider)

        if resolved.provider.api == "openai-completions":
            auth = AuthStorage(path=self.auth_file)
            explicit_api_key = resolve_config_value(self.api_key)
            resolution = auth.resolve_api_key(
                resolved.provider.id,
                explicit_api_key=explicit_api_key,
                env_var=resolved.provider.apiKeyEnv,
            )
            api_key = resolution.key
            if not api_key and resolved.provider.apiKeyEnv:
                raise ValueError(
                    f"OpenAI-compatible provider requires --api-key or "
                    f"{resolved.provider.apiKeyEnv}. You can also run "
                    f"`haiku auth set {resolved.provider.id} --api-key ...`."
                )
            return OpenAICompatibleProvider(
                model=resolved.model.id,
                api_key=api_key,
                base_url=self.base_url or resolved.model.baseUrl or resolved.provider.baseUrl or "",
                headers=resolved.provider.headers,
            )
        raise ValueError(
            f"Unsupported provider api: {resolved.provider.api!r}. "
            f"Supported: openai-completions."
        )

    def model_info(self) -> tuple[str, str]:
        """Returns (provider_id, model_id) for session recording."""
        if self.model is None:
            return "mock", "mock"
        registry = ModelRegistry.load(self.models_config_paths)
        resolved = registry.resolve(self.model, provider_ref=self.provider)
        return resolved.provider.id, resolved.model.id

    def context_window(self) -> int | None:
        """Returns the resolved model's context window, or None for the mock provider."""
        if self.model is None:
            return None
        registry = ModelRegistry.load(self.models_config_paths)
        resolved = registry.resolve(self.model, provider_ref=self.provider)
        return resolved.model.contextWindow
