from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from upstream.auth import AuthStorage, resolve_config_value
from upstream.providers import (
    AnthropicMessagesProvider,
    MockProvider,
    ModelProvider,
    OpenAICodexProvider,
    OpenAICompatibleProvider,
    oauth_anthropic,
    oauth_openai_codex,
)
from upstream.providers.anthropic_messages import DEFAULT_BASE_URL as ANTHROPIC_DEFAULT_BASE_URL
from upstream.providers.anthropic_messages import DEFAULT_MAX_TOKENS as ANTHROPIC_DEFAULT_MAX_TOKENS
from upstream.providers.openai_codex import DEFAULT_BASE_URL as CODEX_DEFAULT_BASE_URL
from upstream.registry import ModelRegistry


@dataclass(frozen=True)
class ProviderConfig:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    models_config_paths: list[Path] | None = None
    auth_file: Path | None = None

    async def build(self) -> ModelProvider:
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
                supports_images="image" in resolved.model.input,
            )
        if resolved.provider.api == "openai-codex":
            auth = AuthStorage(path=self.auth_file)
            access_token = await auth.get_oauth_access_token(
                resolved.provider.id, refresh=oauth_openai_codex.refresh
            )
            if not access_token:
                raise ValueError(
                    f"No OAuth login found for {resolved.provider.id}. "
                    f"Run `haiku auth login {resolved.provider.id}` first."
                )
            return OpenAICodexProvider(
                model=resolved.model.id,
                access_token=access_token,
                base_url=(
                    self.base_url
                    or resolved.model.baseUrl
                    or resolved.provider.baseUrl
                    or CODEX_DEFAULT_BASE_URL
                ),
            )
        if resolved.provider.api == "anthropic-messages":
            auth = AuthStorage(path=self.auth_file)
            explicit_api_key = resolve_config_value(self.api_key)
            oauth_token = None
            if not explicit_api_key:
                oauth_token = await auth.get_oauth_access_token(
                    resolved.provider.id, refresh=oauth_anthropic.refresh
                )
            base_url = (
                self.base_url
                or resolved.model.baseUrl
                or resolved.provider.baseUrl
                or ANTHROPIC_DEFAULT_BASE_URL
            )
            max_tokens = resolved.model.maxTokens or ANTHROPIC_DEFAULT_MAX_TOKENS
            if oauth_token:
                return AnthropicMessagesProvider(
                    model=resolved.model.id,
                    access_token=oauth_token,
                    is_oauth=True,
                    base_url=base_url,
                    max_tokens=max_tokens,
                )
            resolution = auth.resolve_api_key(
                resolved.provider.id,
                explicit_api_key=explicit_api_key,
                env_var=resolved.provider.apiKeyEnv,
            )
            api_key = resolution.key
            if not api_key:
                raise ValueError(
                    f"Anthropic provider requires --api-key, {resolved.provider.apiKeyEnv}, "
                    f"or `haiku auth login {resolved.provider.id}`."
                )
            return AnthropicMessagesProvider(
                model=resolved.model.id,
                access_token=api_key,
                is_oauth=False,
                base_url=base_url,
                max_tokens=max_tokens,
            )
        raise ValueError(
            f"Unsupported provider api: {resolved.provider.api!r}. "
            f"Supported: openai-completions, openai-codex, anthropic-messages."
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
