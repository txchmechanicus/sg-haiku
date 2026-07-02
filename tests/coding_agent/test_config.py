from __future__ import annotations

import json
from pathlib import Path

import pytest
from coding_agent.config import ProviderConfig
from upstream.auth import AuthStorage
from upstream.oauth import OAuthTokens
from upstream.providers import AnthropicMessagesProvider, OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_provider_config_build_enables_images_for_vision_capable_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ProviderConfig(model="gpt-4o-mini", auth_file=tmp_path / "auth.json")

    provider = await config.build()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.supports_images is True


@pytest.mark.asyncio
async def test_provider_config_build_disables_images_for_text_only_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    models_config = tmp_path / "models.json"
    models_config.write_text(
        json.dumps(
            {
                "providers": {
                    "openai-compatible": {
                        "modelOverrides": {"gpt-4o-mini": {"input": ["text"]}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = ProviderConfig(
        model="gpt-4o-mini",
        models_config_paths=[models_config],
        auth_file=tmp_path / "auth.json",
    )

    provider = await config.build()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.supports_images is False


@pytest.mark.asyncio
async def test_provider_config_build_resolves_anthropic_oauth(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    AuthStorage(path=auth_file).set_oauth_credential(
        "anthropic",
        OAuthTokens(
            access_token="oauth-token", refresh_token="refresh-1", expires_at=9_999_999_999
        ),
    )
    config = ProviderConfig(provider="anthropic", model="claude-sonnet-5", auth_file=auth_file)

    provider = await config.build()

    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.is_oauth is True
    assert provider.access_token == "oauth-token"


@pytest.mark.asyncio
async def test_provider_config_build_falls_back_to_api_key_for_anthropic(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")
    config = ProviderConfig(
        provider="anthropic", model="claude-sonnet-5", auth_file=tmp_path / "auth.json"
    )

    provider = await config.build()

    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.is_oauth is False
    assert provider.access_token == "sk-ant-key"


@pytest.mark.asyncio
async def test_provider_config_build_raises_without_anthropic_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = ProviderConfig(
        provider="anthropic", model="claude-sonnet-5", auth_file=tmp_path / "auth.json"
    )

    with pytest.raises(ValueError, match="Anthropic provider requires"):
        await config.build()
