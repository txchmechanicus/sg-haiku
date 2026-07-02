from __future__ import annotations

import json
from pathlib import Path

import pytest
from coding_agent.config import ProviderConfig
from upstream.providers import OpenAICompatibleProvider


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
