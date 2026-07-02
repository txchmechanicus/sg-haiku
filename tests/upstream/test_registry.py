from __future__ import annotations

import json
from pathlib import Path

import pytest
from coding_agent.config import ProviderConfig
from upstream import ModelRegistry, OpenAICompatibleProvider


def write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_registry_includes_builtin_models() -> None:
    registry = ModelRegistry.builtins()

    assert registry.resolve("gpt-4o-mini").model.provider == "openai-compatible"
    assert registry.resolve("gpt-4o-mini", provider_ref="openai-compatible").model.id == (
        "gpt-4o-mini"
    )


def test_registry_resolves_prefixed_model_alias() -> None:
    registry = ModelRegistry.builtins()

    resolved = registry.resolve("openai/gpt-4o-mini")

    assert resolved.provider.id == "openai-compatible"
    assert resolved.model.id == "gpt-4o-mini"


def test_registry_reports_unknown_prefixed_provider() -> None:
    registry = ModelRegistry.builtins()

    with pytest.raises(ValueError, match="Unknown provider: missing"):
        registry.resolve("missing/model")


def test_registry_rejects_provider_model_mismatch() -> None:
    registry = ModelRegistry.builtins()

    with pytest.raises(ValueError, match="not available for provider"):
        registry.resolve("nonexistent-model", provider_ref="openai-compatible")


def test_registry_merges_global_and_project_json_with_project_winning(tmp_path: Path) -> None:
    global_config = write_json(
        tmp_path / "global.json",
        {
            "providers": {
                "local": {
                    "api": "openai-completions",
                    "baseUrl": "http://global.test/v1",
                    "models": [
                        {"id": "llama", "name": "Global Llama", "contextWindow": 1024,
                         "maxTokens": 128},
                    ],
                }
            }
        },
    )
    project_config = write_json(
        tmp_path / "project.json",
        {
            "providers": {
                "local": {
                    "api": "openai-completions",
                    "baseUrl": "http://project.test/v1",
                    "models": [
                        {"id": "llama", "name": "Project Llama", "contextWindow": 2048,
                         "maxTokens": 256},
                    ],
                }
            }
        },
    )

    registry = ModelRegistry.load([global_config, project_config])
    resolved = registry.resolve("local/llama")

    assert resolved.provider.baseUrl == "http://project.test/v1"
    assert resolved.model.name == "Project Llama"
    assert resolved.model.contextWindow == 2048
    assert resolved.model.maxTokens == 256


def test_registry_load_reads_root_models_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_json(
        tmp_path / "models.json",
        {
            "providers": {
                "local": {
                    "api": "openai-completions",
                    "baseUrl": "http://localhost:11434/v1",
                    "models": [{"id": "llama", "name": "Llama"}],
                }
            }
        },
    )

    resolved = ModelRegistry.load().resolve("local/llama")

    assert resolved.provider.id == "local"
    assert resolved.model.id == "llama"


def test_registry_applies_model_overrides(tmp_path: Path) -> None:
    config = write_json(
        tmp_path / "models.json",
        {
            "providers": {
                "openai-compatible": {
                    "modelOverrides": {
                        "gpt-4o-mini": {"name": "Mini Override", "maxTokens": 123},
                    }
                }
            }
        },
    )

    resolved = ModelRegistry.load([config]).resolve("openai-compatible/gpt-4o-mini")

    assert resolved.model.name == "Mini Override"
    assert resolved.model.maxTokens == 123


def test_provider_config_builds_custom_openai_compatible_model(tmp_path: Path) -> None:
    config = write_json(
        tmp_path / "models.json",
        {
            "providers": {
                "local": {
                    "api": "openai-completions",
                    "baseUrl": "http://localhost:11434/v1",
                    "headers": {"X-Test": "yes"},
                    "models": [{"id": "llama", "name": "Llama"}],
                }
            }
        },
    )

    provider = ProviderConfig(
        model="local/llama",
        models_config_paths=[config],
    ).build()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "llama"
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.headers == {"X-Test": "yes"}
