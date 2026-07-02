from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from coding_agent.config import ProviderConfig
from upstream.auth import AuthStorage, MemoryAuthStorage, redact_secret, resolve_config_value
from upstream.providers import OpenAICompatibleProvider


def test_auth_storage_writes_auth_file_with_user_only_permissions(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    storage = AuthStorage(path=path)

    storage.set_api_key("openai-compatible", "sk-test")

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "providers": {
            "openai-compatible": {
                "type": "api_key",
                "key": "sk-test",
            }
        }
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_auth_resolution_prefers_explicit_then_stored_then_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = AuthStorage(path=tmp_path / "auth.json")
    storage.set_api_key("openai-compatible", "stored-key")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    explicit = storage.resolve_api_key(
        "openai-compatible",
        explicit_api_key="explicit-key",
        env_var="OPENAI_API_KEY",
    )
    stored = storage.resolve_api_key("openai-compatible", env_var="OPENAI_API_KEY")
    env = AuthStorage(path=tmp_path / "missing.json").resolve_api_key(
        "openai-compatible",
        env_var="OPENAI_API_KEY",
    )

    assert explicit.key == "explicit-key"
    assert explicit.source == "explicit"
    assert stored.key == "stored-key"
    assert stored.source == "auth_file"
    assert env.key == "env-key"
    assert env.source == "env:OPENAI_API_KEY"


def test_memory_auth_storage_behaves_like_file_storage() -> None:
    storage = MemoryAuthStorage()

    storage.set_api_key("provider", "secret")

    assert storage.get_api_key("provider") == "secret"
    assert storage.list() == ["provider"]
    assert storage.remove("provider") is True
    assert storage.has_auth("provider") is False


@pytest.mark.asyncio
async def test_provider_config_uses_stored_key(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    AuthStorage(path=auth_file).set_api_key("openai-compatible", "stored-key")

    provider = await ProviderConfig(
        provider="openai-compatible",
        model="gpt-4o-mini",
        auth_file=auth_file,
    ).build()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.api_key == "stored-key"


@pytest.mark.asyncio
async def test_provider_config_explicit_key_wins_over_stored_key(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    AuthStorage(path=auth_file).set_api_key("openai-compatible", "stored-key")

    provider = await ProviderConfig(
        provider="openai-compatible",
        model="gpt-4o-mini",
        api_key="explicit-key",
        auth_file=auth_file,
    ).build()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.api_key == "explicit-key"


@pytest.mark.asyncio
async def test_missing_key_error_does_not_leak_other_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTHER_SECRET", "do-not-leak")

    with pytest.raises(ValueError) as exc_info:
        await ProviderConfig(
            provider="openai-compatible",
            model="gpt-4o-mini",
            auth_file=tmp_path / "missing.json",
        ).build()

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "do-not-leak" not in message


@pytest.mark.asyncio
async def test_get_oauth_access_token_returns_stored_token_when_not_expired(
    tmp_path: Path,
) -> None:
    from upstream.oauth import OAuthTokens

    storage = AuthStorage(path=tmp_path / "auth.json")
    storage.set_oauth_credential(
        "openai-codex",
        OAuthTokens(access_token="access-1", refresh_token="refresh-1", expires_at=9_999_999_999),
    )

    async def _refresh(refresh_token: str):
        raise AssertionError("refresh should not be called when token is not expired")

    token = await storage.get_oauth_access_token("openai-codex", refresh=_refresh)

    assert token == "access-1"


@pytest.mark.asyncio
async def test_get_oauth_access_token_refreshes_when_expired(tmp_path: Path) -> None:
    from upstream.oauth import OAuthTokens

    storage = AuthStorage(path=tmp_path / "auth.json")
    storage.set_oauth_credential(
        "openai-codex",
        OAuthTokens(access_token="stale", refresh_token="refresh-1", expires_at=1),
    )
    calls: list[str] = []

    async def _refresh(refresh_token: str) -> OAuthTokens:
        calls.append(refresh_token)
        return OAuthTokens(
            access_token="fresh", refresh_token="refresh-2", expires_at=9_999_999_999
        )

    token = await storage.get_oauth_access_token("openai-codex", refresh=_refresh)

    assert token == "fresh"
    assert calls == ["refresh-1"]
    persisted = storage.get("openai-codex")
    assert persisted.accessToken == "fresh"
    assert persisted.refreshToken == "refresh-2"


@pytest.mark.asyncio
async def test_get_oauth_access_token_returns_none_for_unknown_provider(tmp_path: Path) -> None:
    storage = AuthStorage(path=tmp_path / "auth.json")

    assert await storage.get_oauth_access_token("openai-codex") is None


def test_resolve_config_value_reads_env_and_rejects_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_KEY", "secret")

    assert resolve_config_value("$LOCAL_KEY") == "secret"
    assert resolve_config_value("${LOCAL_KEY}") == "secret"
    with pytest.raises(ValueError, match="not supported"):
        resolve_config_value("!security find-generic-password")


def test_redact_secret() -> None:
    assert redact_secret("sk-1234567890") == "sk-1...7890"
    assert redact_secret("short") == "****"
