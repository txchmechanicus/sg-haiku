from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from upstream.oauth import OAuthTokens

DEFAULT_AUTH_FILE = Path.home() / ".haiku" / "auth.json"


class StoredCredential(BaseModel):
    type: Literal["api_key", "oauth"]


class ApiKeyCredential(StoredCredential):
    type: Literal["api_key"] = "api_key"
    key: str


class OAuthCredential(StoredCredential):
    type: Literal["oauth"] = "oauth"
    accessToken: str | None = None
    refreshToken: str | None = None
    expiresAt: int | None = None


Credential = ApiKeyCredential | OAuthCredential


class AuthFile(BaseModel):
    providers: dict[str, Credential] = Field(default_factory=dict)


@dataclass(frozen=True)
class AuthResolution:
    key: str | None
    source: str | None


class AuthStorage:
    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or DEFAULT_AUTH_FILE
        self._runtime_api_keys: dict[str, str] = {}

    def set_runtime_api_key(self, provider: str, key: str) -> None:
        self._runtime_api_keys[provider] = key

    def get_api_key(self, provider: str) -> str | None:
        if provider in self._runtime_api_keys:
            return self._runtime_api_keys[provider]
        credential = self.get(provider)
        if isinstance(credential, ApiKeyCredential):
            return credential.key
        return None

    async def get_oauth_access_token(
        self,
        provider: str,
        *,
        refresh: Callable[[str], Awaitable[OAuthTokens]] | None = None,
    ) -> str | None:
        """Returns a valid OAuth access token, refreshing (and persisting) it if expired."""
        credential = self.get(provider)
        if not isinstance(credential, OAuthCredential):
            return None
        expired = credential.expiresAt is not None and credential.expiresAt <= int(time.time())
        if not expired or refresh is None or not credential.refreshToken:
            return credential.accessToken
        with self._locked():
            # Re-read under the lock in case another process already refreshed.
            latest = self.get(provider)
            if isinstance(latest, OAuthCredential):
                still_expired = (
                    latest.expiresAt is not None and latest.expiresAt <= int(time.time())
                )
                if not still_expired:
                    return latest.accessToken
                credential = latest
            assert credential.refreshToken is not None
            tokens = await refresh(credential.refreshToken)
            self.set_oauth_credential(provider, tokens)
            return tokens.access_token

    @contextlib.contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def get(self, provider: str) -> Credential | None:
        return self._read().providers.get(provider)

    def set_api_key(self, provider: str, key: str) -> None:
        data = self._read()
        data.providers[provider] = ApiKeyCredential(key=key)
        self._write(data)

    def set_oauth_credential(self, provider: str, tokens: OAuthTokens) -> None:
        data = self._read()
        data.providers[provider] = OAuthCredential(
            accessToken=tokens.access_token,
            refreshToken=tokens.refresh_token,
            expiresAt=tokens.expires_at,
        )
        self._write(data)

    def remove(self, provider: str) -> bool:
        data = self._read()
        existed = provider in data.providers
        data.providers.pop(provider, None)
        if existed:
            self._write(data)
        return existed

    def list(self) -> list[str]:
        return sorted(self._read().providers)

    def has_auth(self, provider: str) -> bool:
        return provider in self._runtime_api_keys or provider in self._read().providers

    def get_auth_status(self, provider: str) -> dict[str, str | bool | None]:
        if provider in self._runtime_api_keys:
            return {
                "provider": provider,
                "authenticated": True,
                "type": "api_key",
                "source": "runtime",
            }
        credential = self.get(provider)
        if credential is None:
            return {"provider": provider, "authenticated": False, "type": None, "source": None}
        return {
            "provider": provider,
            "authenticated": True,
            "type": credential.type,
            "source": "auth_file",
        }

    def resolve_api_key(
        self,
        provider: str,
        *,
        explicit_api_key: str | None = None,
        env_var: str | None = None,
    ) -> AuthResolution:
        if explicit_api_key:
            return AuthResolution(key=explicit_api_key, source="explicit")
        stored = self.get_api_key(provider)
        if stored:
            return AuthResolution(key=stored, source="auth_file")
        if env_var:
            env_value = os.getenv(env_var)
            if env_value:
                return AuthResolution(key=env_value, source=f"env:{env_var}")
        return AuthResolution(key=None, source=None)

    def _read(self) -> AuthFile:
        if not self.path.exists():
            return AuthFile()
        return AuthFile.model_validate(json.loads(self.path.read_text(encoding="utf-8")))

    def _write(self, data: AuthFile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)


class MemoryAuthStorage(AuthStorage):
    def __init__(self) -> None:
        super().__init__(path=Path(":memory:"))
        self._data = AuthFile()

    def _read(self) -> AuthFile:
        return self._data.model_copy(deep=True)

    def _write(self, data: AuthFile) -> None:
        self._data = data.model_copy(deep=True)

    @contextlib.contextmanager
    def _locked(self):
        yield


def redact_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def resolve_config_value(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("!"):
        raise ValueError("Command-based config values are not supported yet.")
    if value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        return _env_or_error(name)
    if value.startswith("$") and len(value) > 1:
        return _env_or_error(value[1:])
    return value


def _env_or_error(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set.")
    return value
