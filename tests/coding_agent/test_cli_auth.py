from __future__ import annotations

import json
from pathlib import Path

from coding_agent.cli import app
from typer.testing import CliRunner
from upstream.oauth import OAuthTokens
from upstream.providers import oauth_openai_codex

runner = CliRunner()


def _read_auth_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_auth_login_browser_writes_oauth_credential(tmp_path: Path, monkeypatch) -> None:
    auth_file = tmp_path / "auth.json"

    async def fake_login_with_browser(**kwargs) -> OAuthTokens:
        return OAuthTokens(access_token="access-1", refresh_token="refresh-1", expires_at=123)

    monkeypatch.setattr(oauth_openai_codex, "login_with_browser", fake_login_with_browser)

    result = runner.invoke(
        app, ["auth", "login", "openai-codex", "--auth-file", str(auth_file)]
    )

    assert result.exit_code == 0
    assert "Logged in to openai-codex" in result.stdout
    data = _read_auth_file(auth_file)
    assert data["providers"]["openai-codex"] == {
        "type": "oauth",
        "accessToken": "access-1",
        "refreshToken": "refresh-1",
        "expiresAt": 123,
    }


def test_cli_auth_login_device_code_writes_oauth_credential(tmp_path: Path, monkeypatch) -> None:
    auth_file = tmp_path / "auth.json"

    async def fake_login_with_device_code(*, on_prompt=None) -> OAuthTokens:
        if on_prompt is not None:
            on_prompt("https://auth.openai.com/codex/device", "ABCD-1234")
        return OAuthTokens(access_token="access-2", refresh_token="refresh-2", expires_at=456)

    monkeypatch.setattr(
        oauth_openai_codex, "login_with_device_code", fake_login_with_device_code
    )

    result = runner.invoke(
        app,
        ["auth", "login", "openai-codex", "--device-code", "--auth-file", str(auth_file)],
    )

    assert result.exit_code == 0
    assert "ABCD-1234" in result.stdout
    data = _read_auth_file(auth_file)
    assert data["providers"]["openai-codex"]["accessToken"] == "access-2"


def test_cli_auth_login_rejects_unsupported_provider(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["auth", "login", "openai-compatible", "--auth-file", str(tmp_path / "auth.json")]
    )

    assert result.exit_code != 0
    assert "not supported" in result.stdout or "not supported" in (result.stderr or "")


def test_cli_auth_login_reports_errors(tmp_path: Path, monkeypatch) -> None:
    auth_file = tmp_path / "auth.json"

    async def fake_login_with_browser(**kwargs):
        raise ValueError("login failed")

    monkeypatch.setattr(oauth_openai_codex, "login_with_browser", fake_login_with_browser)

    result = runner.invoke(
        app, ["auth", "login", "openai-codex", "--auth-file", str(auth_file)]
    )

    assert result.exit_code == 2
    assert not auth_file.exists()


def test_cli_auth_status_shows_oauth_expiry(tmp_path: Path) -> None:
    from upstream.auth import AuthStorage

    auth_file = tmp_path / "auth.json"
    AuthStorage(path=auth_file).set_oauth_credential(
        "openai-codex",
        OAuthTokens(access_token="access-1", refresh_token="refresh-1", expires_at=999),
    )

    result = runner.invoke(
        app, ["auth", "status", "openai-codex", "--auth-file", str(auth_file)]
    )

    assert result.exit_code == 0
    assert "oauth" in result.stdout
    assert "999" in result.stdout
    assert "access-1" not in result.stdout
