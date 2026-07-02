from __future__ import annotations

import httpx
import pytest
from upstream.providers import oauth_anthropic


def test_build_authorize_url_uses_verifier_as_state() -> None:
    url = oauth_anthropic.build_authorize_url(code_challenge="challenge-1", state="verifier-1")

    assert url.startswith(oauth_anthropic.AUTHORIZE_URL)
    assert "client_id=" + oauth_anthropic.CLIENT_ID in url
    assert "code_challenge=challenge-1" in url
    assert "state=verifier-1" in url
    assert "code_challenge_method=S256" in url


def test_start_manual_login_returns_url_and_matching_verifier() -> None:
    url, verifier = oauth_anthropic.start_manual_login()

    assert f"state={verifier}" in url
    assert "code_challenge=" in url
    assert url.startswith(oauth_anthropic.AUTHORIZE_URL)


@pytest.mark.parametrize(
    ("pasted", "expected_code", "expected_state"),
    [
        ("http://localhost:53692/callback?code=abc123&state=verifier-1", "abc123", "verifier-1"),
        ("abc123#verifier-1", "abc123", "verifier-1"),
        ("code=abc123&state=verifier-1", "abc123", "verifier-1"),
        ("abc123", "abc123", None),
    ],
)
def test_parse_authorization_input_handles_all_formats(
    pasted: str, expected_code: str, expected_state: str | None
) -> None:
    code, state = oauth_anthropic._parse_authorization_input(pasted)

    assert code == expected_code
    assert state == expected_state


_RealAsyncClient = httpx.AsyncClient


def _mock_client(handler):
    return _RealAsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_login_with_manual_code_exchanges_tokens(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/oauth/token"
        return httpx.Response(
            200,
            json={"access_token": "access-1", "refresh_token": "refresh-1", "expires_in": 3600},
        )

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _mock_client(handler))

    tokens = await oauth_anthropic.login_with_manual_code(
        "code-1#verifier-1", verifier="verifier-1"
    )

    assert tokens.access_token == "access-1"
    assert tokens.refresh_token == "refresh-1"


@pytest.mark.asyncio
async def test_login_with_manual_code_rejects_state_mismatch() -> None:
    with pytest.raises(oauth_anthropic.OAuthStateMismatchError):
        await oauth_anthropic.login_with_manual_code("code-1#wrong-state", verifier="verifier-1")


@pytest.mark.asyncio
async def test_refresh_returns_rotated_tokens(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/oauth/token"
        body = request.content
        assert b"refresh_token" in body
        return httpx.Response(
            200,
            json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
        )

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _mock_client(handler))

    tokens = await oauth_anthropic.refresh("refresh-1")

    assert tokens.access_token == "access-2"
