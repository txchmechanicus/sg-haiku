from __future__ import annotations

import base64
import json

import httpx
import pytest
from upstream.oauth import DeviceCodeError, decode_jwt_payload, generate_pkce_pair
from upstream.providers import oauth_openai_codex


def test_generate_pkce_pair_produces_matching_s256_challenge() -> None:
    import hashlib

    verifier, challenge = generate_pkce_pair()

    expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
    assert challenge == expected_challenge
    assert len(verifier) > 20


def test_generate_pkce_pair_is_random() -> None:
    first, _ = generate_pkce_pair()
    second, _ = generate_pkce_pair()
    assert first != second


def _fake_jwt(payload: dict[str, object]) -> str:
    def _segment(data: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode("ascii")

    return f"{_segment({'alg': 'none'})}.{_segment(payload)}.signature"


def test_decode_jwt_payload_reads_claims() -> None:
    token = _fake_jwt(
        {"sub": "user-1", "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}}
    )

    payload = decode_jwt_payload(token)

    assert payload["sub"] == "user-1"
    assert payload["https://api.openai.com/auth"]["chatgpt_account_id"] == "acct-1"


def test_decode_jwt_payload_rejects_non_jwt() -> None:
    with pytest.raises(ValueError, match="Not a JWT"):
        decode_jwt_payload("not-a-jwt")


_RealAsyncClient = httpx.AsyncClient


def _mock_client(handler):
    return _RealAsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_exchange_authorization_code_returns_tokens(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        return httpx.Response(
            200,
            json={"access_token": "access-1", "refresh_token": "refresh-1", "expires_in": 3600},
        )

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _mock_client(handler))

    tokens = await oauth_openai_codex.exchange_authorization_code(
        "code-1", "verifier-1", redirect_uri="http://localhost:1455/auth/callback"
    )

    assert tokens.access_token == "access-1"
    assert tokens.refresh_token == "refresh-1"


@pytest.mark.asyncio
async def test_refresh_returns_rotated_tokens(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        return httpx.Response(
            200,
            json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
        )

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _mock_client(handler))

    tokens = await oauth_openai_codex.refresh("refresh-1")

    assert tokens.access_token == "access-2"


@pytest.mark.asyncio
async def test_login_with_device_code_polls_until_success(monkeypatch) -> None:
    calls = {"usercode": 0, "token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/accounts/deviceauth/usercode":
            calls["usercode"] += 1
            return httpx.Response(
                200,
                json={"device_auth_id": "dev-1", "user_code": "ABCD-1234", "interval": 0},
            )
        if request.url.path == "/api/accounts/deviceauth/token":
            calls["token"] += 1
            if calls["token"] < 3:
                return httpx.Response(
                    200, json={"error": {"code": "deviceauth_authorization_pending"}}
                )
            return httpx.Response(
                200,
                json={"authorization_code": "auth-code-1", "code_verifier": "verifier-1"},
            )
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={"access_token": "access-3", "refresh_token": "refresh-3", "expires_in": 3600},
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _mock_client(handler))
    prompts: list[tuple[str, str]] = []

    tokens = await oauth_openai_codex.login_with_device_code(
        on_prompt=lambda uri, code: prompts.append((uri, code))
    )

    assert tokens.access_token == "access-3"
    assert calls["token"] == 3
    assert prompts == [(oauth_openai_codex.DEVICE_VERIFICATION_URI, "ABCD-1234")]


@pytest.mark.asyncio
async def test_login_with_device_code_raises_on_terminal_error(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/accounts/deviceauth/usercode":
            return httpx.Response(
                200, json={"device_auth_id": "dev-1", "user_code": "ABCD-1234", "interval": 0}
            )
        if request.url.path == "/api/accounts/deviceauth/token":
            return httpx.Response(200, json={"error": {"code": "deviceauth_access_denied"}})
        raise AssertionError(f"unexpected path: {request.url.path}")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _mock_client(handler))

    with pytest.raises(DeviceCodeError):
        await oauth_openai_codex.login_with_device_code()


def test_build_authorize_url_includes_expected_params() -> None:
    url = oauth_openai_codex.build_authorize_url(code_challenge="challenge-1", state="state-1")

    assert url.startswith(oauth_openai_codex.AUTHORIZE_URL)
    assert "client_id=" + oauth_openai_codex.CLIENT_ID in url
    assert "code_challenge=challenge-1" in url
    assert "originator=" + oauth_openai_codex.ORIGINATOR in url
