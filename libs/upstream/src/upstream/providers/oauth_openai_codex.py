from __future__ import annotations

import asyncio
import time
import webbrowser
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import httpx

from upstream.oauth import OAuthTokens, generate_pkce_pair, run_local_callback_server

# Values verified against Pi's packages/ai/src/utils/oauth/openai-codex.ts. `originator` is set
# to "haiku" rather than Pi's own "pi" value -- this is our own tool, so it identifies itself
# honestly rather than impersonating Pi to OpenAI's backend.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
SCOPES = "openid profile email offline_access"
ORIGINATOR = "haiku"

_DEVICE_PENDING_CODES = {"deviceauth_authorization_pending"}
_DEVICE_SLOW_DOWN_CODES = {"deviceauth_slow_down"}


def build_authorize_url(*, code_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def login_with_browser(*, open_browser: bool = True) -> OAuthTokens:
    verifier, challenge = generate_pkce_pair()
    url = build_authorize_url(code_challenge=challenge, state=verifier)
    if open_browser:
        webbrowser.open(url)

    params = await asyncio.to_thread(
        run_local_callback_server, CALLBACK_PORT, CALLBACK_PATH, timeout=300.0
    )
    code = params.get("code")
    if not code:
        raise ValueError(f"OAuth callback did not include an authorization code: {params}")

    return await exchange_authorization_code(code, verifier, redirect_uri=REDIRECT_URI)


async def login_with_device_code(
    *, on_prompt: Callable[[str, str], None] | None = None
) -> OAuthTokens:
    async with httpx.AsyncClient() as client:
        response = await client.post(DEVICE_USER_CODE_URL, json={"client_id": CLIENT_ID})
        response.raise_for_status()
        payload = response.json()

        device_auth_id = payload["device_auth_id"]
        user_code = payload["user_code"]
        interval = float(payload.get("interval", 5))
        if on_prompt is not None:
            on_prompt(DEVICE_VERIFICATION_URI, user_code)

        while True:
            await asyncio.sleep(interval)
            poll_response = await client.post(
                DEVICE_TOKEN_URL,
                json={"device_auth_id": device_auth_id, "user_code": user_code},
            )
            if poll_response.status_code in (403, 404):
                continue
            poll_payload = poll_response.json()
            error = (
                (poll_payload.get("error") or {}).get("code")
                if "error" in poll_payload
                else None
            )
            if error in _DEVICE_PENDING_CODES:
                continue
            if error in _DEVICE_SLOW_DOWN_CODES:
                interval += 5
                continue
            if error:
                from upstream.oauth import DeviceCodeError

                raise DeviceCodeError(error)
            break

    authorization_code = poll_payload["authorization_code"]
    code_verifier = poll_payload["code_verifier"]
    return await exchange_authorization_code(
        authorization_code, code_verifier, redirect_uri=DEVICE_REDIRECT_URI
    )


async def exchange_authorization_code(
    code: str, code_verifier: str, *, redirect_uri: str
) -> OAuthTokens:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return _tokens_from_response(response.json())


async def refresh(refresh_token: str) -> OAuthTokens:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return _tokens_from_response(response.json())


def _tokens_from_response(payload: dict[str, Any]) -> OAuthTokens:
    expires_in = int(payload.get("expires_in", 3600))
    refresh_token = payload.get("refresh_token")
    return OAuthTokens(
        access_token=str(payload["access_token"]),
        refresh_token=str(refresh_token) if refresh_token else None,
        expires_at=int(time.time()) + expires_in - 300,
    )
