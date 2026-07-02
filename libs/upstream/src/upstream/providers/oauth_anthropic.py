from __future__ import annotations

import asyncio
import time
import webbrowser
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from upstream.oauth import OAuthTokens, generate_pkce_pair, run_local_callback_server

# Values verified against Pi's packages/ai/src/utils/oauth/anthropic.ts.
#
# NOTE ON IDENTITY: unlike the OpenAI Codex provider (which self-identifies honestly as
# "haiku"), Anthropic's OAuth-gated endpoint requires literally presenting as Claude Code to
# accept requests at all -- see anthropic_messages.py for the forced system-prompt prefix and
# "claude-cli" user-agent. This is a deliberate, user-confirmed exception (2026-07-02): the user
# authenticates with their own Anthropic account/subscription, the same way other open-source
# tools (including Pi itself) do.
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = (
    "org:create_api_key user:profile user:inference user:sessions:claude_code "
    "user:mcp_servers user:file_upload"
)


class OAuthStateMismatchError(Exception):
    pass


def build_authorize_url(*, code_challenge: str, state: str) -> str:
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
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
    state = params.get("state", verifier)
    if state != verifier:
        raise OAuthStateMismatchError("OAuth state mismatch.")

    return await exchange_authorization_code(code, verifier, state=state)


def start_manual_login() -> tuple[str, str]:
    """Returns (authorize_url, verifier) -- print the URL for the user to open by hand, then
    call login_with_manual_code(pasted, verifier=verifier) once they paste back the result."""
    verifier, challenge = generate_pkce_pair()
    return build_authorize_url(code_challenge=challenge, state=verifier), verifier


async def login_with_manual_code(pasted: str, *, verifier: str) -> OAuthTokens:
    """Parses a manually pasted redirect URL, `code#state` pair, querystring, or bare code."""
    code, state = _parse_authorization_input(pasted)
    if state is not None and state != verifier:
        raise OAuthStateMismatchError("OAuth state mismatch.")
    return await exchange_authorization_code(code, verifier, state=state or verifier)


def _parse_authorization_input(pasted: str) -> tuple[str, str | None]:
    text = pasted.strip()
    if text.startswith("http://") or text.startswith("https://"):
        query = parse_qs(urlparse(text).query)
        code = query.get("code", [None])[0]
        state = query.get("state", [None])[0]
        if code:
            return code, state
    if "#" in text:
        code, _, state = text.partition("#")
        return code, state or None
    if "code=" in text:
        query = parse_qs(text)
        code = query.get("code", [None])[0]
        state = query.get("state", [None])[0]
        if code:
            return code, state
    return text, None


async def exchange_authorization_code(code: str, code_verifier: str, *, state: str) -> OAuthTokens:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": code_verifier,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return _tokens_from_response(response.json())


async def refresh(refresh_token: str) -> OAuthTokens:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
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
