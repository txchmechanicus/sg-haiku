from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str
    refresh_token: str | None
    expires_at: int  # epoch seconds, already includes a safety buffer


class OAuthCallbackError(Exception):
    pass


class DeviceCodeError(Exception):
    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def generate_pkce_pair() -> tuple[str, str]:
    """Returns (code_verifier, code_challenge) for PKCE S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def decode_jwt_payload(token: str) -> dict[str, object]:
    """Decodes a JWT payload without verifying its signature (it's our own token)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Not a JWT (expected 3 dot-separated segments).")
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    decoded = base64.urlsafe_b64decode(payload_segment + padding)
    return json.loads(decoded)


def run_local_callback_server(port: int, path: str, *, timeout: float = 300.0) -> dict[str, str]:
    """Blocks the calling thread until a single request hits `path`, returns its query params.

    Intended to be run via `asyncio.to_thread` from async callers so it doesn't block the
    event loop.
    """
    result: dict[str, list[str]] = {}
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # silence default request logging
            pass

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name
            parsed = urlparse(self.path)
            if parsed.path == path:
                result.update(parse_qs(parsed.query))
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body>Login complete. You can close this tab.</body></html>"
                )
                received.set()
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("127.0.0.1", port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        if not received.wait(timeout):
            raise OAuthCallbackError("Timed out waiting for the OAuth browser callback.")
    finally:
        server.shutdown()
        server_thread.join(timeout=5)
    return {key: values[0] for key, values in result.items() if values}


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
