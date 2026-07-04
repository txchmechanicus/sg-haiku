from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from upstream.auth import DEFAULT_AUTH_FILE, AuthStorage, redact_secret
from upstream.providers import oauth_anthropic, oauth_openai_codex

from coding_agent.cli.console import console, error_console

auth_app = typer.Typer(add_completion=False, no_args_is_help=True)

_OAUTH_PROVIDERS = {"openai-codex": oauth_openai_codex, "anthropic": oauth_anthropic}


@auth_app.command("set")
def auth_set(
    provider: Annotated[str, typer.Argument(help="Provider id.")],
    api_key: Annotated[str, typer.Option("--api-key", help="API key to store.")],
    auth_file: Annotated[
        Path,
        typer.Option("--auth-file", help="Path to auth JSON file."),
    ] = DEFAULT_AUTH_FILE,
) -> None:
    AuthStorage(path=auth_file).set_api_key(provider, api_key)
    console.print(f"Stored API key for {provider}.")


@auth_app.command("unset")
def auth_unset(
    provider: Annotated[str, typer.Argument(help="Provider id.")],
    auth_file: Annotated[
        Path,
        typer.Option("--auth-file", help="Path to auth JSON file."),
    ] = DEFAULT_AUTH_FILE,
) -> None:
    removed = AuthStorage(path=auth_file).remove(provider)
    console.print(f"Removed auth for {provider}." if removed else f"No auth stored for {provider}.")


@auth_app.command("list")
def auth_list(
    auth_file: Annotated[
        Path,
        typer.Option("--auth-file", help="Path to auth JSON file."),
    ] = DEFAULT_AUTH_FILE,
) -> None:
    storage = AuthStorage(path=auth_file)
    providers = storage.list()
    if not providers:
        console.print("No auth entries.")
        return
    for provider in providers:
        status = storage.get_auth_status(provider)
        console.print(f"{provider}\t{status['type']}\t{status['source']}")


@auth_app.command("status")
def auth_status(
    provider: Annotated[str, typer.Argument(help="Provider id.")],
    auth_file: Annotated[
        Path,
        typer.Option("--auth-file", help="Path to auth JSON file."),
    ] = DEFAULT_AUTH_FILE,
) -> None:
    storage = AuthStorage(path=auth_file)
    credential = storage.get(provider)
    if credential is None:
        console.print(f"{provider}: not authenticated")
        return
    if getattr(credential, "type", None) == "api_key":
        console.print(f"{provider}: api_key {redact_secret(getattr(credential, 'key', None))}")
        return
    if getattr(credential, "type", None) == "oauth":
        expires_at = getattr(credential, "expiresAt", None)
        console.print(f"{provider}: oauth (expires at {expires_at})")
        return
    console.print(f"{provider}: {credential.type}")


@auth_app.command("login")
def auth_login(
    provider: Annotated[str, typer.Argument(help="Provider id (e.g. openai-codex, anthropic).")],
    device_code: Annotated[
        bool,
        typer.Option("--device-code", help="Use device-code login (for headless sessions)."),
    ] = False,
    manual_code: Annotated[
        bool,
        typer.Option(
            "--manual-code",
            help="Print the login URL and paste back the code (for headless sessions).",
        ),
    ] = False,
    auth_file: Annotated[
        Path,
        typer.Option("--auth-file", help="Path to auth JSON file."),
    ] = DEFAULT_AUTH_FILE,
) -> None:
    """Log in to a provider via OAuth."""
    module = _OAUTH_PROVIDERS.get(provider)
    if module is None:
        supported = ", ".join(sorted(_OAUTH_PROVIDERS))
        raise typer.BadParameter(
            f"OAuth login is not supported for provider: {provider!r}. Supported: {supported}."
        )
    if device_code and not hasattr(module, "login_with_device_code"):
        raise typer.BadParameter(f"{provider!r} does not support --device-code login.")
    if manual_code and not hasattr(module, "start_manual_login"):
        raise typer.BadParameter(f"{provider!r} does not support --manual-code login.")

    async def _login():
        if device_code:
            def _on_prompt(verification_uri: str, user_code: str) -> None:
                console.print(f"Go to {verification_uri} and enter code: {user_code}")

            return await module.login_with_device_code(on_prompt=_on_prompt)
        if manual_code:
            url, verifier = module.start_manual_login()
            console.print(f"Open this URL to log in:\n{url}\n")
            pasted = console.input("Paste the redirect URL or code here: ")
            return await module.login_with_manual_code(pasted, verifier=verifier)
        console.print("Opening your browser to log in...")
        return await module.login_with_browser()

    try:
        tokens = asyncio.run(_login())
    except Exception as exc:
        error_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc

    AuthStorage(path=auth_file).set_oauth_credential(provider, tokens)
    console.print(f"Logged in to {provider}.")
