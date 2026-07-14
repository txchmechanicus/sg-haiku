from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from coding_agent.cli.console import console, error_console
from coding_agent.extensions.install import (
    ExtensionInstallError,
    install_extension,
    list_installed_extensions,
    uninstall_extension,
)

extensions_app = typer.Typer(add_completion=False, no_args_is_help=True)


@extensions_app.command("install")
def extensions_install(
    source: Annotated[str, typer.Argument(help="Git URL or local directory path.")],
    name: Annotated[
        str | None, typer.Option("--name", help="Override the derived extension name.")
    ] = None,
    project: Annotated[
        bool,
        typer.Option(
            "--project", help="Install into ./.haiku/extensions instead of ~/.haiku/extensions."
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing install with the same name.")
    ] = False,
    skip_venv: Annotated[
        bool,
        typer.Option(
            "--skip-venv",
            help="Skip creating a per-extension venv even if dependencies are declared.",
        ),
    ] = False,
) -> None:
    async def _install():
        return await install_extension(
            source,
            name=name,
            project=project,
            force=force,
            cwd=Path.cwd(),
            skip_venv=skip_venv,
        )

    try:
        result = asyncio.run(_install())
    except ExtensionInstallError as exc:
        error_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc

    console.print(f"Installed extension '{result.name}' to {result.destination}")
    if result.venv_created:
        console.print(
            "Created an isolated venv for this extension's dependencies "
            "(never touches the host's own venv)."
        )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


@extensions_app.command("list")
def extensions_list() -> None:
    entries = list_installed_extensions(cwd=Path.cwd())
    if not entries:
        console.print("No extensions installed.")
        return
    for entry in entries:
        # Parens, not square brackets -- rich's Console interprets `[...]` as markup tags
        # and would otherwise silently swallow this text as an invalid style.
        venv_marker = " (venv)" if entry["has_venv"] else ""
        broken_marker = " (broken: no entry point)" if entry["entry"] is None else ""
        console.print(f"{entry['name']}\t{entry['location']}{venv_marker}{broken_marker}")


@extensions_app.command("uninstall")
def extensions_uninstall(
    name: Annotated[str, typer.Argument(help="Extension name to remove.")],
    project: Annotated[
        bool,
        typer.Option(
            "--project", help="Remove from ./.haiku/extensions instead of ~/.haiku/extensions."
        ),
    ] = False,
) -> None:
    removed = uninstall_extension(name, project=project, cwd=Path.cwd())
    if removed:
        console.print(f"Removed extension '{name}'.")
    else:
        error_console.print(f"[red]error:[/red] no extension named '{name}' found.")
        raise typer.Exit(2)
