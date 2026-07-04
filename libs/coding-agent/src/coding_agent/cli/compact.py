from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from agent.compaction import compact
from agent.sessions import DEFAULT_SESSION_DIR, SessionManager, resolve_session_reference

from coding_agent.cli.console import console, error_console
from coding_agent.config import ProviderConfig


def register(app: typer.Typer) -> None:
    app.command("compact")(compact_session)


def compact_session(
    session_ref: Annotated[str, typer.Argument(help="Session file path or id.")],
    session_dir: Annotated[
        Path,
        typer.Option("--session-dir", help="Directory to search for sessions."),
    ] = DEFAULT_SESSION_DIR,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Model provider."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Model name.")
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="OpenAI-compatible API base URL."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="API key. Falls back to OPENAI_API_KEY."),
    ] = None,
    auth_file: Annotated[
        Path | None,
        typer.Option("--auth-file", help="Path to auth JSON file."),
    ] = None,
    models_config: Annotated[
        list[Path] | None,
        typer.Option("--models-config", help="Read models from this JSON file."),
    ] = None,
    compaction_keep_tokens: Annotated[
        int,
        typer.Option(
            "--compaction-keep-tokens",
            help="Approximate number of recent tokens to keep uncompacted.",
        ),
    ] = 8000,
    summary: Annotated[
        str | None,
        typer.Option(
            "--summary",
            help=(
                "Use this text as the summary instead of asking the model. "
                "Recorded with fromHook=true, since it bypasses the default LLM summarization."
            ),
        ),
    ] = None,
) -> None:
    """Manually compact a session, without running a new prompt turn."""
    if model == "mock" or provider == "mock":
        raise typer.BadParameter("mock is a dev-only provider and cannot be selected via CLI.")

    try:
        loaded = resolve_session_reference(session_ref, session_dir)
        config = ProviderConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            models_config_paths=models_config,
            auth_file=auth_file,
        )

        async def _build_and_compact():
            model_provider = await config.build()
            return await compact(
                model_provider,
                list(loaded.entry_refs),
                previous_summary=loaded.compaction_summary,
                keep_recent_tokens=compaction_keep_tokens,
                provided_summary=summary,
            )

        result = asyncio.run(_build_and_compact())
        manager = SessionManager.create(
            explicit_path=loaded.path,
            session_id=loaded.session_id,
            cwd=Path.cwd(),
            append=True,
            header=loaded.header,
            leaf_id=loaded.leaf_id,
            known_ids=loaded.entry_ids,
        )
        manager.record_compaction(
            summary=result.summary,
            first_kept_entry_id=result.first_kept_entry_id,
            tokens_before=result.tokens_before,
            details=result.details.to_json() if result.details else None,
            from_hook=result.from_hook,
        )
        console.print(result.summary)
    except ValueError as exc:
        error_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
