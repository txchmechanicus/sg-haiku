from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import typer.core
from agent import Agent
from agent.compaction import compact, estimate_context_tokens, should_compact
from agent.context import PromptContextBuilder
from agent.events import AgentEvent
from agent.sessions import (
    DEFAULT_SESSION_DIR,
    LoadedSession,
    SessionManager,
    latest_session,
    resolve_session_reference,
)
from rich.console import Console
from upstream.auth import DEFAULT_AUTH_FILE, AuthStorage, redact_secret
from upstream.models import AssistantMessage, Message, TextContent
from upstream.registry import ModelRegistry

from coding_agent.config import ProviderConfig
from coding_agent.tools import ToolRegistry, default_registry


class _HaikuGroup(typer.core.TyperGroup):
    """Converts the first non-flag, non-subcommand positional arg to --prompt."""

    def parse_args(self, ctx, args):  # type: ignore[override]
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["--prompt", args[0], *args[1:]]
        return super().parse_args(ctx, args)


app = typer.Typer(cls=_HaikuGroup, add_completion=False, no_args_is_help=True)
auth_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(auth_app, name="auth")
console = Console()
error_console = Console(stderr=True)

_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh"}


def _version_callback(value: bool) -> None:
    if value:
        import importlib.metadata

        try:
            v = importlib.metadata.version("coding-agent")
        except importlib.metadata.PackageNotFoundError:
            v = "dev"
        console.print(f"haiku {v}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Option("--prompt", hidden=True, help="Prompt to run.")
    ] = None,
    print_mode: Annotated[
        bool,
        typer.Option("--print", "-p", help="Non-interactive print mode (default, no TUI)."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            is_eager=True,
            callback=_version_callback,
            expose_value=False,
            help="Show version and exit.",
        ),
    ] = False,
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
    list_models: Annotated[
        bool,
        typer.Option("--list-models", help="List available models and exit."),
    ] = False,
    thinking: Annotated[
        str | None,
        typer.Option(
            "--thinking",
            help=f"Thinking level: {', '.join(sorted(_THINKING_LEVELS))}.",
        ),
    ] = None,
    no_tools: Annotated[
        bool, typer.Option("--no-tools", "-nt", help="Disable tool execution.")
    ] = False,
    no_builtin_tools: Annotated[
        bool,
        typer.Option("--no-builtin-tools", "-nbt", help="Disable built-in tools."),
    ] = False,
    tools: Annotated[
        str | None,
        typer.Option("--tools", "-t", help="Comma-separated allowlist of tool names."),
    ] = None,
    exclude_tools: Annotated[
        str | None,
        typer.Option("--exclude-tools", "-xt", help="Comma-separated denylist of tool names."),
    ] = None,
    no_context_files: Annotated[
        bool,
        typer.Option("--no-context-files", "-nc", help="Do not load AGENTS.md or CLAUDE.md."),
    ] = False,
    system_prompt: Annotated[
        str | None,
        typer.Option("--system-prompt", help="System prompt text or path to a prompt file."),
    ] = None,
    append_system_prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--append-system-prompt",
            help="Append system prompt text or a path to a prompt file.",
        ),
    ] = None,
    prompt_template: Annotated[
        Path | None,
        typer.Option("--prompt-template", help="Path to a prompt template file."),
    ] = None,
    no_prompt_templates: Annotated[
        bool,
        typer.Option("--no-prompt-templates", "-np", help="Ignore prompt template flags."),
    ] = False,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Output mode: text, json, or rpc."),
    ] = "text",
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Session display name."),
    ] = None,
    session: Annotated[
        Path | None,
        typer.Option("--session", help="Write session JSONL to this file."),
    ] = None,
    session_dir: Annotated[
        Path,
        typer.Option("--session-dir", help="Directory for generated and resolved sessions."),
    ] = DEFAULT_SESSION_DIR,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Use this id for the new session."),
    ] = None,
    no_session: Annotated[
        bool,
        typer.Option("--no-session", help="Do not write a session file."),
    ] = False,
    continue_session: Annotated[
        bool,
        typer.Option("--continue", "-c", help="Continue the latest session."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", "-r", help="Resume a session by file path or id."),
    ] = None,
    fork: Annotated[
        str | None,
        typer.Option("--fork", help="Create a new session from an existing session."),
    ] = None,
    no_compaction: Annotated[
        bool,
        typer.Option("--no-compaction", help="Disable automatic session compaction."),
    ] = False,
    compaction_reserve_tokens: Annotated[
        int,
        typer.Option(
            "--compaction-reserve-tokens",
            help="Trigger compaction once fewer than this many tokens remain in the "
            "context window.",
        ),
    ] = 4096,
    compaction_keep_tokens: Annotated[
        int,
        typer.Option(
            "--compaction-keep-tokens",
            help="Approximate number of recent tokens to keep uncompacted.",
        ),
    ] = 8000,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if prompt is None:
        if list_models:
            try:
                _print_models(models_config)
            except ValueError as exc:
                error_console.print(f"[red]error:[/red] {exc}")
                raise typer.Exit(2) from exc
            return
        raise typer.BadParameter("Provide a prompt as a positional argument.")
    if mode not in {"text", "json", "rpc"}:
        raise typer.BadParameter("--mode must be one of: text, json, rpc.")
    if model == "mock" or provider == "mock":
        raise typer.BadParameter("mock is a dev-only provider and cannot be selected via CLI.")
    if provider is not None and model is None:
        raise typer.BadParameter("--provider requires --model to be specified.")
    if thinking is not None and thinking not in _THINKING_LEVELS:
        raise typer.BadParameter(
            f"--thinking must be one of: {', '.join(sorted(_THINKING_LEVELS))}."
        )

    try:
        if list_models:
            _print_models(models_config)
            return
        prompt_context = PromptContextBuilder(cwd=Path.cwd()).build(
            prompt=prompt,
            include_context_files=not no_context_files,
            system_prompt=system_prompt,
            append_system_prompts=append_system_prompt,
            prompt_template=prompt_template,
            use_prompt_templates=not no_prompt_templates,
        )
        config = ProviderConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            models_config_paths=models_config,
            auth_file=auth_file,
        )
        asyncio.run(
            _run(
                prompt_context.prompt,
                config,
                system_prompt=prompt_context.system_prompt,
                use_tools=not no_tools,
                mode=mode,
                thinking=thinking,
                session_path=session,
                session_dir=session_dir,
                session_id=session_id,
                session_name=name,
                write_session=not no_session,
                continue_session=continue_session,
                resume=resume,
                fork=fork,
                no_builtin_tools=no_builtin_tools,
                tools=_parse_tool_list(tools),
                exclude_tools=_parse_tool_list(exclude_tools),
                compaction_enabled=not no_compaction,
                compaction_reserve_tokens=compaction_reserve_tokens,
                compaction_keep_tokens=compaction_keep_tokens,
            )
        )
    except ValueError as exc:
        error_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


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
    console.print(f"{provider}: {credential.type}")


@app.command("compact")
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
        model_provider = config.build()
        result = asyncio.run(
            compact(
                model_provider,
                list(loaded.messages),
                previous_summary=loaded.compaction_summary,
                keep_recent_tokens=compaction_keep_tokens,
            )
        )
        manager = SessionManager.create(
            explicit_path=loaded.path,
            session_id=loaded.session_id,
            cwd=Path.cwd(),
            append=True,
            header=loaded.header,
        )
        manager.record_compaction(
            summary=result.summary,
            first_kept_entry_id=f"entry-{result.cut_index}",
            tokens_before=result.tokens_before,
        )
        console.print(result.summary)
    except ValueError as exc:
        error_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


async def _run(
    prompt: str,
    config: ProviderConfig,
    *,
    system_prompt: str | None,
    use_tools: bool,
    mode: str,
    thinking: str | None,
    session_path: Path | None,
    session_dir: Path,
    session_id: str | None,
    session_name: str | None,
    write_session: bool,
    continue_session: bool,
    resume: str | None,
    fork: str | None,
    no_builtin_tools: bool,
    tools: set[str] | None,
    exclude_tools: set[str] | None,
    compaction_enabled: bool = True,
    compaction_reserve_tokens: int = 4096,
    compaction_keep_tokens: int = 8000,
) -> None:
    registry = _build_tool_registry(
        no_builtin_tools=no_builtin_tools,
        tools=tools,
        exclude_tools=exclude_tools,
    )
    agent = Agent(provider=config.build(), tools=registry)
    initial_messages, session, compaction_summary = _build_session_manager(
        session_path=session_path,
        session_dir=session_dir,
        session_id=session_id,
        session_name=session_name,
        write_session=write_session,
        continue_session=continue_session,
        resume=resume,
        fork=fork,
    )
    _provider_id, _model_id = config.model_info()
    session.record_model_change(provider=_provider_id, model_id=_model_id)
    if thinking is not None and thinking != "off":
        session.record_thinking_level_change(thinking_level=thinking)
    stream_json = mode == "json"
    if stream_json:
        print(json.dumps(session.header(), ensure_ascii=False))

    context_window = config.context_window()
    if compaction_enabled and context_window is not None:
        context_tokens = estimate_context_tokens(initial_messages)
        if should_compact(context_tokens, context_window, compaction_reserve_tokens):
            result = await compact(
                agent.provider,
                initial_messages,
                previous_summary=compaction_summary,
                keep_recent_tokens=compaction_keep_tokens,
            )
            compaction_record = session.record_compaction(
                summary=result.summary,
                first_kept_entry_id=f"entry-{result.cut_index}",
                tokens_before=result.tokens_before,
            )
            initial_messages = initial_messages[result.cut_index :]
            compaction_summary = result.summary
            if stream_json:
                print(json.dumps(compaction_record, ensure_ascii=False))

    effective_system_prompt = system_prompt or agent.system_prompt
    if compaction_summary:
        effective_system_prompt = (
            f"{effective_system_prompt}\n\nCompacted conversation summary:\n{compaction_summary}"
        )

    had_error = False
    events: list[AgentEvent] = []
    async for event in agent.run(
        prompt,
        initial_messages=initial_messages,
        system_prompt=effective_system_prompt,
        use_tools=use_tools,
    ):
        events.append(event)
        session.record_event(event)
        if event.type == "message_end" and event.message is not None:
            session.record_message(event.message)
            if isinstance(event.message, AssistantMessage) and event.message.stopReason == "error":
                had_error = True
        if event.type == "message_end" and isinstance(event.message, AssistantMessage):
            if mode == "text":
                text = _assistant_text(event.message)
                if text:
                    console.print(text, end="")
                if text and not text.endswith("\n"):
                    console.print()
                if event.message.stopReason == "error":
                    error_console.print(f"[red]error:[/red] {event.message.errorMessage or text}")

        if stream_json:
            print(json.dumps(event.model_dump(mode="json", exclude_none=True), ensure_ascii=False))

    if mode == "rpc":
        print(json.dumps(_json_result(events), ensure_ascii=False))

    if had_error:
        raise typer.Exit(1)


def _build_session_manager(
    *,
    session_path: Path | None,
    session_dir: Path,
    session_id: str | None,
    session_name: str | None,
    write_session: bool,
    continue_session: bool,
    resume: str | None,
    fork: str | None,
) -> tuple[list[Message], SessionManager, str | None]:
    mode_count = sum(bool(value) for value in (continue_session, resume, fork))
    if mode_count > 1:
        raise ValueError("Use only one of --continue, --resume, or --fork.")

    loaded: LoadedSession | None = None
    append = False
    parent_session: str | None = None
    header: dict[str, object] | None = None
    output_path = session_path

    if continue_session:
        loaded = latest_session(session_dir)
        if output_path is None:
            output_path = loaded.path
            append = True
            header = loaded.header
    elif resume is not None:
        loaded = resolve_session_reference(resume, session_dir)
        if output_path is None:
            output_path = loaded.path
            append = True
            header = loaded.header
    elif fork is not None:
        loaded = resolve_session_reference(fork, session_dir)
        parent_session = loaded.session_id

    if append and session_id is not None:
        raise ValueError("--session-id cannot be used when appending to an existing session.")

    manager = SessionManager.create(
        explicit_path=output_path,
        session_dir=session_dir,
        session_id=session_id or (loaded.session_id if append and loaded else None),
        session_name=session_name,
        cwd=Path.cwd(),
        write_enabled=write_session,
        parent_session=parent_session,
        append=append,
        header=header,
    )
    return (
        list(loaded.messages) if loaded is not None else [],
        manager,
        loaded.compaction_summary if loaded is not None else None,
    )


def _build_tool_registry(
    *,
    no_builtin_tools: bool,
    tools: set[str] | None,
    exclude_tools: set[str] | None,
) -> ToolRegistry:
    registry = ToolRegistry() if no_builtin_tools else default_registry(Path.cwd())
    if no_builtin_tools and (tools or exclude_tools):
        raise ValueError(
            "--tools and --exclude-tools cannot be used with --no-builtin-tools "
            "(no tools are registered to filter)."
        )
    return registry.filtered(include=tools, exclude=exclude_tools)


def _parse_tool_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {part.strip() for part in value.split(",") if part.strip()}
    return names


def _print_models(models_config: list[Path] | None) -> None:
    registry = ModelRegistry.load(models_config)
    print("provider\tid\tname\tapi\tcontextWindow\tmaxTokens")
    for model in registry.list_models():
        print(
            "\t".join(
                [
                    model.provider,
                    model.id,
                    model.name,
                    model.api,
                    str(model.contextWindow or ""),
                    str(model.maxTokens or ""),
                ]
            )
        )


def _json_result(events: list[AgentEvent]) -> dict[str, object]:
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and isinstance(event.message, AssistantMessage)
    ]
    answer = "".join(_assistant_text(message) for message in assistant_messages)
    return {
        "answer": answer,
        "tool_calls": [
            part.model_dump(mode="json", exclude_none=True)
            for event in events
            if event.type == "message_end" and isinstance(event.message, AssistantMessage)
            for part in event.message.content
            if getattr(part, "type", None) == "toolCall"
        ],
        "tool_results": [
            event.message.model_dump(mode="json", exclude_none=True)
            for event in events
            if event.type == "message_end"
            and getattr(event.message, "role", None) == "toolResult"
        ],
        "errors": [
            message.errorMessage
            for message in assistant_messages
            if message.stopReason == "error" and message.errorMessage
        ],
        "events": [event.model_dump(mode="json", exclude_none=True) for event in events],
    }


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextContent))


if __name__ == "__main__":
    app()
