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
from agent.entries import EntryRef
from agent.events import AgentEvent
from agent.sessions import (
    DEFAULT_SESSION_DIR,
    LoadedSession,
    SessionManager,
    latest_session,
    resolve_session_reference,
)
from upstream.models import AssistantMessage, SystemMessage, TextContent
from upstream.registry import ModelRegistry

from coding_agent.cli import interactive
from coding_agent.cli.console import console, error_console
from coding_agent.cli.helpers import build_extension_runner, build_tool_registry, parse_tool_list
from coding_agent.config import ProviderConfig
from coding_agent.extensions import (
    AfterProviderResponseEvent,
    ExtensionRunner,
    ModelSelectEvent,
    SessionBeforeCompactEvent,
    SessionCompactEvent,
    SessionShutdownEvent,
    SessionStartEvent,
    ThinkingLevelSelectEvent,
)


class _HaikuGroup(typer.core.TyperGroup):
    """Converts the first non-flag, non-subcommand positional arg to --prompt."""

    def parse_args(self, ctx, args):  # type: ignore[override]
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["--prompt", args[0], *args[1:]]
        return super().parse_args(ctx, args)


app = typer.Typer(cls=_HaikuGroup, add_completion=False)

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
    no_skills: Annotated[
        bool,
        typer.Option("--no-skills", help="Do not discover or offer skills to the model."),
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
        try:
            prompt_context = PromptContextBuilder(cwd=Path.cwd()).build(
                prompt="",
                include_context_files=not no_context_files,
                include_skills=not no_skills,
                system_prompt=system_prompt,
                append_system_prompts=append_system_prompt,
                use_prompt_templates=False,
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
                interactive.run(
                    config,
                    system_prompt=prompt_context.system_prompt,
                    use_tools=not no_tools,
                    no_builtin_tools=no_builtin_tools,
                    tools=parse_tool_list(tools),
                    exclude_tools=parse_tool_list(exclude_tools),
                )
            )
        except ValueError as exc:
            error_console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(2) from exc
        return
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
                prompt,
                config,
                no_context_files=no_context_files,
                no_skills=no_skills,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                prompt_template=prompt_template,
                no_prompt_templates=no_prompt_templates,
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
                tools=parse_tool_list(tools),
                exclude_tools=parse_tool_list(exclude_tools),
                compaction_enabled=not no_compaction,
                compaction_reserve_tokens=compaction_reserve_tokens,
                compaction_keep_tokens=compaction_keep_tokens,
            )
        )
    except ValueError as exc:
        error_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


async def _run(
    prompt: str,
    config: ProviderConfig,
    *,
    no_context_files: bool,
    no_skills: bool,
    system_prompt: str | None,
    append_system_prompt: list[str] | None,
    prompt_template: Path | None,
    no_prompt_templates: bool,
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
    cwd = Path.cwd()
    registry = build_tool_registry(
        no_builtin_tools=no_builtin_tools,
        tools=tools,
        exclude_tools=exclude_tools,
    )
    initial_entries, session, compaction_summary, compaction_details = _build_session_manager(
        session_path=session_path,
        session_dir=session_dir,
        session_id=session_id,
        session_name=session_name,
        write_session=write_session,
        continue_session=continue_session,
        resume=resume,
        fork=fork,
    )

    runner = await build_extension_runner(
        cwd=cwd,
        registry=registry,
        session_manager=session,
        get_system_prompt=lambda: effective_system_prompt,
    )
    if fork is not None:
        session_reason = "fork"
    elif continue_session or resume:
        session_reason = "resume"
    else:
        session_reason = "new"
    await runner.notify("session_start", SessionStartEvent(reason=session_reason))
    resources = await runner.emit_resources_discover(cwd=cwd, reason="startup")

    prompt_context = PromptContextBuilder(cwd=cwd).build(
        prompt=prompt,
        include_context_files=not no_context_files,
        include_skills=not no_skills,
        system_prompt=system_prompt,
        append_system_prompts=append_system_prompt,
        prompt_template=prompt_template,
        use_prompt_templates=not no_prompt_templates,
        extra_skill_paths=resources.skill_paths,
        tool_prompt_snippets=registry.prompt_snippets(),
        tool_prompt_guidelines=registry.prompt_guidelines(),
    )
    prompt = prompt_context.prompt
    effective_system_prompt = prompt_context.system_prompt

    async def before_tool_call(call):  # noqa: ANN001, ANN202 - shape matches agent.core hooks
        if not runner.has_handlers("tool_call"):
            return None
        return await runner.emit_tool_call(call)

    async def after_tool_call(call, result, is_error):  # noqa: ANN001, ANN202
        if not runner.has_handlers("tool_result"):
            return None
        return await runner.emit_tool_result(call, result, is_error)

    async def before_provider_request(payload):  # noqa: ANN001, ANN202
        if not runner.has_handlers("before_provider_request"):
            return payload
        return await runner.emit_before_provider_request(payload)

    async def after_provider_response(info):  # noqa: ANN001, ANN202
        if not runner.has_handlers("after_provider_response"):
            return
        await runner.notify(
            "after_provider_response",
            AfterProviderResponseEvent(
                durationMs=info.duration_ms, status=info.status, headers=info.headers
            ),
        )

    async def before_agent_start(prompt_text, prompt_system_prompt):  # noqa: ANN001, ANN202
        if not runner.has_handlers("before_agent_start"):
            return None
        return await runner.emit_before_agent_start(prompt_text, prompt_system_prompt)

    try:
        agent = Agent(
            provider=await config.build(),
            tools=registry,
            system_prompt=effective_system_prompt,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            before_provider_request=before_provider_request,
            after_provider_response=after_provider_response,
            before_agent_start=before_agent_start,
            provide_tool_context=runner.create_context,
        )
        await _run_agent(
            agent,
            runner,
            prompt,
            session=session,
            config=config,
            mode=mode,
            thinking=thinking,
            use_tools=use_tools,
            initial_entries=initial_entries,
            compaction_summary=compaction_summary,
            compaction_details=compaction_details,
            compaction_enabled=compaction_enabled,
            compaction_reserve_tokens=compaction_reserve_tokens,
            compaction_keep_tokens=compaction_keep_tokens,
        )
    finally:
        await runner.notify("session_shutdown", SessionShutdownEvent(reason="quit"))


async def _run_agent(
    agent: Agent,
    runner: ExtensionRunner,
    prompt: str,
    *,
    session: SessionManager,
    config: ProviderConfig,
    mode: str,
    thinking: str | None,
    use_tools: bool,
    initial_entries: list[EntryRef],
    compaction_summary: str | None,
    compaction_details: dict[str, object] | None,
    compaction_enabled: bool,
    compaction_reserve_tokens: int,
    compaction_keep_tokens: int,
) -> None:
    _provider_id, _model_id = config.model_info()
    session.record_model_change(provider=_provider_id, model_id=_model_id)
    await runner.notify(
        "model_select", ModelSelectEvent(provider=_provider_id, modelId=_model_id)
    )
    if thinking is not None and thinking != "off":
        session.record_thinking_level_change(thinking_level=thinking)
        await runner.notify(
            "thinking_level_select", ThinkingLevelSelectEvent(thinkingLevel=thinking)
        )
    stream_json = mode == "json"
    if stream_json:
        print(json.dumps(session.header(), ensure_ascii=False))

    context_window = config.context_window()
    if compaction_enabled and context_window is not None:
        context_tokens = estimate_context_tokens(initial_entries)
        if should_compact(context_tokens, context_window, compaction_reserve_tokens):
            provided_summary: str | None = None
            cancel_compaction = False
            if runner.has_handlers("session_before_compact"):
                before_result = await runner.emit_session_before_compact(
                    SessionBeforeCompactEvent(
                        reason="threshold", previousSummary=compaction_summary
                    )
                )
                if before_result is not None:
                    cancel_compaction = before_result.cancel
                    provided_summary = before_result.summary

            if not cancel_compaction:
                result = await compact(
                    agent.provider,
                    initial_entries,
                    previous_summary=compaction_summary,
                    keep_recent_tokens=compaction_keep_tokens,
                    provided_summary=provided_summary,
                )
                compaction_record = session.record_compaction(
                    summary=result.summary,
                    first_kept_entry_id=result.first_kept_entry_id,
                    tokens_before=result.tokens_before,
                    details=result.details.to_json() if result.details else None,
                    from_hook=result.from_hook,
                )
                await runner.notify(
                    "session_compact",
                    SessionCompactEvent(
                        summary=result.summary,
                        firstKeptEntryId=result.first_kept_entry_id,
                        tokensBefore=result.tokens_before,
                        fromExtension=provided_summary is not None,
                    ),
                )
                cut_position = next(
                    (
                        index
                        for index, entry in enumerate(initial_entries)
                        if entry.id == result.first_kept_entry_id
                    ),
                    len(initial_entries),
                )
                initial_entries = initial_entries[cut_position:]
                compaction_summary = result.summary
                compaction_details = result.details.to_json() if result.details else None
                if stream_json:
                    print(json.dumps(compaction_record, ensure_ascii=False))

    initial_messages = [entry.message for entry in initial_entries]
    summary_message = _build_compaction_summary_message(compaction_summary, compaction_details)
    if summary_message is not None:
        initial_messages = [summary_message, *initial_messages]
    if runner.has_handlers("context"):
        initial_messages = await runner.emit_context(initial_messages)

    had_error = False
    events: list[AgentEvent] = []
    async for event in agent.run(
        prompt,
        initial_messages=initial_messages,
        system_prompt=agent.system_prompt,
        use_tools=use_tools,
        reasoning=thinking if thinking not in (None, "off") else None,
    ):
        events.append(event)
        session.record_event(event)
        if event.type == "message_end" and event.message is not None:
            if runner.has_handlers("message_end"):
                replacement = await runner.emit_message_end(event.message)
                if replacement is not None:
                    event.message = replacement
            session.record_message(event.message)
            if isinstance(event.message, AssistantMessage) and event.message.stopReason == "error":
                had_error = True
        elif runner.has_handlers(event.type):
            await runner.notify(event.type, event)

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
) -> tuple[list[EntryRef], SessionManager, str | None, dict[str, object] | None]:
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
        leaf_id=loaded.leaf_id if append and loaded else None,
        known_ids=loaded.entry_ids if append and loaded else None,
    )
    return (
        list(loaded.entry_refs) if loaded is not None else [],
        manager,
        loaded.compaction_summary if loaded is not None else None,
        loaded.compaction_details if loaded is not None else None,
    )


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


def _build_compaction_summary_message(
    summary: str | None, details: dict[str, object] | None
) -> SystemMessage | None:
    if not summary:
        return None
    parts = [f"Compacted conversation summary:\n{summary}"]
    if details:
        read_files = details.get("readFiles") or []
        modified_files = details.get("modifiedFiles") or []
        if read_files or modified_files:
            parts.append(
                "Files touched before compaction:\n"
                f"Read: {', '.join(read_files) or 'none'}\n"
                f"Modified: {', '.join(modified_files) or 'none'}"
            )
    return SystemMessage(content="\n\n".join(parts))


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
