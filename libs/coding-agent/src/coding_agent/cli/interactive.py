from __future__ import annotations

import dataclasses
from pathlib import Path

from agent import Agent
from agent.entries import EntryRef
from agent.sessions import SessionManager
from upstream.providers import ModelProvider
from upstream.registry import ModelRegistry

from coding_agent.cli.helpers import (
    build_compaction_summary_message,
    build_extension_runner,
    build_tool_registry,
)
from coding_agent.config import ProviderConfig
from coding_agent.extensions import SessionShutdownEvent, SessionStartEvent
from coding_agent.modes.interactive import run_interactive


async def run(
    config: ProviderConfig,
    *,
    system_prompt: str | None,
    use_tools: bool,
    no_builtin_tools: bool,
    tools: set[str] | None,
    exclude_tools: set[str] | None,
    initial_entries: list[EntryRef],
    session: SessionManager,
    session_reason: str,
    compaction_summary: str | None,
    compaction_details: dict[str, object] | None,
    session_dir: Path,
    write_session: bool,
) -> None:
    cwd = Path.cwd()
    registry = build_tool_registry(
        no_builtin_tools=no_builtin_tools,
        tools=tools,
        exclude_tools=exclude_tools,
    )
    runner = await build_extension_runner(
        cwd=cwd,
        registry=registry,
        session_manager=session,
        get_system_prompt=lambda: system_prompt or "",
    )
    await runner.notify("session_start", SessionStartEvent(reason=session_reason))

    async def before_tool_call(call):  # noqa: ANN001, ANN202
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

    async def before_agent_start(prompt_text, prompt_system_prompt):  # noqa: ANN001, ANN202
        if not runner.has_handlers("before_agent_start"):
            return None
        return await runner.emit_before_agent_start(prompt_text, prompt_system_prompt)

    agent = Agent(
        provider=await config.build(),
        tools=registry,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        before_provider_request=before_provider_request,
        before_agent_start=before_agent_start,
    )
    models = ModelRegistry.load(config.models_config_paths).list_models()
    provider_id, model_id = config.model_info()
    session.record_model_change(provider=provider_id, model_id=model_id)

    async def on_model_change(new_provider_id: str, new_model_id: str) -> ModelProvider:
        new_config = dataclasses.replace(config, provider=new_provider_id, model=new_model_id)
        return await new_config.build()

    def new_session_factory() -> SessionManager:
        return SessionManager.create(cwd=cwd, session_dir=session_dir, write_enabled=write_session)

    compaction_message = build_compaction_summary_message(compaction_summary, compaction_details)

    try:
        await run_interactive(
            agent,
            use_tools=use_tools,
            system_prompt=system_prompt,
            models=models,
            on_model_change=on_model_change,
            model_label=f"{provider_id}/{model_id}",
            initial_entries=initial_entries,
            compaction_message=compaction_message,
            session=session,
            new_session_factory=new_session_factory,
        )
    finally:
        await runner.notify("session_shutdown", SessionShutdownEvent(reason="quit"))
