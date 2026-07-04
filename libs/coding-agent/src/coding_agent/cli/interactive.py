from __future__ import annotations

import dataclasses
from pathlib import Path

from agent import Agent
from agent.sessions import SessionManager
from upstream.providers import ModelProvider
from upstream.registry import ModelRegistry

from coding_agent.cli.helpers import build_extension_runner, build_tool_registry
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
) -> None:
    cwd = Path.cwd()
    registry = build_tool_registry(
        no_builtin_tools=no_builtin_tools,
        tools=tools,
        exclude_tools=exclude_tools,
    )
    # Interactive mode keeps conversation state in-memory only (no `.haiku/sessions` entry
    # yet — see PLANS.md), so this is a throwaway, non-writing SessionManager purely to give
    # extensions something to observe via `ExtensionContext.session_manager`.
    ephemeral_session = SessionManager.create(cwd=cwd, write_enabled=False)
    runner = await build_extension_runner(
        cwd=cwd,
        registry=registry,
        session_manager=ephemeral_session,
        get_system_prompt=lambda: system_prompt or "",
    )
    await runner.notify("session_start", SessionStartEvent(reason="new"))

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

    async def on_model_change(new_provider_id: str, new_model_id: str) -> ModelProvider:
        new_config = dataclasses.replace(config, provider=new_provider_id, model=new_model_id)
        return await new_config.build()

    try:
        await run_interactive(
            agent,
            use_tools=use_tools,
            system_prompt=system_prompt,
            models=models,
            on_model_change=on_model_change,
            model_label=f"{provider_id}/{model_id}",
        )
    finally:
        await runner.notify("session_shutdown", SessionShutdownEvent(reason="quit"))
