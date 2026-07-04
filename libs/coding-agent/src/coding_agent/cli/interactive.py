from __future__ import annotations

import dataclasses

from agent import Agent
from upstream.providers import ModelProvider
from upstream.registry import ModelRegistry

from coding_agent.cli.helpers import build_tool_registry
from coding_agent.config import ProviderConfig
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
    registry = build_tool_registry(
        no_builtin_tools=no_builtin_tools,
        tools=tools,
        exclude_tools=exclude_tools,
    )
    agent = Agent(provider=await config.build(), tools=registry)
    models = ModelRegistry.load(config.models_config_paths).list_models()
    provider_id, model_id = config.model_info()

    async def on_model_change(new_provider_id: str, new_model_id: str) -> ModelProvider:
        new_config = dataclasses.replace(config, provider=new_provider_id, model=new_model_id)
        return await new_config.build()

    await run_interactive(
        agent,
        use_tools=use_tools,
        system_prompt=system_prompt,
        models=models,
        on_model_change=on_model_change,
        model_label=f"{provider_id}/{model_id}",
    )
