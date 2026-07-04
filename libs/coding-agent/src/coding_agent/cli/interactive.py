from __future__ import annotations

from agent import Agent

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
    await run_interactive(
        agent,
        use_tools=use_tools,
        system_prompt=system_prompt,
    )
