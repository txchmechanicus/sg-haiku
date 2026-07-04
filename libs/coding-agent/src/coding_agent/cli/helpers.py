from __future__ import annotations

from pathlib import Path

from coding_agent.tools import ToolRegistry, default_registry


def build_tool_registry(
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


def parse_tool_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}
