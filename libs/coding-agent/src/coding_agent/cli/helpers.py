from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from upstream.models import AssistantMessage, SystemMessage, TextContent

from coding_agent.cli.console import error_console
from coding_agent.extensions import ExtensionRunner, discover_and_load_extensions
from coding_agent.extensions.types import SessionManagerProtocol
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


async def build_extension_runner(
    *,
    cwd: Path,
    registry: ToolRegistry,
    session_manager: SessionManagerProtocol,
    get_system_prompt: Callable[[], str] = lambda: "",
) -> ExtensionRunner:
    """Discover `.haiku/extensions/` (project + global), load them, and register any
    extension-contributed tools into `registry`. Load errors are reported (per-extension
    isolation) but never abort startup."""
    result = await discover_and_load_extensions(None, cwd)
    for load_error in result.errors:
        error_console.print(
            f"[yellow]extension warning:[/yellow] {load_error.path}: {load_error.error}"
        )
    runner = ExtensionRunner(
        result.extensions,
        cwd=cwd,
        session_manager=session_manager,
        get_system_prompt=get_system_prompt,
    )
    runner.register_tools(registry)
    return runner


def parse_tool_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def assistant_text(message: AssistantMessage) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextContent))


def build_compaction_summary_message(
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
