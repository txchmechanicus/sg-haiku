"""Example haiku extension: blocks `bash` tool calls whose command contains `rm -rf`.

Install it by copying (or symlinking) this file into `.haiku/extensions/` (project-local) or
`~/.haiku/extensions/` (global) — sg-haiku discovers `*.py` files there directly, no build
step needed. See `coding_agent.extensions` for the full hook API (and `/PLAN.md`, untracked,
for what's not implemented yet).

A `tool_call` handler inspects the incoming call and returns `{block: True, reason: ...}` to
veto it.
"""

from __future__ import annotations

from agent.core import ToolCallHookResult
from coding_agent.extensions import ExtensionAPI, ExtensionContext
from upstream.models import ToolCall


async def _guard_dangerous_bash(
    call: ToolCall, ctx: ExtensionContext
) -> ToolCallHookResult | None:
    if call.name != "bash":
        return None
    command = str(call.arguments.get("command", ""))
    if "rm -rf" in command:
        return ToolCallHookResult(
            block=True,
            reason=f"Blocked by dirty_command_guard extension: refusing to run {command!r}.",
        )
    return None


def activate(api: ExtensionAPI) -> None:
    api.on("tool_call", _guard_dangerous_bash)
