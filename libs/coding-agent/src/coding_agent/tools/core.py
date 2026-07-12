from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from agent.core import ToolCallContext
from upstream.models import ToolCall
from upstream.types import AgentToolResult, ToolSpec

ToolHandler = Callable[[dict[str, Any], ToolCallContext], Awaitable[tuple[AgentToolResult, bool]]]
PrepareArguments = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    label: str | None = None
    execution_mode: Literal["sequential", "parallel"] | None = None
    """Per-tool override of `Agent.tool_execution_mode`. `None` means the agent's global
    default applies."""
    prepare_arguments: PrepareArguments | None = None
    """Optional shim run on the raw tool-call arguments before `handler` — e.g. coercing
    loosely-typed values a model produced into the shape `handler` expects. A raised
    exception is reported as a failed tool result, like a `handler` exception."""
    prompt_snippet: str | None = None
    """Optional one-line description surfaced in the system prompt's tools block. Omitted
    entirely (not just left blank) for tools that don't set this — built-in tools already
    describe themselves via their `ToolSpec`, so this is opt-in and mainly useful for
    extension-registered tools that want to call out something in prose."""
    prompt_guidelines: tuple[str, ...] = ()
    """Optional guideline bullets surfaced in the system prompt's guidelines block when this
    tool is registered."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def execution_mode_for(self, name: str) -> Literal["sequential", "parallel"] | None:
        tool = self._tools.get(name)
        return tool.execution_mode if tool is not None else None

    def prompt_snippets(self) -> list[str]:
        return [
            tool.prompt_snippet for tool in self._tools.values() if tool.prompt_snippet is not None
        ]

    def prompt_guidelines(self) -> list[str]:
        return [
            guideline
            for tool in self._tools.values()
            for guideline in tool.prompt_guidelines
        ]

    def filtered(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> ToolRegistry:
        include = include or set(self._tools)
        exclude = exclude or set()
        unknown = (include | exclude) - set(self._tools)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown tool name: {names}")

        registry = ToolRegistry()
        for name, tool in self._tools.items():
            if name in include and name not in exclude:
                registry.register(tool)
        return registry

    async def run(
        self, call: ToolCall, ctx: ToolCallContext | None = None
    ) -> tuple[AgentToolResult, bool]:
        tool = self._tools.get(call.name)
        if tool is None:
            return AgentToolResult.text(f"Unknown tool: {call.name}"), True
        arguments = call.arguments
        if tool.prepare_arguments is not None:
            try:
                arguments = tool.prepare_arguments(arguments)
            except Exception as exc:  # noqa: BLE001 - tool failures are model-visible results.
                message = f"Tool {call.name} argument preparation failed: {exc}"
                return AgentToolResult.text(message), True
        try:
            return await tool.handler(arguments, ctx or _NOOP_CONTEXT)
        except Exception as exc:  # noqa: BLE001 - tool failures are model-visible results.
            return AgentToolResult.text(f"Tool {call.name} failed: {exc}"), True


_NOOP_CONTEXT = ToolCallContext(on_update=lambda _update: None)
