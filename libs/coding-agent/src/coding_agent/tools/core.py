from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent.core import ToolCallContext
from upstream.models import ToolCall
from upstream.types import AgentToolResult, ToolSpec

ToolHandler = Callable[[dict[str, Any], ToolCallContext], Awaitable[tuple[AgentToolResult, bool]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    label: str | None = None

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
        try:
            return await tool.handler(call.arguments, ctx or _NOOP_CONTEXT)
        except Exception as exc:  # noqa: BLE001 - tool failures are model-visible results.
            return AgentToolResult.text(f"Tool {call.name} failed: {exc}"), True


_NOOP_CONTEXT = ToolCallContext(on_update=lambda _update: None)
