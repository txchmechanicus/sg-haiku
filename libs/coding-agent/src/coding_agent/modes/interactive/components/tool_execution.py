from __future__ import annotations

from typing import Any

from tui import Component


class ToolExecutionComponent(Component):
    """Renders a single tool call's start/end status line."""

    def __init__(self, tool_name: str, args: Any) -> None:
        self.tool_name = tool_name
        self.args = args
        self.finished = False
        self.is_error = False

    def finish(self, *, is_error: bool) -> None:
        self.finished = True
        self.is_error = is_error
        self.invalidate()

    def render(self, width: int) -> list[str]:
        line = f"tool: {self.tool_name}({self.args})"
        if self.finished:
            status = "error" if self.is_error else "ok"
            line += f" -> {status}"
        if width > 0 and len(line) > width:
            line = line[: max(width - 1, 0)] + "…"
        return [line]
