"""`ExtensionAPI` — the object an extension's `activate(api)` factory receives, scoped to
the events wired in this milestone (see `coding_agent.extensions.types` module docstring and
`/PLAN.md` for what's deferred).
"""

from __future__ import annotations

from coding_agent.extensions.types import Extension, Handler, RegisteredTool, SourceInfo
from coding_agent.tools.core import Tool


class ExtensionAPI:
    """Registration-only surface available during `activate()`. There is no separate
    "action" surface (`send_message`, `set_model`, ...) that throws until bound —
    sg-haiku doesn't yet have a live session an extension could act on mid-load, since there
    is no reload/interactive multi-session concept (see `/PLAN.md`)."""

    def __init__(self, extension: Extension) -> None:
        self._extension = extension

    def on(self, event: str, handler: Handler) -> None:
        self._extension.handlers.setdefault(event, []).append(handler)

    def register_tool(self, tool: Tool) -> None:
        self._extension.tools[tool.name] = RegisteredTool(
            tool=tool,
            source_info=SourceInfo(
                path=self._extension.path, resolved_path=self._extension.resolved_path
            ),
        )
