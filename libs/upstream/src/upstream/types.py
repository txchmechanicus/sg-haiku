from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from upstream.models import ImageContent, TextContent


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AgentToolResult(BaseModel):
    content: list[TextContent | ImageContent]
    details: Any = None
    terminate: bool | None = None

    @classmethod
    def text(
        cls,
        text: str,
        *,
        details: Any = None,
        terminate: bool | None = None,
    ) -> AgentToolResult:
        return cls(content=[TextContent(text=text)], details=details, terminate=terminate)
