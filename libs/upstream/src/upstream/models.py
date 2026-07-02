from __future__ import annotations

from time import time
from typing import Any, Literal

from pydantic import BaseModel, Field

Api = str
ProviderId = str
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


def timestamp_ms() -> int:
    return int(time() * 1000)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    textSignature: str | dict[str, Any] | None = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinkingSignature: str | None = None
    redacted: bool | None = None


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str
    mimeType: str


class ToolCall(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    thoughtSignature: str | None = None


class UsageCost(BaseModel):
    input: float = 0
    output: float = 0
    cacheRead: float = 0
    cacheWrite: float = 0
    total: float = 0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cacheRead: int = 0
    cacheWrite: int = 0
    cacheWrite1h: int | None = None
    reasoning: int | None = None
    totalTokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


class SystemMessage(BaseModel):
    """Non-provider-authored context injected into conversation history (e.g. a compaction
    summary), rather than the base system prompt. Rendered as a `system` role message by
    providers that support it."""

    role: Literal["system"] = "system"
    content: str
    timestamp: int = Field(default_factory=timestamp_ms)


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: int = Field(default_factory=timestamp_ms)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall] = Field(default_factory=list)
    api: Api = "openai-completions"
    provider: ProviderId = "mock"
    model: str = "mock"
    responseModel: str | None = None
    responseId: str | None = None
    diagnostics: list[dict[str, Any]] | None = None
    usage: Usage = Field(default_factory=Usage)
    stopReason: StopReason = "stop"
    errorMessage: str | None = None
    timestamp: int = Field(default_factory=timestamp_ms)


class ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    toolCallId: str
    toolName: str
    content: list[TextContent | ImageContent]
    details: Any = None
    isError: bool
    timestamp: int = Field(default_factory=timestamp_ms)


Message = UserMessage | AssistantMessage | ToolResultMessage | SystemMessage


class AssistantMessageEvent(BaseModel):
    type: Literal[
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
        "error",
    ]
    partial: AssistantMessage | None = None
    contentIndex: int | None = None
    delta: str | None = None
    content: str | None = None
    toolCall: ToolCall | None = None
    reason: StopReason | None = None
    message: AssistantMessage | None = None
    error: AssistantMessage | None = None


class AssistantResponse(BaseModel):
    message: AssistantMessage

    @property
    def content(self) -> str:
        return "".join(
            part.text for part in self.message.content if isinstance(part, TextContent)
        )

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [part for part in self.message.content if isinstance(part, ToolCall)]
