from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel
from upstream.models import AssistantMessageEvent, Message, ToolResultMessage


class AgentEvent(BaseModel):
    type: Literal[
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
    ]
    messages: list[Message] | None = None
    message: Message | None = None
    toolResults: list[ToolResultMessage] | None = None
    assistantMessageEvent: AssistantMessageEvent | None = None
    toolCallId: str | None = None
    toolName: str | None = None
    args: Any = None
    partialResult: Any = None
    result: Any = None
    isError: bool | None = None

    @classmethod
    def agent_start(cls) -> AgentEvent:
        return cls(type="agent_start")

    @classmethod
    def agent_end(cls, messages: list[Message]) -> AgentEvent:
        return cls(type="agent_end", messages=messages)

    @classmethod
    def turn_start(cls) -> AgentEvent:
        return cls(type="turn_start")

    @classmethod
    def turn_end(cls, message: Message, tool_results: list[ToolResultMessage]) -> AgentEvent:
        return cls(type="turn_end", message=message, toolResults=tool_results)

    @classmethod
    def message_start(cls, message: Message) -> AgentEvent:
        return cls(type="message_start", message=message)

    @classmethod
    def message_update(cls, message: Message, event: AssistantMessageEvent) -> AgentEvent:
        return cls(type="message_update", message=message, assistantMessageEvent=event)

    @classmethod
    def message_end(cls, message: Message) -> AgentEvent:
        return cls(type="message_end", message=message)

    @classmethod
    def tool_execution_start(cls, tool_call_id: str, tool_name: str, args: Any) -> AgentEvent:
        return cls(
            type="tool_execution_start",
            toolCallId=tool_call_id,
            toolName=tool_name,
            args=args,
        )

    @classmethod
    def tool_execution_end(
        cls,
        tool_call_id: str,
        tool_name: str,
        args: Any,
        result: Any,
        is_error: bool,
    ) -> AgentEvent:
        return cls(
            type="tool_execution_end",
            toolCallId=tool_call_id,
            toolName=tool_name,
            args=args,
            result=result,
            isError=is_error,
        )
