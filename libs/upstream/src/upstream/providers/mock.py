from __future__ import annotations

from collections.abc import AsyncIterator

from upstream.models import (
    AssistantMessage,
    AssistantMessageEvent,
    Message,
    TextContent,
    ToolCall,
)
from upstream.providers.base import ModelProvider
from upstream.types import ToolSpec


class MockProvider(ModelProvider):
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        last = messages[-1]
        if last.role == "toolResult":
            text = last.content[0].text if last.content else ""
            message = AssistantMessage(
                content=[TextContent(text=f"Tool {last.toolName} returned:\n{text}")],
                stopReason="stop",
            )
            yield AssistantMessageEvent(type="start", partial=message)
            yield AssistantMessageEvent(type="text_start", contentIndex=0, partial=message)
            yield AssistantMessageEvent(
                type="text_delta",
                contentIndex=0,
                delta=message.content[0].text,
                partial=message,
            )
            yield AssistantMessageEvent(
                type="text_end",
                contentIndex=0,
                content=message.content[0].text,
                partial=message,
            )
            yield AssistantMessageEvent(type="done", reason="stop", message=message)
            return

        prompt = str(last.content).lower()
        tool_names = {tool.name for tool in tools}
        if ("list" in prompt or "ls" in prompt or "files" in prompt) and "ls" in tool_names:
            call = ToolCall(id="mock-call-1", name="ls", arguments={"path": "."})
            message = AssistantMessage(content=[call], stopReason="toolUse")
            yield AssistantMessageEvent(type="start", partial=message)
            yield AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=message)
            yield AssistantMessageEvent(
                type="toolcall_delta",
                contentIndex=0,
                delta='{"path":"."}',
                partial=message,
            )
            yield AssistantMessageEvent(
                type="toolcall_end",
                contentIndex=0,
                toolCall=call,
                partial=message,
            )
            yield AssistantMessageEvent(type="done", reason="toolUse", message=message)
            return
        if ("read" in prompt or "show" in prompt) and "read" in tool_names:
            call = ToolCall(id="mock-call-1", name="read", arguments={"path": "README.md"})
            message = AssistantMessage(content=[call], stopReason="toolUse")
            yield AssistantMessageEvent(type="start", partial=message)
            yield AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=message)
            yield AssistantMessageEvent(
                type="toolcall_delta",
                contentIndex=0,
                delta='{"path":"README.md"}',
                partial=message,
            )
            yield AssistantMessageEvent(
                type="toolcall_end",
                contentIndex=0,
                toolCall=call,
                partial=message,
            )
            yield AssistantMessageEvent(type="done", reason="toolUse", message=message)
            return

        text = f"Mock response: {last.content}"
        message = AssistantMessage(content=[TextContent(text=text)], stopReason="stop")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="text_start", contentIndex=0, partial=message)
        yield AssistantMessageEvent(type="text_delta", contentIndex=0, delta=text, partial=message)
        yield AssistantMessageEvent(type="text_end", contentIndex=0, content=text, partial=message)
        yield AssistantMessageEvent(type="done", reason="stop", message=message)
