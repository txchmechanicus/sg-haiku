from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from upstream.models import (
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    Usage,
)
from upstream.providers.base import ModelProvider
from upstream.providers.sse import iter_sse_data
from upstream.types import ToolSpec


@dataclass
class _ToolCallBuffer:
    content_index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


class OpenAICompatibleProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
        supports_images: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.headers = headers or {}
        self.supports_images = supports_images

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        message = AssistantMessage(
            content=[],
            api="openai-completions",
            provider="openai",
            model=self.model,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages, system_prompt=system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = [self._format_tool(tool) for tool in tools]

        yield AssistantMessageEvent(type="start", partial=message)

        text_index: int | None = None
        tool_buffers: dict[int, _ToolCallBuffer] = {}
        finish_reason: str | None = None

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for data in iter_sse_data(response):
                        if abort_event is not None and abort_event.is_set():
                            break
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage:
                            message.usage = _usage_from_openai(usage)

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice_data = choices[0]
                        finish_reason = choice_data.get("finish_reason") or finish_reason
                        delta = choice_data.get("delta") or {}

                        text_delta = delta.get("content")
                        if text_delta:
                            if text_index is None:
                                text_index = len(message.content)
                                message.content.append(TextContent(text=""))
                                yield AssistantMessageEvent(
                                    type="text_start",
                                    contentIndex=text_index,
                                    partial=message,
                                )
                            text_part = message.content[text_index]
                            if isinstance(text_part, TextContent):
                                text_part.text += text_delta
                            yield AssistantMessageEvent(
                                type="text_delta",
                                contentIndex=text_index,
                                delta=text_delta,
                                partial=message,
                            )

                        for raw_call in delta.get("tool_calls") or []:
                            call_index = int(raw_call.get("index", 0))
                            buffer = tool_buffers.get(call_index)
                            if buffer is None:
                                buffer = _ToolCallBuffer(content_index=len(message.content))
                                tool_buffers[call_index] = buffer
                                message.content.append(ToolCall(id="", name="", arguments={}))
                                yield AssistantMessageEvent(
                                    type="toolcall_start",
                                    contentIndex=buffer.content_index,
                                    partial=message,
                                )

                            if raw_call.get("id"):
                                buffer.id = raw_call["id"]
                            function = raw_call.get("function") or {}
                            if function.get("name"):
                                buffer.name += function["name"]
                            argument_delta = function.get("arguments") or ""
                            if argument_delta:
                                buffer.arguments += argument_delta
                            current = message.content[buffer.content_index]
                            if isinstance(current, ToolCall):
                                current.id = buffer.id
                                current.name = buffer.name
                            event_delta = (
                                argument_delta
                                or function.get("name")
                                or raw_call.get("id")
                                or ""
                            )
                            yield AssistantMessageEvent(
                                type="toolcall_delta",
                                contentIndex=buffer.content_index,
                                delta=event_delta,
                                partial=message,
                            )
        except Exception as exc:
            error = AssistantMessage(
                content=[TextContent(text=str(exc))],
                api="openai-completions",
                provider="openai",
                model=self.model,
                stopReason="error",
                errorMessage=str(exc),
            )
            yield AssistantMessageEvent(type="error", error=error)
            return

        if text_index is not None:
            text_part = message.content[text_index]
            content = text_part.text if isinstance(text_part, TextContent) else ""
            yield AssistantMessageEvent(
                type="text_end",
                contentIndex=text_index,
                content=content,
                partial=message,
            )

        for _, buffer in sorted(tool_buffers.items()):
            part = message.content[buffer.content_index]
            arguments = _parse_tool_arguments(buffer.arguments)
            if isinstance(part, ToolCall):
                part.id = buffer.id
                part.name = buffer.name
                part.arguments = arguments
                tool_call = part
            else:
                tool_call = ToolCall(id=buffer.id, name=buffer.name, arguments=arguments)
                message.content[buffer.content_index] = tool_call
            yield AssistantMessageEvent(
                type="toolcall_end",
                contentIndex=buffer.content_index,
                toolCall=tool_call,
                partial=message,
            )

        if abort_event is not None and abort_event.is_set():
            message.stopReason = "aborted"
            message.errorMessage = "Operation aborted"
            yield AssistantMessageEvent(type="error", reason="aborted", error=message)
            return

        message.stopReason = _map_finish_reason(finish_reason, has_tool_calls=bool(tool_buffers))
        yield AssistantMessageEvent(type="done", reason=message.stopReason, message=message)

    def _format_message(self, message: Message) -> dict[str, Any]:
        if message.role == "tool":
            raise ValueError("legacy tool role is not supported")
        if message.role == "system":
            return {"role": "system", "content": message.content}
        if message.role == "toolResult":
            return {
                "role": "tool",
                "tool_call_id": message.toolCallId,
                "name": message.toolName,
                "content": "\n".join(
                    part.text for part in message.content if isinstance(part, TextContent)
                ),
            }

        if message.role == "user":
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mimeType};base64,{part.data}"},
                    })
            return {"role": "user", "content": parts}
        text = "".join(part.text for part in message.content if isinstance(part, TextContent))
        formatted: dict[str, Any] = {
            "role": "assistant",
            "content": text or None,
        }
        tool_calls = [part for part in message.content if isinstance(part, ToolCall)]
        if tool_calls:
            formatted["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ]
        return formatted

    def _format_messages(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        pending_images: list[dict[str, Any]] = []

        def flush_images() -> None:
            if pending_images:
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Attached image(s) from tool result:"},
                            *pending_images,
                        ],
                    }
                )
                pending_images.clear()

        for message in messages:
            if message.role == "toolResult":
                formatted.append(self._format_message(message))
                if self.supports_images:
                    pending_images.extend(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{part.mimeType};base64,{part.data}"},
                        }
                        for part in message.content
                        if isinstance(part, ImageContent)
                    )
                continue
            flush_images()
            formatted.append(self._format_message(message))
        flush_images()
        return formatted

    def _format_tool(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _usage_from_openai(usage: dict[str, Any]) -> Usage:
    return Usage(
        input=usage.get("prompt_tokens", 0) or 0,
        output=usage.get("completion_tokens", 0) or 0,
        totalTokens=usage.get("total_tokens", 0) or 0,
    )


def _parse_tool_arguments(raw_args: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_args}
    return value if isinstance(value, dict) else {"_raw": raw_args}


def _map_finish_reason(reason: str | None, *, has_tool_calls: bool) -> str:
    if reason == "tool_calls" or has_tool_calls:
        return "toolUse"
    if reason == "stop" or reason is None:
        return "stop"
    if reason == "length":
        return "length"
    if reason == "content_filter":
        return "aborted"
    return "error"
