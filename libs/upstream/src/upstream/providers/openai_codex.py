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
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from upstream.oauth import decode_jwt_payload
from upstream.providers.base import ModelProvider
from upstream.providers.sse import iter_sse_data
from upstream.types import ThinkingLevel, ToolSpec

# This is OpenAI's internal ChatGPT-backend "Responses API" surface used by the official
# Codex CLI/OAuth login -- not the public api.openai.com/v1/chat/completions format.
# Unverified against a live account; see PLANS.md for the caveat.
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"
JWT_ACCOUNT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
ORIGINATOR = "haiku"


class CodexApiError(Exception):
    pass


@dataclass
class _ToolCallBuffer:
    content_index: int
    id: str
    name: str
    arguments: str = ""


class OpenAICodexProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str,
        access_token: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.account_id = _extract_account_id(access_token)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: ThinkingLevel | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        message = AssistantMessage(
            content=[],
            api="openai-responses",
            provider="openai-codex",
            model=self.model,
        )
        payload = self._build_payload(messages, tools, system_prompt, reasoning)

        yield AssistantMessageEvent(type="start", partial=message)

        item_kinds: dict[int, str] = {}
        item_content_index: dict[int, int] = {}
        tool_buffers: dict[int, _ToolCallBuffer] = {}
        stop_reason = "stop"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/codex/responses",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for data in iter_sse_data(response):
                        if abort_event is not None and abort_event.is_set():
                            break
                        if not data:
                            continue
                        event = json.loads(data)
                        event_type = event.get("type")
                        if event_type == "response.done":
                            event_type = "response.completed"

                        if event_type == "response.output_item.added":
                            output_index = event["output_index"]
                            item = event["item"]
                            kind = _item_kind(item.get("type"))
                            content_index = len(message.content)
                            item_kinds[output_index] = kind
                            item_content_index[output_index] = content_index
                            if kind == "text":
                                message.content.append(TextContent(text=""))
                                yield AssistantMessageEvent(
                                    type="text_start", contentIndex=content_index, partial=message
                                )
                            elif kind == "thinking":
                                message.content.append(ThinkingContent(thinking=""))
                                yield AssistantMessageEvent(
                                    type="thinking_start",
                                    contentIndex=content_index,
                                    partial=message,
                                )
                            elif kind == "toolcall":
                                call_id = str(item.get("call_id") or item.get("id") or "")
                                name = str(item.get("name") or "")
                                tool_buffers[output_index] = _ToolCallBuffer(
                                    content_index=content_index, id=call_id, name=name
                                )
                                message.content.append(
                                    ToolCall(id=call_id, name=name, arguments={})
                                )
                                yield AssistantMessageEvent(
                                    type="toolcall_start",
                                    contentIndex=content_index,
                                    partial=message,
                                )
                            continue

                        if event_type in ("response.output_text.delta", "response.refusal.delta"):
                            content_index = item_content_index.get(event.get("output_index", -1))
                            if content_index is None:
                                continue
                            delta = event.get("delta", "")
                            part = message.content[content_index]
                            if isinstance(part, TextContent):
                                part.text += delta
                            yield AssistantMessageEvent(
                                type="text_delta",
                                contentIndex=content_index,
                                delta=delta,
                                partial=message,
                            )
                            continue

                        if event_type in (
                            "response.reasoning_summary_text.delta",
                            "response.reasoning_text.delta",
                        ):
                            content_index = item_content_index.get(event.get("output_index", -1))
                            if content_index is None:
                                continue
                            delta = event.get("delta", "")
                            part = message.content[content_index]
                            if isinstance(part, ThinkingContent):
                                part.thinking += delta
                            yield AssistantMessageEvent(
                                type="thinking_delta",
                                contentIndex=content_index,
                                delta=delta,
                                partial=message,
                            )
                            continue

                        if event_type == "response.function_call_arguments.delta":
                            buffer = tool_buffers.get(event.get("output_index", -1))
                            if buffer is None:
                                continue
                            delta = event.get("delta", "")
                            buffer.arguments += delta
                            yield AssistantMessageEvent(
                                type="toolcall_delta",
                                contentIndex=buffer.content_index,
                                delta=delta,
                                partial=message,
                            )
                            continue

                        if event_type == "response.function_call_arguments.done":
                            buffer = tool_buffers.get(event.get("output_index", -1))
                            if buffer is None:
                                continue
                            part = message.content[buffer.content_index]
                            if isinstance(part, ToolCall):
                                part.arguments = _parse_tool_arguments(buffer.arguments)
                            continue

                        if event_type == "response.output_item.done":
                            output_index = event.get("output_index", -1)
                            kind = item_kinds.get(output_index)
                            content_index = item_content_index.get(output_index)
                            if kind == "text" and content_index is not None:
                                part = message.content[content_index]
                                text = part.text if isinstance(part, TextContent) else ""
                                yield AssistantMessageEvent(
                                    type="text_end",
                                    contentIndex=content_index,
                                    content=text,
                                    partial=message,
                                )
                            elif kind == "thinking" and content_index is not None:
                                part = message.content[content_index]
                                thinking = (
                                    part.thinking if isinstance(part, ThinkingContent) else ""
                                )
                                yield AssistantMessageEvent(
                                    type="thinking_end",
                                    contentIndex=content_index,
                                    content=thinking,
                                    partial=message,
                                )
                            elif kind == "toolcall":
                                buffer = tool_buffers.get(output_index)
                                if buffer is not None:
                                    tool_call = message.content[buffer.content_index]
                                    if isinstance(tool_call, ToolCall):
                                        yield AssistantMessageEvent(
                                            type="toolcall_end",
                                            contentIndex=buffer.content_index,
                                            toolCall=tool_call,
                                            partial=message,
                                        )
                                        stop_reason = "toolUse"
                            continue

                        if event_type in ("response.completed", "response.incomplete"):
                            response_obj = event.get("response") or {}
                            usage = response_obj.get("usage")
                            if usage:
                                message.usage = _usage_from_responses(usage)
                            if event_type == "response.incomplete":
                                stop_reason = "length"
                            continue

                        if event_type in ("error", "response.failed"):
                            raise CodexApiError(json.dumps(event))
        except Exception as exc:
            error = AssistantMessage(
                content=[TextContent(text=str(exc))],
                api="openai-responses",
                provider="openai-codex",
                model=self.model,
                stopReason="error",
                errorMessage=str(exc),
            )
            yield AssistantMessageEvent(type="error", error=error)
            return

        if abort_event is not None and abort_event.is_set():
            message.stopReason = "aborted"
            message.errorMessage = "Operation aborted"
            yield AssistantMessageEvent(type="error", reason="aborted", error=message)
            return

        message.stopReason = stop_reason
        yield AssistantMessageEvent(type="done", reason=stop_reason, message=message)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "originator": ORIGINATOR,
            "User-Agent": f"{ORIGINATOR} (python)",
            "OpenAI-Beta": "responses=experimental",
        }
        if self.account_id:
            headers["chatgpt-account-id"] = self.account_id
        return headers

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None,
        reasoning: ThinkingLevel | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "store": False,
            "stream": True,
            "instructions": system_prompt or DEFAULT_INSTRUCTIONS,
            "input": _convert_input_items(messages),
            "text": {"verbosity": "low"},
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            payload["tools"] = [_convert_tool(tool) for tool in tools]
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if reasoning is not None:
            effort = "high" if reasoning == "xhigh" else reasoning
            payload["reasoning"] = {"effort": effort}
        return payload


def _item_kind(item_type: str | None) -> str:
    if item_type == "message":
        return "text"
    if item_type == "reasoning":
        return "thinking"
    if item_type == "function_call":
        return "toolcall"
    return "unknown"


def _extract_account_id(access_token: str) -> str:
    try:
        claims = decode_jwt_payload(access_token)
        auth_claim = claims.get(JWT_ACCOUNT_CLAIM_PATH)
        if isinstance(auth_claim, dict):
            return str(auth_claim.get("chatgpt_account_id") or "")
    except Exception:
        pass
    return ""


def _convert_input_items(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            items.append({"role": "user", "content": _convert_user_content(message)})
        elif isinstance(message, SystemMessage):
            items.append({"role": "developer", "content": message.content})
        elif isinstance(message, AssistantMessage):
            items.extend(_convert_assistant_items(message))
        elif isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.toolCallId,
                    "output": "\n".join(
                        part.text for part in message.content if isinstance(part, TextContent)
                    ),
                }
            )
    return items


def _convert_user_content(message: UserMessage) -> list[dict[str, Any]] | str:
    if isinstance(message.content, str):
        return [{"type": "input_text", "text": message.content}]
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImageContent):
            parts.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{part.mimeType};base64,{part.data}",
                }
            )
    return parts


def _convert_assistant_items(message: AssistantMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    text_parts = [part for part in message.content if isinstance(part, TextContent)]
    if text_parts:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": part.text, "annotations": []}
                    for part in text_parts
                ],
            }
        )
    for part in message.content:
        if isinstance(part, ToolCall):
            items.append(
                {
                    "type": "function_call",
                    "call_id": part.id,
                    "name": part.name,
                    "arguments": json.dumps(part.arguments),
                }
            )
    return items


def _convert_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


def _usage_from_responses(usage: dict[str, Any]) -> Usage:
    return Usage(
        input=usage.get("input_tokens", 0) or 0,
        output=usage.get("output_tokens", 0) or 0,
        totalTokens=usage.get("total_tokens", 0) or 0,
    )


def _parse_tool_arguments(raw_args: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_args}
    return value if isinstance(value, dict) else {"_raw": raw_args}
