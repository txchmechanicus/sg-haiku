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
from upstream.providers.base import ModelProvider
from upstream.providers.sse import iter_sse_data
from upstream.types import ThinkingLevel, ToolSpec

# The Messages API itself is the real public api.anthropic.com surface; the OAuth-specific
# headers/system-prompt/tool naming below are what's required to be accepted on the
# OAuth-gated (Claude Pro/Max subscription) path -- see oauth_anthropic.py for the identity
# note.
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_CODE_USER_AGENT = "claude-cli/2.1.75"
OAUTH_BETA_HEADER = "claude-code-20250219,oauth-2025-04-20"
DEFAULT_MAX_TOKENS = 8192

_THINKING_BUDGET_TOKENS: dict[ThinkingLevel, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 16384,
}

_CLAUDE_CODE_TOOL_NAMES = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]
_CLAUDE_CODE_TOOL_LOOKUP = {name.lower(): name for name in _CLAUDE_CODE_TOOL_NAMES}


def _to_claude_code_name(name: str) -> str:
    return _CLAUDE_CODE_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: list[ToolSpec]) -> str:
    for tool in tools:
        if tool.name.lower() == name.lower():
            return tool.name
    return name


class AnthropicApiError(Exception):
    pass


@dataclass
class _ToolCallBuffer:
    content_index: int
    id: str
    name: str
    arguments: str = ""


class AnthropicMessagesProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str,
        access_token: str,
        is_oauth: bool,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.access_token = access_token
        self.is_oauth = is_oauth
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.max_tokens = max_tokens

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
            api="anthropic-messages",
            provider="anthropic",
            model=self.model,
        )
        payload = self._build_payload(messages, tools, system_prompt, reasoning)

        yield AssistantMessageEvent(type="start", partial=message)

        content_index_by_block: dict[int, int] = {}
        block_kinds: dict[int, str] = {}
        tool_buffers: dict[int, _ToolCallBuffer] = {}
        stop_reason = "stop"
        error_message: str | None = None

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
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

                        if event_type == "message_start":
                            usage = (event.get("message") or {}).get("usage")
                            if usage:
                                message.usage = _merge_usage(message.usage, usage)
                            continue

                        if event_type == "content_block_start":
                            block_index = event["index"]
                            block = event["content_block"]
                            block_type = block.get("type")
                            content_index = len(message.content)
                            content_index_by_block[block_index] = content_index
                            if block_type == "text":
                                block_kinds[block_index] = "text"
                                message.content.append(TextContent(text=""))
                                yield AssistantMessageEvent(
                                    type="text_start", contentIndex=content_index, partial=message
                                )
                            elif block_type in ("thinking", "redacted_thinking"):
                                block_kinds[block_index] = "thinking"
                                message.content.append(ThinkingContent(thinking=""))
                                yield AssistantMessageEvent(
                                    type="thinking_start",
                                    contentIndex=content_index,
                                    partial=message,
                                )
                            elif block_type == "tool_use":
                                block_kinds[block_index] = "toolcall"
                                call_id = str(block.get("id", ""))
                                raw_name = str(block.get("name", ""))
                                name = (
                                    _from_claude_code_name(raw_name, tools)
                                    if self.is_oauth
                                    else raw_name
                                )
                                tool_buffers[block_index] = _ToolCallBuffer(
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

                        if event_type == "content_block_delta":
                            block_index = event["index"]
                            delta = event.get("delta") or {}
                            delta_type = delta.get("type")
                            content_index = content_index_by_block.get(block_index)
                            if content_index is None:
                                continue
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                part = message.content[content_index]
                                if isinstance(part, TextContent):
                                    part.text += text
                                yield AssistantMessageEvent(
                                    type="text_delta",
                                    contentIndex=content_index,
                                    delta=text,
                                    partial=message,
                                )
                            elif delta_type == "thinking_delta":
                                thinking = delta.get("thinking", "")
                                part = message.content[content_index]
                                if isinstance(part, ThinkingContent):
                                    part.thinking += thinking
                                yield AssistantMessageEvent(
                                    type="thinking_delta",
                                    contentIndex=content_index,
                                    delta=thinking,
                                    partial=message,
                                )
                            elif delta_type == "input_json_delta":
                                buffer = tool_buffers.get(block_index)
                                if buffer is None:
                                    continue
                                partial_json = delta.get("partial_json", "")
                                buffer.arguments += partial_json
                                yield AssistantMessageEvent(
                                    type="toolcall_delta",
                                    contentIndex=content_index,
                                    delta=partial_json,
                                    partial=message,
                                )
                            elif delta_type == "signature_delta":
                                part = message.content[content_index]
                                if isinstance(part, ThinkingContent):
                                    part.thinkingSignature = (part.thinkingSignature or "") + (
                                        delta.get("signature", "")
                                    )
                            continue

                        if event_type == "content_block_stop":
                            block_index = event["index"]
                            kind = block_kinds.get(block_index)
                            content_index = content_index_by_block.get(block_index)
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
                                buffer = tool_buffers.get(block_index)
                                if buffer is not None:
                                    part = message.content[buffer.content_index]
                                    if isinstance(part, ToolCall):
                                        part.arguments = _parse_tool_arguments(buffer.arguments)
                                        yield AssistantMessageEvent(
                                            type="toolcall_end",
                                            contentIndex=buffer.content_index,
                                            toolCall=part,
                                            partial=message,
                                        )
                            continue

                        if event_type == "message_delta":
                            delta = event.get("delta") or {}
                            raw_stop_reason = delta.get("stop_reason")
                            if raw_stop_reason:
                                stop_reason, error_message = _map_stop_reason(raw_stop_reason)
                            usage = event.get("usage")
                            if usage:
                                message.usage = _merge_usage(message.usage, usage)
                            continue

                        if event_type == "message_stop":
                            continue

                        if event_type == "error":
                            raise AnthropicApiError(json.dumps(event))
        except Exception as exc:
            error = AssistantMessage(
                content=[TextContent(text=str(exc))],
                api="anthropic-messages",
                provider="anthropic",
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
        if stop_reason == "error" and error_message:
            message.errorMessage = error_message
        yield AssistantMessageEvent(type="done", reason=stop_reason, message=message)

    def _headers(self) -> dict[str, str]:
        headers = {
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.is_oauth:
            headers["Authorization"] = f"Bearer {self.access_token}"
            headers["anthropic-beta"] = OAUTH_BETA_HEADER
            headers["anthropic-dangerous-direct-browser-access"] = "true"
            headers["user-agent"] = CLAUDE_CODE_USER_AGENT
            headers["x-app"] = "cli"
        else:
            headers["x-api-key"] = self.access_token
        return headers

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None,
        reasoning: ThinkingLevel | None = None,
    ) -> dict[str, Any]:
        system_blocks: list[dict[str, str]] = []
        if self.is_oauth:
            system_blocks.append({"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX})
        if system_prompt:
            system_blocks.append({"type": "text", "text": system_prompt})
        for entry in messages:
            if isinstance(entry, SystemMessage):
                system_blocks.append({"type": "text", "text": entry.content})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "messages": _convert_messages(messages, is_oauth=self.is_oauth),
        }
        if system_blocks:
            payload["system"] = system_blocks
        if tools:
            payload["tools"] = [_convert_tool(tool, self.is_oauth) for tool in tools]
            payload["tool_choice"] = {"type": "auto"}
        if reasoning is not None:
            budget = _THINKING_BUDGET_TOKENS[reasoning]
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(payload["max_tokens"], budget + 1024)
        return payload


def _convert_messages(messages: list[Message], *, is_oauth: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            result.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for entry in messages:
        if isinstance(entry, ToolResultMessage):
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": entry.toolCallId,
                    "content": [_convert_content_part(part) for part in entry.content],
                    "is_error": entry.isError,
                }
            )
            continue
        flush_tool_results()
        if isinstance(entry, SystemMessage):
            continue
        if isinstance(entry, UserMessage):
            result.append({"role": "user", "content": _convert_user_content(entry)})
        elif isinstance(entry, AssistantMessage):
            result.append(
                {"role": "assistant", "content": _convert_assistant_content(entry, is_oauth)}
            )
    flush_tool_results()
    return result


def _convert_user_content(message: UserMessage) -> list[dict[str, Any]] | str:
    if isinstance(message.content, str):
        return message.content
    return [_convert_content_part(part) for part in message.content]


def _convert_assistant_content(message: AssistantMessage, is_oauth: bool) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ThinkingContent):
            if part.redacted:
                blocks.append({"type": "redacted_thinking", "data": part.thinking})
            else:
                block: dict[str, Any] = {"type": "thinking", "thinking": part.thinking}
                if part.thinkingSignature:
                    block["signature"] = part.thinkingSignature
                blocks.append(block)
        elif isinstance(part, ToolCall):
            name = _to_claude_code_name(part.name) if is_oauth else part.name
            blocks.append(
                {"type": "tool_use", "id": part.id, "name": name, "input": part.arguments}
            )
    return blocks


def _convert_content_part(part: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(part, ImageContent):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": part.mimeType, "data": part.data},
        }
    return {"type": "text", "text": part.text}


def _convert_tool(tool: ToolSpec, is_oauth: bool) -> dict[str, Any]:
    name = _to_claude_code_name(tool.name) if is_oauth else tool.name
    schema = tool.parameters or {}
    return {
        "name": name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        },
    }


def _merge_usage(existing: Usage, raw: dict[str, Any]) -> Usage:
    input_tokens = int(raw.get("input_tokens") or existing.input)
    output_tokens = int(raw.get("output_tokens") or existing.output)
    cache_read = int(raw.get("cache_read_input_tokens") or existing.cacheRead)
    cache_write = int(raw.get("cache_creation_input_tokens") or existing.cacheWrite)
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cacheRead=cache_read,
        cacheWrite=cache_write,
        totalTokens=input_tokens + output_tokens + cache_read + cache_write,
    )


def _map_stop_reason(raw: str) -> tuple[str, str | None]:
    if raw in ("end_turn", "pause_turn", "stop_sequence"):
        return "stop", None
    if raw == "max_tokens":
        return "length", None
    if raw == "tool_use":
        return "toolUse", None
    if raw in ("refusal", "sensitive"):
        return "error", f"Anthropic stop_reason: {raw}"
    return "error", f"Unhandled stop reason: {raw}"


def _parse_tool_arguments(raw_args: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_args}
    return value if isinstance(value, dict) else {"_raw": raw_args}
