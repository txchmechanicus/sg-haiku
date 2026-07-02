from __future__ import annotations

import json

import httpx
import pytest
from upstream import (
    AssistantMessage,
    ImageContent,
    OpenAICompatibleProvider,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)


def sse(*payloads: dict[str, object]) -> str:
    lines = [f"data: {json.dumps(payload)}\n" for payload in payloads]
    lines.append("data: [DONE]\n")
    return "\n".join(lines)


def provider_with_response(
    body: str,
    *,
    status_code: int = 200,
    seen_requests: list[dict[str, object]] | None = None,
) -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen_requests is not None:
            seen_requests.append(json.loads(request.content))
        return httpx.Response(status_code, content=body, request=request)

    return OpenAICompatibleProvider(
        model="gpt-test",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )


async def collect(provider: OpenAICompatibleProvider, tools: list[ToolSpec] | None = None):
    return [
        event
        async for event in provider.stream(
            [UserMessage(content="hello", timestamp=123)],
            tools or [],
        )
    ]


async def collect_with_system_prompt(
    provider: OpenAICompatibleProvider,
    system_prompt: str,
):
    return [
        event
        async for event in provider.stream(
            [UserMessage(content="hello", timestamp=123)],
            [],
            system_prompt=system_prompt,
        )
    ]


@pytest.mark.asyncio
async def test_openai_compatible_streams_text_events() -> None:
    provider = provider_with_response(
        sse(
            {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )
    )

    events = await collect(provider)

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert [event.delta for event in events if event.type == "text_delta"] == ["Hel", "lo"]
    assert events[-1].message is not None
    assert events[-1].message.content[0].text == "Hello"
    assert events[-1].message.stopReason == "stop"


@pytest.mark.asyncio
async def test_openai_compatible_streams_tool_call_events() -> None:
    provider = provider_with_response(
        sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "ls",
                                        "arguments": "{\"pa",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "th\":\".\"}"},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        )
    )

    events = await collect(provider)
    tool_call = events[-2].toolCall

    assert [event.type for event in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert tool_call == ToolCall(id="call-1", name="ls", arguments={"path": "."})
    assert events[-1].message is not None
    assert events[-1].message.stopReason == "toolUse"


@pytest.mark.asyncio
async def test_openai_compatible_maps_stream_usage() -> None:
    provider = provider_with_response(
        sse(
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )
    )

    events = await collect(provider)

    assert events[-1].message is not None
    assert events[-1].message.usage.input == 3
    assert events[-1].message.usage.output == 2
    assert events[-1].message.usage.totalTokens == 5


@pytest.mark.asyncio
async def test_openai_compatible_encodes_http_errors_as_error_event() -> None:
    provider = provider_with_response("server error", status_code=500)

    events = await collect(provider)

    assert [event.type for event in events] == ["start", "error"]
    assert events[-1].error is not None
    assert events[-1].error.stopReason == "error"
    assert "500" in (events[-1].error.errorMessage or "")


@pytest.mark.asyncio
async def test_openai_compatible_request_includes_stream_and_tools() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        seen_requests=seen_requests,
    )
    tools = [
        ToolSpec(
            name="read",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]

    await collect(provider, tools)

    request = seen_requests[0]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_openai_compatible_request_includes_system_prompt() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        seen_requests=seen_requests,
    )

    await collect_with_system_prompt(provider, "system instructions")

    request = seen_requests[0]
    assert request["messages"][0] == {
        "role": "system",
        "content": "system instructions",
    }
    assert request["messages"][1] == {"role": "user", "content": "hello"}


def test_openai_compatible_formats_tool_result_message() -> None:
    provider = OpenAICompatibleProvider(model="gpt-test", api_key="test-key")
    message = ToolResultMessage(
        toolCallId="call-1",
        toolName="read",
        content=[TextContent(text="file contents")],
        isError=False,
        timestamp=123,
    )

    assert provider._format_message(message) == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "read",
        "content": "file contents",
    }


def test_openai_compatible_formats_system_message() -> None:
    provider = OpenAICompatibleProvider(model="gpt-test", api_key="test-key")
    message = SystemMessage(content="compacted summary", timestamp=123)

    assert provider._format_message(message) == {
        "role": "system",
        "content": "compacted summary",
    }


def test_openai_compatible_formats_user_message_with_list_content() -> None:
    provider = OpenAICompatibleProvider(model="gpt-test", api_key="test-key")
    message = UserMessage(
        content=[
            TextContent(text="describe this image"),
            ImageContent(data="abc123", mimeType="image/png"),
        ],
        timestamp=123,
    )

    result = provider._format_message(message)
    assert result["role"] == "user"
    assert result["content"] == [
        {"type": "text", "text": "describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]


def test_openai_compatible_formats_assistant_with_only_tool_calls_has_null_content() -> None:
    provider = OpenAICompatibleProvider(model="gpt-test", api_key="test-key")
    message = AssistantMessage(
        content=[ToolCall(id="call-1", name="ls", arguments={})],
        timestamp=123,
    )

    result = provider._format_message(message)
    assert result["content"] is None


def test_openai_compatible_formats_assistant_tool_calls() -> None:
    provider = OpenAICompatibleProvider(model="gpt-test", api_key="test-key")
    message = AssistantMessage(
        content=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})],
        timestamp=123,
    )

    assert provider._format_message(message)["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": "{\"path\": \"README.md\"}",
            },
        }
    ]
