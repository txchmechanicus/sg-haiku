from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from upstream.models import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from upstream.providers.anthropic_messages import AnthropicMessagesProvider
from upstream.types import ToolSpec


def sse(*events: dict[str, object]) -> str:
    lines = [f"data: {json.dumps(event)}\n" for event in events]
    return "\n".join(lines)


def provider_with_response(
    body: str,
    *,
    is_oauth: bool = True,
    seen_requests: list[dict[str, object]] | None = None,
    seen_headers: list[httpx.Headers] | None = None,
    access_token: str = "token-1",
) -> AnthropicMessagesProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen_requests is not None:
            seen_requests.append(json.loads(request.content))
        if seen_headers is not None:
            seen_headers.append(request.headers)
        return httpx.Response(200, content=body, request=request)

    return AnthropicMessagesProvider(
        model="claude-sonnet-5",
        access_token=access_token,
        is_oauth=is_oauth,
        transport=httpx.MockTransport(handler),
    )


async def collect(
    provider: AnthropicMessagesProvider, tools: list[ToolSpec] | None = None, **kwargs
):
    return [
        event
        async for event in provider.stream(
            [UserMessage(content="hello", timestamp=123)], tools or [], **kwargs
        )
    ]


def test_oauth_headers_include_identity_and_beta_flag() -> None:
    seen_headers: list[httpx.Headers] = []
    provider = provider_with_response(
        sse({"type": "message_stop"}), is_oauth=True, seen_headers=seen_headers
    )

    asyncio.run(collect(provider))

    headers = seen_headers[0]
    assert headers["Authorization"] == "Bearer token-1"
    assert headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
    assert headers["user-agent"] == "claude-cli/2.1.75"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert headers["x-app"] == "cli"


def test_api_key_headers_have_no_identity_spoofing() -> None:
    seen_headers: list[httpx.Headers] = []
    provider = provider_with_response(
        sse({"type": "message_stop"}), is_oauth=False, seen_headers=seen_headers
    )

    asyncio.run(collect(provider))

    headers = seen_headers[0]
    assert headers["x-api-key"] == "token-1"
    assert "Authorization" not in headers
    assert "anthropic-beta" not in headers
    assert "user-agent" not in headers or headers["user-agent"] != "claude-cli/2.1.75"


def test_oauth_forces_claude_code_system_prefix() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "message_stop"}), is_oauth=True, seen_requests=seen_requests
    )

    asyncio.run(collect(provider))

    system = seen_requests[0]["system"]
    assert system[0] == {
        "type": "text",
        "text": "You are Claude Code, Anthropic's official CLI for Claude.",
    }


def test_api_key_path_has_no_forced_system_prefix() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "message_stop"}), is_oauth=False, seen_requests=seen_requests
    )

    asyncio.run(collect(provider))

    assert "system" not in seen_requests[0]


def test_oauth_maps_tool_name_to_claude_code_name() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "message_stop"}), is_oauth=True, seen_requests=seen_requests
    )
    tools = [ToolSpec(name="read", description="Read a file", parameters={"type": "object"})]

    asyncio.run(collect(provider, tools))

    assert seen_requests[0]["tools"][0]["name"] == "Read"


def test_non_oauth_keeps_original_tool_name() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "message_stop"}), is_oauth=False, seen_requests=seen_requests
    )
    tools = [ToolSpec(name="read", description="Read a file", parameters={"type": "object"})]

    asyncio.run(collect(provider, tools))

    assert seen_requests[0]["tools"][0]["name"] == "read"


@pytest.mark.asyncio
async def test_streams_text_output() -> None:
    provider = provider_with_response(
        sse(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 3, "output_tokens": 0}},
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
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
    assert events[-1].message.content[0].text == "Hello"
    assert events[-1].message.stopReason == "stop"
    assert events[-1].message.usage.input == 3
    assert events[-1].message.usage.output == 2


@pytest.mark.asyncio
async def test_streams_tool_call() -> None:
    provider = provider_with_response(
        sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call-1", "name": "ls"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"path"'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ':"."}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
            {"type": "message_stop"},
        ),
        is_oauth=False,
    )

    events = await collect(provider)

    assert [event.type for event in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    tool_call = events[-2].toolCall
    assert tool_call == ToolCall(id="call-1", name="ls", arguments={"path": "."})
    assert events[-1].message.stopReason == "toolUse"


@pytest.mark.asyncio
async def test_refusal_stop_reason_maps_to_error() -> None:
    provider = provider_with_response(
        sse(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}},
            {"type": "message_stop"},
        )
    )

    events = await collect(provider)

    assert events[-1].message.stopReason == "error"
    assert "refusal" in (events[-1].message.errorMessage or "")


@pytest.mark.asyncio
async def test_error_event_yields_error() -> None:
    provider = provider_with_response(sse({"type": "error", "error": {"message": "boom"}}))

    events = await collect(provider)

    assert [event.type for event in events] == ["start", "error"]
    assert events[-1].error is not None
    assert events[-1].error.stopReason == "error"


def test_convert_messages_batches_consecutive_tool_results() -> None:
    from upstream.providers.anthropic_messages import _convert_messages

    messages = [
        AssistantMessage(
            content=[ToolCall(id="call-1", name="ls", arguments={})],
            stopReason="toolUse",
        ),
        ToolResultMessage(
            toolCallId="call-1", toolName="ls", content=[TextContent(text="a.txt")], isError=False
        ),
        ToolResultMessage(
            toolCallId="call-2", toolName="ls", content=[TextContent(text="b.txt")], isError=False
        ),
    ]

    converted = _convert_messages(messages, is_oauth=False)

    assert converted[0]["role"] == "assistant"
    assert converted[1]["role"] == "user"
    assert len(converted[1]["content"]) == 2
    assert converted[1]["content"][0]["tool_use_id"] == "call-1"
    assert converted[1]["content"][1]["tool_use_id"] == "call-2"
