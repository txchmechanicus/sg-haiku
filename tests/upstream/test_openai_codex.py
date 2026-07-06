from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from upstream.models import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from upstream.providers.openai_codex import OpenAICodexProvider
from upstream.types import ToolSpec


def _fake_jwt(account_id: str) -> str:
    def _segment(data: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode("ascii")

    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    return f"{_segment({'alg': 'none'})}.{_segment(payload)}.sig"


def sse(*events: dict[str, object]) -> str:
    lines = [f"data: {json.dumps(event)}\n" for event in events]
    return "\n".join(lines)


def provider_with_response(
    body: str,
    *,
    seen_requests: list[dict[str, object]] | None = None,
    seen_headers: list[httpx.Headers] | None = None,
    access_token: str | None = None,
) -> OpenAICodexProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen_requests is not None:
            seen_requests.append(json.loads(request.content))
        if seen_headers is not None:
            seen_headers.append(request.headers)
        return httpx.Response(200, content=body, request=request)

    return OpenAICodexProvider(
        model="gpt-5.5",
        access_token=access_token or _fake_jwt("acct-1"),
        transport=httpx.MockTransport(handler),
    )


async def collect(provider: OpenAICodexProvider, tools: list[ToolSpec] | None = None, **kwargs):
    return [
        event
        async for event in provider.stream(
            [UserMessage(content="hello", timestamp=123)], tools or [], **kwargs
        )
    ]


def test_headers_include_account_id_and_originator() -> None:
    seen_headers: list[httpx.Headers] = []
    provider = provider_with_response(
        sse({"type": "response.completed", "response": {}}), seen_headers=seen_headers
    )

    asyncio.run(collect(provider))

    headers = seen_headers[0]
    assert headers["chatgpt-account-id"] == "acct-1"
    assert headers["originator"] == "haiku"
    assert headers["Authorization"].startswith("Bearer ")


def test_request_payload_shape() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "response.completed", "response": {}}), seen_requests=seen_requests
    )

    asyncio.run(collect(provider))

    payload = seen_requests[0]
    assert payload["model"] == "gpt-5.5"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["instructions"] == "You are a helpful assistant."
    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert "tools" not in payload


def test_request_includes_tools_when_provided() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "response.completed", "response": {}}), seen_requests=seen_requests
    )
    tools = [ToolSpec(name="read", description="Read a file", parameters={"type": "object"})]

    asyncio.run(collect(provider, tools))

    payload = seen_requests[0]
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]


def test_reasoning_none_leaves_payload_unchanged() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "response.completed", "response": {}}), seen_requests=seen_requests
    )

    asyncio.run(collect(provider))

    assert "reasoning" not in seen_requests[0]


def test_reasoning_sets_effort_and_clamps_xhigh() -> None:
    seen_requests: list[dict[str, object]] = []
    provider = provider_with_response(
        sse({"type": "response.completed", "response": {}}), seen_requests=seen_requests
    )

    asyncio.run(collect(provider, reasoning="medium"))
    assert seen_requests[-1]["reasoning"] == {"effort": "medium"}

    asyncio.run(collect(provider, reasoning="xhigh"))
    assert seen_requests[-1]["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_streams_text_output() -> None:
    provider = provider_with_response(
        sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message"},
            },
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"},
            {"type": "response.output_item.done", "output_index": 0},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}},
            },
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
    assert events[-1].message.usage.totalTokens == 5
    assert events[-1].message.stopReason == "stop"


@pytest.mark.asyncio
async def test_streams_tool_call() -> None:
    provider = provider_with_response(
        sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call-1", "name": "ls"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": '{"path"',
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": ':"."}',
            },
            {"type": "response.function_call_arguments.done", "output_index": 0},
            {"type": "response.output_item.done", "output_index": 0},
            {"type": "response.completed", "response": {}},
        )
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
async def test_response_failed_yields_error_event() -> None:
    provider = provider_with_response(sse({"type": "response.failed", "response": {}}))

    events = await collect(provider)

    assert [event.type for event in events] == ["start", "error"]
    assert events[-1].error is not None
    assert events[-1].error.stopReason == "error"


@pytest.mark.asyncio
async def test_abort_event_set_before_stream_yields_aborted() -> None:
    provider = provider_with_response(
        sse(
            {"type": "response.completed", "response": {}},
        )
    )
    abort_event = asyncio.Event()
    abort_event.set()

    events = await collect(provider, abort_event=abort_event)

    assert events[-1].type == "error"
    assert events[-1].error is not None
    assert events[-1].error.stopReason == "aborted"


def test_convert_input_items_round_trips_tool_call_and_result() -> None:
    from upstream.providers.openai_codex import _convert_input_items

    messages = [
        AssistantMessage(
            content=[
                TextContent(text="ok"),
                ToolCall(id="call-1", name="ls", arguments={"path": "."}),
            ],
            stopReason="toolUse",
        ),
        ToolResultMessage(
            toolCallId="call-1",
            toolName="ls",
            content=[TextContent(text="a.txt")],
            isError=False,
        ),
    ]

    items = _convert_input_items(messages)

    assert items[0]["type"] == "message"
    assert items[0]["content"] == [{"type": "output_text", "text": "ok", "annotations": []}]
    assert items[1] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "ls",
        "arguments": '{"path": "."}',
    }
    assert items[2] == {"type": "function_call_output", "call_id": "call-1", "output": "a.txt"}
