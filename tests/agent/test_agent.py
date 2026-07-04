from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from agent import Agent
from agent.core import (
    BeforeAgentStartResult,
    ProviderRequestPayload,
    ToolCallHookResult,
    ToolResultHookResult,
)
from coding_agent.tools import default_registry
from upstream import (
    AssistantMessage,
    AssistantMessageEvent,
    Message,
    MockProvider,
    ModelProvider,
    ToolCall,
    ToolSpec,
    UserMessage,
)
from upstream.types import AgentToolResult


async def collect(agent: Agent, prompt: str, *, use_tools: bool = True):
    return [event async for event in agent.run(prompt, use_tools=use_tools)]


@pytest.mark.asyncio
async def test_agent_returns_contract_events_without_tools(tmp_path: Path) -> None:
    agent = Agent(provider=MockProvider(), cwd=tmp_path)

    events = await collect(agent, "hello")

    assert events[0].type == "agent_start"
    assert [event.type for event in events].count("message_update") >= 1
    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[0].content[0].text == "Mock response: hello"


@pytest.mark.asyncio
async def test_agent_executes_ls_tool_with_contract_events(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")
    agent = Agent(provider=MockProvider(), tools=default_registry(tmp_path), cwd=tmp_path)

    events = await collect(agent, "list files")

    assert "tool_execution_start" in [event.type for event in events]
    assert "tool_execution_end" in [event.type for event in events]
    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].content[0].text == "example.txt"


@pytest.mark.asyncio
async def test_agent_reports_disabled_tools_as_tool_result(tmp_path: Path) -> None:
    agent = Agent(provider=MockProvider(), cwd=tmp_path)

    events = await collect(agent, "list files", use_tools=False)

    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[0].content[0].text == "Mock response: list files"


class AlwaysToolProvider(ModelProvider):
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        call = ToolCall(id="call-1", name="missing", arguments={})
        message = AssistantMessage(content=[call], stopReason="toolUse")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=message)
        yield AssistantMessageEvent(
            type="toolcall_end",
            contentIndex=0,
            toolCall=call,
            partial=message,
        )
        yield AssistantMessageEvent(type="done", reason="toolUse", message=message)


@pytest.mark.asyncio
async def test_agent_stops_after_max_tool_iterations(tmp_path: Path) -> None:
    agent = Agent(provider=AlwaysToolProvider(), cwd=tmp_path, max_tool_iterations=1)

    events = await collect(agent, "loop")

    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].stopReason == "error"
    assert "Stopped after 1 tool iterations" in assistant_messages[-1].errorMessage


class CapturingProvider(ModelProvider):
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.system_prompt: str | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.messages = list(messages)
        self.system_prompt = system_prompt
        message = AssistantMessage(content=[], stopReason="stop")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="done", reason="stop", message=message)


@pytest.mark.asyncio
async def test_agent_passes_initial_messages_to_provider(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)

    events = [
        event
        async for event in agent.run(
            "current",
            initial_messages=[UserMessage(content="previous", timestamp=123)],
        )
    ]

    assert [message.content for message in provider.messages if message.role == "user"] == [
        "previous",
        "current",
    ]
    started_users = [
        event.message.content
        for event in events
        if event.type == "message_start" and getattr(event.message, "role", None) == "user"
    ]
    assert started_users == ["current"]


@pytest.mark.asyncio
async def test_agent_passes_system_prompt_to_provider(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)

    await collect(agent, "current", use_tools=False)
    assert provider.system_prompt == agent.system_prompt

    _ = [
        event
        async for event in agent.run(
            "current",
            system_prompt="custom system prompt",
            use_tools=False,
        )
    ]
    assert provider.system_prompt == "custom system prompt"


@pytest.mark.asyncio
async def test_before_tool_call_hook_blocks_execution(tmp_path: Path) -> None:
    async def before_tool_call(call: ToolCall) -> ToolCallHookResult | None:
        return ToolCallHookResult(block=True, reason="nope")

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        before_tool_call=before_tool_call,
    )

    events = await collect(agent, "list files")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].isError is True
    assert tool_results[0].content[0].text == "nope"


@pytest.mark.asyncio
async def test_before_tool_call_hook_exception_becomes_tool_error(tmp_path: Path) -> None:
    async def before_tool_call(call: ToolCall) -> ToolCallHookResult | None:
        raise RuntimeError("extension exploded")

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        before_tool_call=before_tool_call,
    )

    events = await collect(agent, "list files")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].isError is True
    assert "extension exploded" in tool_results[0].content[0].text


@pytest.mark.asyncio
async def test_after_tool_call_hook_patches_result(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")

    async def after_tool_call(
        call: ToolCall, result: AgentToolResult, is_error: bool
    ) -> ToolResultHookResult | None:
        return ToolResultHookResult(details={"patched": True})

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        after_tool_call=after_tool_call,
    )

    events = await collect(agent, "list files")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].details == {"patched": True}
    assert tool_results[0].content[0].text == "example.txt"


@pytest.mark.asyncio
async def test_before_provider_request_hook_patches_payload() -> None:
    provider = CapturingProvider()

    async def before_provider_request(
        payload: ProviderRequestPayload,
    ) -> ProviderRequestPayload:
        payload.system_prompt = "patched by extension"
        return payload

    agent = Agent(provider=provider, before_provider_request=before_provider_request)

    await collect(agent, "hello", use_tools=False)

    assert provider.system_prompt == "patched by extension"


@pytest.mark.asyncio
async def test_before_agent_start_hook_injects_messages_and_system_prompt() -> None:
    provider = CapturingProvider()

    async def before_agent_start(prompt: str, system_prompt: str) -> BeforeAgentStartResult:
        return BeforeAgentStartResult(
            messages=[UserMessage(content="injected", timestamp=1)],
            system_prompt="overridden system prompt",
        )

    agent = Agent(provider=provider, before_agent_start=before_agent_start)

    await collect(agent, "hello", use_tools=False)

    assert provider.system_prompt == "overridden system prompt"
    assert [message.content for message in provider.messages if message.role == "user"] == [
        "injected",
        "hello",
    ]
