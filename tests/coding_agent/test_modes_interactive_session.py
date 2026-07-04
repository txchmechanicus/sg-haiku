from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent import Agent
from coding_agent.modes.interactive import InteractiveSession
from upstream.models import (
    AssistantMessage,
    AssistantMessageEvent,
    Message,
    TextContent,
    ToolCall,
)
from upstream.providers.base import ModelProvider
from upstream.types import AgentToolResult, ToolSpec

from tests.tui.fake_terminal import FakeTerminal


def _text_turn(text: str) -> list[AssistantMessageEvent]:
    message = AssistantMessage(content=[TextContent(text="")], stopReason="stop")
    events = [AssistantMessageEvent(type="start", partial=message)]
    for chunk in text:
        message.content[0].text += chunk
        events.append(AssistantMessageEvent(type="text_delta", delta=chunk, partial=message))
    final = AssistantMessage(content=[TextContent(text=text)], stopReason="stop")
    events.append(AssistantMessageEvent(type="done", reason="stop", message=final))
    return events


def _tool_call_turn(tool_call_id: str, tool_name: str) -> list[AssistantMessageEvent]:
    call = ToolCall(id=tool_call_id, name=tool_name, arguments={"x": 1})
    message = AssistantMessage(content=[call], stopReason="toolUse")
    return [
        AssistantMessageEvent(type="start", partial=message),
        AssistantMessageEvent(type="done", reason="toolUse", message=message),
    ]


class _ScriptedProvider(ModelProvider):
    def __init__(self, turns: list[list[AssistantMessageEvent]]) -> None:
        self._turns = iter(turns)
        self.seen_initial_messages: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.seen_initial_messages.append(list(messages))
        for event in next(self._turns):
            yield event


class _NoTools:
    def specs(self) -> list[ToolSpec]:
        return []

    async def run(self, call):  # pragma: no cover - not expected to run
        raise AssertionError("no tool should run")


class _EchoTool:
    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name="echo", description="Echo.", parameters={"type": "object"})]

    async def run(self, call):
        return AgentToolResult.text("echoed"), False


async def _run_scripted(
    session: InteractiveSession, terminal: FakeTerminal, inputs: list[str]
) -> None:
    run_task = asyncio.ensure_future(session.run())
    for line in inputs:
        for char in line:
            terminal.feed(char.encode())
        terminal.feed(b"\r")
        await asyncio.sleep(0.01)
    terminal.feed(b"\x04")
    await asyncio.wait_for(run_task, timeout=2)


async def test_session_streams_deltas() -> None:
    provider = _ScriptedProvider([_text_turn("Hello")])
    agent = Agent(provider=provider, tools=_NoTools())
    terminal = FakeTerminal()
    session = InteractiveSession(agent, terminal=terminal)

    await _run_scripted(session, terminal, ["hi"])

    assert "Hello" in terminal.output


async def test_session_carries_messages_across_turns() -> None:
    provider = _ScriptedProvider([_text_turn("first"), _text_turn("second")])
    agent = Agent(provider=provider, tools=_NoTools())
    terminal = FakeTerminal()
    session = InteractiveSession(agent, terminal=terminal)

    await _run_scripted(session, terminal, ["one", "two"])

    assert len(provider.seen_initial_messages) == 2
    assert len(provider.seen_initial_messages[0]) == 1
    second_turn_messages = provider.seen_initial_messages[1]
    assert any(
        isinstance(m, AssistantMessage)
        and m.content
        and isinstance(m.content[0], TextContent)
        and m.content[0].text == "first"
        for m in second_turn_messages
    )


async def test_session_renders_tool_execution() -> None:
    provider = _ScriptedProvider([_tool_call_turn("call-1", "echo"), _text_turn("done")])
    agent = Agent(provider=provider, tools=_EchoTool())
    terminal = FakeTerminal()
    session = InteractiveSession(agent, terminal=terminal)

    await _run_scripted(session, terminal, ["run tool"])

    assert "tool: echo" in terminal.output
    assert "-> ok" in terminal.output


async def test_session_ignores_blank_input() -> None:
    provider = _ScriptedProvider([_text_turn("Hello")])
    agent = Agent(provider=provider, tools=_NoTools())
    terminal = FakeTerminal()
    session = InteractiveSession(agent, terminal=terminal)

    await _run_scripted(session, terminal, ["", "  ", "hi"])

    assert len(provider.seen_initial_messages) == 1
