from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from agent import Agent
from agent.entries import EntryRef
from agent.sessions import SessionManager, load_session
from coding_agent.modes.interactive import InteractiveSession
from upstream.models import (
    AssistantMessage,
    AssistantMessageEvent,
    Message,
    TextContent,
    ToolCall,
    UserMessage,
)
from upstream.providers.base import ModelProvider
from upstream.types import AgentToolResult, ToolSpec

from tests.tui.fake_terminal import FakeTerminal


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
        *,
        reasoning: object | None = None,
        abort_event: object | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.seen_initial_messages.append(list(messages))
        for event in next(self._turns):
            yield event


class _NoTools:
    def specs(self) -> list[ToolSpec]:
        return []

    async def run(self, call, ctx):  # pragma: no cover - not expected to run
        raise AssertionError("no tool should run")

    def execution_mode_for(self, name: str) -> str | None:
        return None


class _EchoTool:
    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name="echo", description="Echo.", parameters={"type": "object"})]

    async def run(self, call, ctx):
        return AgentToolResult.text("echoed"), False

    def execution_mode_for(self, name: str) -> str | None:
        return None


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


async def test_session_persists_turns_to_disk(tmp_path: Path) -> None:
    provider = _ScriptedProvider([_text_turn("Hello")])
    agent = Agent(provider=provider, tools=_NoTools())
    terminal = FakeTerminal()
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    session = InteractiveSession(agent, terminal=terminal, session=manager)

    await _run_scripted(session, terminal, ["hi"])

    loaded = load_session(path)
    assert [message.role for message in loaded.messages] == ["user", "assistant"]
    records = _read_jsonl(path)
    assert any(record["type"] == "event" for record in records)


async def test_session_records_model_change_on_switch(tmp_path: Path) -> None:
    from upstream.registry import ModelInfo

    provider = _ScriptedProvider([])
    new_provider = _ScriptedProvider([])
    terminal = FakeTerminal()
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)

    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        return new_provider

    agent = Agent(provider=provider, tools=_NoTools())
    session = InteractiveSession(
        agent,
        terminal=terminal,
        session=manager,
        models=[ModelInfo(id="b", name="Model B", api="mock", provider="p1")],
        on_model_change=on_model_change,
        model_label="p1/a",
    )
    await session._apply_model_change("p1", "b")

    records = _read_jsonl(path)
    model_changes = [record for record in records if record["type"] == "model_change"]
    assert model_changes == [
        {
            "type": "model_change",
            "provider": "p1",
            "modelId": "b",
            "id": model_changes[0]["id"],
            "parentId": None,
            "timestamp": model_changes[0]["timestamp"],
        }
    ]


async def test_session_replays_initial_entries_into_transcript() -> None:
    provider = _ScriptedProvider([_text_turn("second")])
    agent = Agent(provider=provider, tools=_NoTools())
    terminal = FakeTerminal()
    initial_entries = [
        EntryRef(id="1", message=UserMessage(content="hi there")),
        EntryRef(
            id="2",
            message=AssistantMessage(content=[TextContent(text="hello back")], stopReason="stop"),
        ),
    ]

    session = InteractiveSession(agent, terminal=terminal, initial_entries=initial_entries)

    assert len(session.messages) == 2
    rendered = "\n".join(session.transcript.render(80))
    assert "hi there" in rendered
    assert "hello back" in rendered
