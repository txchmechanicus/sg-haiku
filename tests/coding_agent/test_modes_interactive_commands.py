from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent import Agent
from coding_agent.modes.interactive import InteractiveSession
from tui import Text
from upstream.models import AssistantMessageEvent, Message
from upstream.providers.base import ModelProvider
from upstream.registry import ModelInfo
from upstream.types import ToolSpec

from tests.tui.fake_terminal import FakeTerminal


class _NoStreamProvider(ModelProvider):
    """Fails the test if the agent is ever actually invoked."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        raise AssertionError("agent should not run for a slash command")
        yield  # pragma: no cover - unreachable, keeps this an async generator


class _NoTools:
    def specs(self) -> list[ToolSpec]:
        return []

    async def run(self, call):  # pragma: no cover - not expected to run
        raise AssertionError("no tool should run")


def _models() -> list[ModelInfo]:
    return [
        ModelInfo(id="a", name="Model A", api="mock", provider="p1"),
        ModelInfo(id="b", name="Model B", api="mock", provider="p1"),
    ]


async def _feed_line(terminal: FakeTerminal, text: str) -> None:
    for char in text:
        terminal.feed(char.encode())
    terminal.feed(b"\r")
    await asyncio.sleep(0.01)


async def _feed_raw(terminal: FakeTerminal, data: bytes) -> None:
    terminal.feed(data)
    await asyncio.sleep(0.01)


def _new_session(*, models=None, on_model_change=None) -> tuple[InteractiveSession, FakeTerminal]:
    agent = Agent(provider=_NoStreamProvider(), tools=_NoTools())
    terminal = FakeTerminal()
    session = InteractiveSession(
        agent,
        terminal=terminal,
        models=models,
        on_model_change=on_model_change,
        model_label="p1/a",
    )
    return session, terminal


async def test_help_lists_registered_commands() -> None:
    session, terminal = _new_session()
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/help")
    await _feed_raw(terminal, b"\x04")
    await asyncio.wait_for(run_task, timeout=2)

    assert "/help" in terminal.output
    assert "/model" in terminal.output
    assert "/quit" in terminal.output
    assert "/clear" in terminal.output


async def test_unknown_command_does_not_reach_agent() -> None:
    session, terminal = _new_session()
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/nope")
    await _feed_raw(terminal, b"\x04")
    await asyncio.wait_for(run_task, timeout=2)

    assert "Unknown command: /nope" in terminal.output


async def test_quit_ends_the_session() -> None:
    session, terminal = _new_session()
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/quit")
    await asyncio.wait_for(run_task, timeout=2)

    assert session.quit_requested is True


async def test_clear_empties_transcript_and_messages() -> None:
    session, terminal = _new_session()
    session.messages = ["placeholder"]  # type: ignore[list-item]
    session.transcript.add(Text("placeholder"))
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/clear")
    await _feed_raw(terminal, b"\x04")
    await asyncio.wait_for(run_task, timeout=2)

    assert session.messages == []
    assert session.transcript.children == []


async def test_model_unavailable_without_config() -> None:
    session, terminal = _new_session()
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/model")
    await _feed_raw(terminal, b"\x04")
    await asyncio.wait_for(run_task, timeout=2)

    assert "Model switching is not available." in terminal.output


async def test_model_select_switches_provider() -> None:
    calls: list[tuple[str, str]] = []
    new_provider = _NoStreamProvider()

    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        calls.append((provider_id, model_id))
        return new_provider

    session, terminal = _new_session(models=_models(), on_model_change=on_model_change)
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/model")
    await _feed_raw(terminal, b"\x1b[B")  # down to the second model
    await _feed_raw(terminal, b"\r")  # confirm selection
    await asyncio.sleep(0.05)  # let the scheduled provider-switch task finish
    await _feed_raw(terminal, b"\x04")
    await asyncio.wait_for(run_task, timeout=2)

    assert calls == [("p1", "b")]
    assert session.agent.provider is new_provider
    assert session.model_label == "p1/b"
    assert "Switched to p1/b." in terminal.output


async def test_model_select_cancel_leaves_provider_untouched() -> None:
    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        raise AssertionError("on_model_change should not run on cancel")

    session, terminal = _new_session(models=_models(), on_model_change=on_model_change)
    original_provider = session.agent.provider
    run_task = asyncio.ensure_future(session.run())

    await _feed_line(terminal, "/model")
    await _feed_raw(terminal, b"\x1b")  # escape cancels
    await _feed_raw(terminal, b"\x04")
    await asyncio.wait_for(run_task, timeout=2)

    assert session.agent.provider is original_provider
