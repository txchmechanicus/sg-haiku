from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from agent import Agent
from agent.entries import EntryRef
from agent.sessions import SessionManager, load_session
from coding_agent.modes.interactive import HaikuApp
from coding_agent.modes.interactive.components import ToolExecutionComponent
from textual.pilot import Pilot
from textual.widgets import Static
from upstream.models import (
    AssistantMessage,
    AssistantMessageEvent,
    Message,
    TextContent,
    ToolCall,
    UserMessage,
)
from upstream.providers.base import ModelProvider
from upstream.registry import ModelInfo
from upstream.types import AgentToolResult, ToolSpec


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


def _error_turn(error_text: str) -> list[AssistantMessageEvent]:
    """Some providers (or a transport failure) surface the whole response — including
    errors — in a single non-`text_delta` event rather than streaming it. Regression
    coverage for a bug where such turns rendered as permanently blank."""
    message = AssistantMessage(
        content=[TextContent(text=error_text)], stopReason="error", errorMessage=error_text
    )
    return [
        AssistantMessageEvent(type="start", partial=message),
        AssistantMessageEvent(type="error", partial=message),
        AssistantMessageEvent(type="done", reason="error", message=message),
    ]


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


async def _submit(pilot: Pilot, text: str) -> None:
    if text:
        await pilot.press(*text)
    await pilot.press("enter")
    await pilot.pause()
    app = pilot.app
    # Submission now runs as a detached task (see session.py's on_input_submitted) so
    # that Ctrl-C can interrupt it -- wait for it to actually finish before returning,
    # since the scripted providers used here resolve almost instantly.
    for _ in range(200):
        if app._submission_task is None or app._submission_task.done():
            break
        await asyncio.sleep(0.01)
    await pilot.pause()


def _rendered_statics(app: HaikuApp) -> list[str]:
    return [str(widget.render()) for widget in app.query(Static)]


async def test_session_streams_deltas() -> None:
    provider = _ScriptedProvider([_text_turn("Hello")])
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test() as pilot:
        await _submit(pilot, "hi")

        assert any("Hello" in text for text in _rendered_statics(app))


async def test_session_carries_messages_across_turns() -> None:
    provider = _ScriptedProvider([_text_turn("first"), _text_turn("second")])
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test() as pilot:
        await _submit(pilot, "one")
        await _submit(pilot, "two")

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
    app = HaikuApp(agent)

    async with app.run_test() as pilot:
        await _submit(pilot, "run tool")

        tool_components = list(app.query(ToolExecutionComponent))
        assert len(tool_components) == 1
        rendered = str(tool_components[0].render())
        assert rendered == '⏺ echo(x=1)\n  ⎿  echoed'
        assert tool_components[0].has_class("ok")

        # The tool line and the final "done" text should appear as separate,
        # chronologically ordered components — not merged into one block mounted
        # before the tool ever ran.
        statics = list(app.query(Static))
        tool_index = statics.index(tool_components[0])
        assert any(
            "done" in str(s.render()) for s in statics[tool_index + 1 :]
        ), "final assistant text should render after the tool line"


async def test_session_ignores_blank_input() -> None:
    provider = _ScriptedProvider([_text_turn("Hello")])
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test() as pilot:
        await _submit(pilot, "")
        await _submit(pilot, "  ")
        await _submit(pilot, "hi")

    assert len(provider.seen_initial_messages) == 1


async def test_session_persists_turns_to_disk(tmp_path: Path) -> None:
    provider = _ScriptedProvider([_text_turn("Hello")])
    agent = Agent(provider=provider, tools=_NoTools())
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    app = HaikuApp(agent, session=manager)

    async with app.run_test() as pilot:
        await _submit(pilot, "hi")

    loaded = load_session(path)
    assert [message.role for message in loaded.messages] == ["user", "assistant"]
    records = _read_jsonl(path)
    assert any(record["type"] == "event" for record in records)


async def test_session_records_model_change_on_switch(tmp_path: Path) -> None:
    provider = _ScriptedProvider([])
    new_provider = _ScriptedProvider([])
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)

    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        return new_provider

    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(
        agent,
        session=manager,
        models=[ModelInfo(id="b", name="Model B", api="mock", provider="p1")],
        on_model_change=on_model_change,
        model_label="p1/a",
    )

    async with app.run_test():
        await app._apply_model_change("p1", "b")

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
    initial_entries = [
        EntryRef(id="1", message=UserMessage(content="hi there")),
        EntryRef(
            id="2",
            message=AssistantMessage(content=[TextContent(text="hello back")], stopReason="stop"),
        ),
    ]
    app = HaikuApp(agent, initial_entries=initial_entries)

    async with app.run_test():
        assert len(app.messages) == 2
        rendered = _rendered_statics(app)
        assert any("hi there" in text for text in rendered)
        assert any("hello back" in text for text in rendered)


async def test_session_renders_non_streamed_error_response() -> None:
    provider = _ScriptedProvider([_error_turn("Server error '502 Bad Gateway'")])
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test() as pilot:
        await _submit(pilot, "hi")

        assert any("502 Bad Gateway" in text for text in _rendered_statics(app))


async def test_transcript_auto_scrolls_as_messages_grow() -> None:
    turns = [_text_turn(f"Response number {i}\n" * 3) for i in range(20)]
    provider = _ScriptedProvider(turns)
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test(size=(100, 24)) as pilot:
        for i in range(20):
            await _submit(pilot, f"msg{i}")

        transcript = app.transcript
        assert transcript.scroll_y == transcript.max_scroll_y


async def test_transcript_scrolls_manually_while_input_has_focus() -> None:
    turns = [_text_turn(f"Response number {i}\n" * 3) for i in range(20)]
    provider = _ScriptedProvider(turns)
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test(size=(100, 24)) as pilot:
        for i in range(20):
            await _submit(pilot, f"msg{i}")

        transcript = app.transcript
        assert app.focused is app.prompt_bar.input
        before = transcript.scroll_y

        await pilot.press("pageup")
        await pilot.pause()
        assert transcript.scroll_y < before

        await pilot.press("ctrl+end")
        await pilot.pause()
        assert transcript.scroll_y == transcript.max_scroll_y


async def test_passive_updates_do_not_override_a_manual_scroll() -> None:
    turns = [_text_turn(f"Response number {i}\n" * 3) for i in range(20)]
    provider = _ScriptedProvider(turns)
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)

    async with app.run_test(size=(100, 24)) as pilot:
        for i in range(20):
            await _submit(pilot, f"msg{i}")

        transcript = app.transcript
        await pilot.press("pageup")
        await pilot.pause()
        scrolled_position = transcript.scroll_y
        assert scrolled_position < transcript.max_scroll_y

        # A passive content update (not a direct user action) must not fight a manual
        # scroll — this was the actual reported bug: unconditional auto-scroll made
        # scrolling up feel completely broken, since any new content snapped the view
        # back down immediately.
        app._scroll_to_end()
        await pilot.pause()
        assert transcript.scroll_y == scrolled_position

        # Sending a new message is a direct user action and should jump back to the
        # bottom, matching normal chat-app "sticky scroll" behavior.
        await _submit(pilot, "back to bottom")
        assert transcript.scroll_y == transcript.max_scroll_y


class _AbortAwareProvider(ModelProvider):
    """Actually checks `abort_event`, like every real provider does -- simulates the
    cooperative-cancellation path (Ctrl-C should stop this promptly on its own, no
    watchdog hard-cancel required)."""

    def __init__(self) -> None:
        self.saw_abort = False

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        message = AssistantMessage(content=[TextContent(text="")], stopReason="stop")
        yield AssistantMessageEvent(type="start", partial=message)
        assert abort_event is not None
        for _ in range(200):
            if abort_event.is_set():
                self.saw_abort = True
                final = AssistantMessage(content=[TextContent(text="")], stopReason="aborted")
                yield AssistantMessageEvent(type="done", reason="aborted", message=final)
                return
            await asyncio.sleep(0.01)
        raise AssertionError("abort_event was never set")


class _HangingProvider(ModelProvider):
    """Never yields and never checks the abort event -- simulates a stalled connection
    that only the watchdog's hard `task.cancel()` can stop."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        await asyncio.sleep(3600)
        yield AssistantMessageEvent(type="done", reason="stop")  # pragma: no cover


async def _wait_until(predicate, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was never met")


async def test_ctrl_c_calls_agent_abort_and_lets_a_cooperative_provider_finish_cleanly() -> None:
    provider = _AbortAwareProvider()
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await _wait_until(lambda: app._turn_task is not None)

        await pilot.press("ctrl+c")
        await _wait_until(lambda: app._turn_task is None)

        assert provider.saw_abort is True


async def test_ctrl_c_hard_cancels_a_stuck_turn_after_grace_period() -> None:
    provider = _HangingProvider()
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await _wait_until(lambda: app._turn_task is not None)

        await pilot.press("ctrl+c")

        # The provider ignores the abort signal entirely, so the turn must still be
        # running right after Ctrl-C -- only the watchdog's grace-period fallback will
        # eventually stop it.
        await asyncio.sleep(0.2)
        assert app._turn_task is not None and not app._turn_task.done()

        await _wait_until(lambda: app._turn_task is None, timeout=3.0)


async def test_second_ctrl_c_force_cancels_immediately() -> None:
    provider = _HangingProvider()
    agent = Agent(provider=provider, tools=_NoTools())
    app = HaikuApp(agent)
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await _wait_until(lambda: app._turn_task is not None)

        await pilot.press("ctrl+c")
        await asyncio.sleep(0.2)
        assert app._turn_task is not None and not app._turn_task.done()

        # A second Ctrl-C force-cancels right away, well before the ~1.5s watchdog
        # grace period would otherwise kick in.
        started = asyncio.get_event_loop().time()
        await pilot.press("ctrl+c")
        await _wait_until(lambda: app._turn_task is None, timeout=1.0)
        elapsed = asyncio.get_event_loop().time() - started
        assert elapsed < 1.0
