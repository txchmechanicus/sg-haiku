from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from agent import Agent
from agent.sessions import SessionManager
from coding_agent.modes.interactive import HaikuApp
from textual.pilot import Pilot
from textual.widgets import Static
from upstream.models import AssistantMessageEvent, Message, UserMessage
from upstream.providers.base import ModelProvider
from upstream.registry import ModelInfo
from upstream.types import ToolSpec


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _NoStreamProvider(ModelProvider):
    """Fails the test if the agent is ever actually invoked."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: object | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        raise AssertionError("agent should not run for a slash command")
        yield  # pragma: no cover - unreachable, keeps this an async generator


class _NoTools:
    def specs(self) -> list[ToolSpec]:
        return []

    async def run(self, call, ctx):  # pragma: no cover - not expected to run
        raise AssertionError("no tool should run")

    def execution_mode_for(self, name: str) -> str | None:
        return None


def _models() -> list[ModelInfo]:
    return [
        ModelInfo(id="a", name="Model A", api="mock", provider="p1"),
        ModelInfo(id="b", name="Model B", api="mock", provider="p1"),
    ]


def _new_app(
    *, models=None, on_model_change=None, session=None, new_session_factory=None
) -> HaikuApp:
    agent = Agent(provider=_NoStreamProvider(), tools=_NoTools())
    return HaikuApp(
        agent,
        models=models,
        on_model_change=on_model_change,
        model_label="p1/a",
        session=session,
        new_session_factory=new_session_factory,
    )


async def _submit(pilot: Pilot, text: str) -> None:
    if text:
        await pilot.press(*text)
    await pilot.press("enter")
    await pilot.pause()


def _rendered_statics(app: HaikuApp) -> list[str]:
    return [str(widget.render()) for widget in app.query(Static)]


async def test_help_lists_registered_commands() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await _submit(pilot, "/help")

        rendered = "\n".join(_rendered_statics(app))
        assert "/help" in rendered
        assert "/model" in rendered
        assert "/quit" in rendered
        assert "/clear" in rendered


async def test_unknown_command_does_not_reach_agent() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await _submit(pilot, "/nope")

        assert any("Unknown command: /nope" in text for text in _rendered_statics(app))


async def test_quit_ends_the_session() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await _submit(pilot, "/quit")

    assert app.quit_requested is True


async def test_clear_empties_transcript_and_messages() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        app.messages = [UserMessage(content="placeholder")]
        app.transcript.mount(Static("placeholder"))
        await pilot.pause()

        await _submit(pilot, "/clear")

        assert app.messages == []
        assert list(app.transcript.children) == []


async def test_model_unavailable_without_config() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await _submit(pilot, "/model")

        assert any(
            "Model switching is not available." in text for text in _rendered_statics(app)
        )


async def test_model_select_switches_provider() -> None:
    calls: list[tuple[str, str]] = []
    new_provider = _NoStreamProvider()

    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        calls.append((provider_id, model_id))
        return new_provider

    app = _new_app(models=_models(), on_model_change=on_model_change)
    async with app.run_test() as pilot:
        await _submit(pilot, "/model")
        await pilot.press("down")  # to the second model
        await pilot.press("enter")  # confirm selection
        await pilot.pause()

        assert calls == [("p1", "b")]
        assert app.agent.provider is new_provider
        assert app.model_label == "p1/b"
        assert any("Switched to p1/b." in text for text in _rendered_statics(app))


async def test_model_select_cancel_leaves_provider_untouched() -> None:
    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        raise AssertionError("on_model_change should not run on cancel")

    app = _new_app(models=_models(), on_model_change=on_model_change)
    original_provider = app.agent.provider
    async with app.run_test() as pilot:
        await _submit(pilot, "/model")
        await pilot.press("escape")
        await pilot.pause()

        assert app.agent.provider is original_provider


async def test_model_select_records_model_change_when_session_attached(tmp_path: Path) -> None:
    new_provider = _NoStreamProvider()

    async def on_model_change(provider_id: str, model_id: str) -> ModelProvider:
        return new_provider

    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    app = _new_app(models=_models(), on_model_change=on_model_change, session=manager)

    async with app.run_test() as pilot:
        await _submit(pilot, "/model")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

    records = _read_jsonl(path)
    model_changes = [r for r in records if r["type"] == "model_change"]
    assert model_changes[-1]["provider"] == "p1"
    assert model_changes[-1]["modelId"] == "b"


async def test_clear_starts_a_new_session_file(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    paths = iter([second_path])
    first_manager = SessionManager.create(explicit_path=first_path, session_id="s1", cwd=tmp_path)

    def new_session_factory() -> SessionManager:
        return SessionManager.create(explicit_path=next(paths), session_id="s2", cwd=tmp_path)

    app = _new_app(session=first_manager, new_session_factory=new_session_factory)

    async with app.run_test() as pilot:
        app.messages = [UserMessage(content="placeholder")]
        app.transcript.mount(Static("placeholder"))
        await pilot.pause()

        await _submit(pilot, "/clear")

        assert app.session is not first_manager
        assert app.messages == []
        assert list(app.transcript.children) == []
    assert second_path.exists()


async def test_typing_slash_shows_filtered_command_hints() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()

        hints = app.prompt_bar.hints
        assert hints.has_class("-visible")
        assert hints.option_count == 5  # help, quit, clear, model, theme

        await pilot.press("c")
        await pilot.pause()

        assert hints.option_count == 1
        assert hints.get_option_at_index(0).id == "clear"


async def test_hints_hide_once_a_space_or_no_match_is_typed() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await pilot.press(*"/clear")
        await pilot.pause()
        assert app.prompt_bar.hints.has_class("-visible")

        await pilot.press(" ")
        await pilot.pause()
        assert not app.prompt_bar.hints.has_class("-visible")

        await pilot.press(*"/nope")
        await pilot.pause()
        assert not app.prompt_bar.hints.has_class("-visible")


async def test_tab_immediately_runs_a_no_args_command_hint() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        app.messages = [UserMessage(content="placeholder")]
        app.transcript.mount(Static("placeholder"))
        await pilot.pause()

        await pilot.press(*"/cl")
        await pilot.pause()

        await pilot.press("tab")
        await pilot.pause()

        # /clear takes no arguments, so selecting it (Tab or Enter) runs it right
        # away instead of just filling "/clear " and waiting for a second press.
        assert app.prompt_bar.input.value == ""
        assert not app.prompt_bar.hints.has_class("-visible")
        assert app.messages == []


async def test_arrow_keys_navigate_hints_and_enter_runs_immediately() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()
        assert app.prompt_bar.hints.highlighted == 0  # "clear" (alphabetically first)

        await pilot.press("down")
        await pilot.pause()
        assert app.prompt_bar.hints.highlighted == 1  # "help"

        await pilot.press("enter")
        await pilot.pause()

        # /help takes no arguments, so a single Enter runs it immediately.
        assert app.prompt_bar.input.value == ""
        assert not app.prompt_bar.hints.has_class("-visible")
        assert any("/help" in text for text in _rendered_statics(app))


async def test_hint_selection_fills_text_for_a_command_that_takes_args(
    monkeypatch,
) -> None:
    from coding_agent.modes.interactive import commands as commands_module

    async def _noop(app, args):  # pragma: no cover - never actually dispatched here
        raise AssertionError("should not run: only completion is expected")

    fake_command = commands_module.SlashCommand(
        "resume", "Resume a session by id.", _noop, takes_args=True
    )
    monkeypatch.setitem(commands_module.COMMANDS, "resume", fake_command)

    app = _new_app()
    async with app.run_test() as pilot:
        await pilot.press(*"/res")
        await pilot.pause()

        await pilot.press("tab")
        await pilot.pause()

        assert app.prompt_bar.input.value == "/resume "
        assert not app.prompt_bar.hints.has_class("-visible")


async def test_escape_dismisses_hints_without_clearing_input() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        await pilot.press(*"/cl")
        await pilot.pause()
        assert app.prompt_bar.hints.has_class("-visible")

        await pilot.press("escape")
        await pilot.pause()

        assert not app.prompt_bar.hints.has_class("-visible")
        assert app.prompt_bar.input.value == "/cl"


async def test_theme_command_lists_and_switches_themes() -> None:
    app = _new_app()
    async with app.run_test() as pilot:
        original_theme = app.theme
        assert "nord" in app.available_themes  # a Textual built-in, free of charge

        await _submit(pilot, "/theme")
        await pilot.pause()

        # The picker lists every registered theme, sorted; pick something other than
        # whatever's currently active so the assertion below is meaningful regardless
        # of which theme happens to be the default.
        screen = app.screen
        option_list = screen.query_one("OptionList")
        target_index = next(
            i
            for i in range(option_list.option_count)
            if option_list.get_option_at_index(i).id != original_theme
        )
        while option_list.highlighted != target_index:
            await pilot.press("down")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.theme != original_theme
