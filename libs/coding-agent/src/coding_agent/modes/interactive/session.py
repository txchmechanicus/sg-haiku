from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable

from agent import Agent
from agent.entries import EntryRef
from agent.events import AgentEvent
from agent.sessions import SessionManager
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.theme import Theme
from textual.widgets import Input, Static
from textual.widgets.option_list import Option
from upstream.models import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from upstream.providers import ModelProvider
from upstream.registry import ModelInfo

from coding_agent.modes.interactive import commands
from coding_agent.modes.interactive.components import (
    AssistantMessageComponent,
    PromptBar,
    ToolExecutionComponent,
)
from coding_agent.modes.interactive.theme import STARGAZER_DARK


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextContent))


class HaikuApp(App[None]):
    """Owns the agent, conversation state, and the Textual widget tree for a single
    interactive-mode run. Mirrors the event-wiring slice of Pi's
    `interactive-mode.ts`: session persistence and slash commands (/help, /quit, /clear,
    /model) are implemented; a single built-in dark theme is applied, no theme
    switching/autocomplete are implemented.
    """

    CSS = """
    #transcript {
        height: 1fr;
        padding: 0 1;
    }
    .user-message {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }
    """

    # `priority=True` so these reach the transcript even while the prompt bar's `Input`
    # holds focus (Input doesn't bind PageUp/PageDown/Home/End for its own single-line
    # editing, but focused widgets normally get first refusal at key events).
    BINDINGS = [
        Binding("pageup", "scroll_transcript_up", "Scroll up", show=False, priority=True),
        Binding("pagedown", "scroll_transcript_down", "Scroll down", show=False, priority=True),
        Binding("ctrl+home", "scroll_transcript_home", "Scroll to top", show=False, priority=True),
        Binding("ctrl+end", "scroll_transcript_end", "Scroll to bottom", show=False, priority=True),
    ]

    def __init__(
        self,
        agent: Agent,
        *,
        use_tools: bool = True,
        system_prompt: str | None = None,
        models: list[ModelInfo] | None = None,
        on_model_change: Callable[[str, str], Awaitable[ModelProvider]] | None = None,
        model_label: str | None = None,
        session: SessionManager | None = None,
        initial_entries: list[EntryRef] | None = None,
        compaction_message: SystemMessage | None = None,
        new_session_factory: Callable[[], SessionManager] | None = None,
        themes: list[Theme] | None = None,
        theme_name: str | None = None,
    ) -> None:
        super().__init__()
        self.register_theme(STARGAZER_DARK)
        for theme in themes or []:
            self.register_theme(theme)
        self._unknown_theme_name: str | None = None
        if theme_name is not None and theme_name in self.available_themes:
            self.theme = theme_name
        else:
            self.theme = STARGAZER_DARK.name
            if theme_name is not None:
                self._unknown_theme_name = theme_name
        self.agent = agent
        self.use_tools = use_tools
        self.system_prompt = system_prompt
        self.models = models
        self.on_model_change = on_model_change
        self.model_label = model_label or "mock"
        self.session = session
        self.new_session_factory = new_session_factory

        entries = initial_entries or []
        self.messages: list[Message]
        if compaction_message is not None:
            self.messages = [compaction_message, *(ref.message for ref in entries)]
        else:
            self.messages = [ref.message for ref in entries]
        self._initial_entries = entries

        self._tool_components: dict[str, ToolExecutionComponent] = {}
        self._current_assistant: AssistantMessageComponent | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self.quit_requested = False
        # "Sticky" auto-scroll: true while the user is following the bottom of the
        # transcript. Kept in sync with the *real* scroll position via `watch()` below —
        # not inferred by comparing `scroll_y`/`max_scroll_y` at arbitrary points in time,
        # which raced against Textual's own (deferred) layout passes and made auto-scroll
        # stop following after the very first streamed delta of a response.
        self._follow_bottom = True

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield PromptBar(status=self._status_line())

    def on_mount(self) -> None:
        if self._unknown_theme_name is not None:
            self.transcript.mount(
                Static(
                    f"Unknown theme '{self._unknown_theme_name}', using '{STARGAZER_DARK.name}'.",
                    classes="error",
                )
            )
        self._replay_history(self._initial_entries)
        self.prompt_bar.input.focus()
        # Fires on *any* change to the transcript's scroll position — keyboard (PageUp/
        # Down, Ctrl+Home/End), mouse wheel, scrollbar drag, or our own programmatic
        # `scroll_end()` calls alike — so `_follow_bottom` always reflects where the
        # viewport actually is, regardless of what moved it.
        self.watch(self.transcript, "scroll_y", self._sync_follow_bottom, init=False)

    @property
    def transcript(self) -> VerticalScroll:
        return self.query_one("#transcript", VerticalScroll)

    @property
    def prompt_bar(self) -> PromptBar:
        return self.query_one(PromptBar)

    def _status_line(self) -> str:
        return f"model: {self.model_label} · Ctrl-C cancel · Ctrl-D exit · /help for commands"

    def _sync_follow_bottom(self) -> None:
        transcript = self.transcript
        self._follow_bottom = transcript.scroll_y >= transcript.max_scroll_y - 1

    def _scroll_to_end(self, *, force: bool = False) -> None:
        # Don't fight a manual scroll: if the user has scrolled up to read earlier
        # messages, an in-progress response streaming in shouldn't yank them back down on
        # every delta (that's what made scrolling feel "broken" — any attempt to scroll up
        # during/around an active response was immediately overridden). Only auto-follow
        # when already at the bottom, unless `force=True` for a direct result of the
        # user's own action (submitting a message, running a command).
        if force:
            self._follow_bottom = True
        if not self._follow_bottom:
            return
        # `scroll_end()` already defers itself via its own internal `call_after_refresh`
        # (it needs a completed layout pass to know the real `max_scroll_y`) — wrapping
        # this call in *another* `call_after_refresh` double-defers it, and under a
        # continuous stream of deltas that second callback could keep losing the race to
        # newer ones and never actually fire. Call it directly.
        self.transcript.scroll_end(animate=False)

    def action_scroll_transcript_up(self) -> None:
        self.transcript.scroll_page_up()

    def action_scroll_transcript_down(self) -> None:
        self.transcript.scroll_page_down()

    def action_scroll_transcript_home(self) -> None:
        self.transcript.scroll_home()

    def action_scroll_transcript_end(self) -> None:
        self.transcript.scroll_end()

    def _replay_history(self, entries: list[EntryRef]) -> None:
        """Populates the transcript with a resumed/forked session's prior turns before any
        new input is fed, so `self.messages` (used as `initial_messages` for the next
        `agent.run()` call) and the on-screen transcript agree from the start."""
        pending_tool_components: dict[str, ToolExecutionComponent] = {}
        for ref in entries:
            message = ref.message
            if isinstance(message, UserMessage):
                text = (
                    message.content
                    if isinstance(message.content, str)
                    else "".join(
                        part.text for part in message.content if isinstance(part, TextContent)
                    )
                )
                self.transcript.mount(Static(f"> {text}", classes="user-message"))
            elif isinstance(message, AssistantMessage):
                component = AssistantMessageComponent()
                component.text = _assistant_text(message)
                component.update(component.text)
                self.transcript.mount(component)
                for part in message.content:
                    if getattr(part, "type", None) == "toolCall":
                        tool_component = ToolExecutionComponent(part.name, part.arguments)
                        pending_tool_components[part.id] = tool_component
                        self.transcript.mount(tool_component)
            elif isinstance(message, ToolResultMessage):
                tool_component = pending_tool_components.pop(message.toolCallId, None)
                if tool_component is not None:
                    tool_component.finish(is_error=message.isError, result=message.content)
        self._scroll_to_end(force=True)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        await self._handle_submitted_text(text)

    async def _handle_submitted_text(self, text: str) -> None:
        if not text:
            return

        if await commands.dispatch(self, text):
            if self.quit_requested:
                self.exit()
            return

        self.transcript.mount(Static(f"> {text}", classes="user-message"))
        self._scroll_to_end(force=True)

        self._turn_task = asyncio.ensure_future(self._run_turn(text))
        try:
            await self._turn_task
        except asyncio.CancelledError:
            pass
        self._turn_task = None

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_command_hints(event.value)

    def _update_command_hints(self, text: str) -> None:
        """Shows a filtered dropdown of matching slash commands while the input holds an
        in-progress command name (`/` with no space yet) — hidden the moment there's a
        space (an argument has started) or the text no longer starts with `/`."""
        hints = self.prompt_bar.hints
        matches = []
        if text.startswith("/") and " " not in text:
            query = text[1:].lower()
            matches = [
                command
                for command in sorted(commands.COMMANDS.values(), key=lambda c: c.name)
                if command.name.startswith(query)
            ]
        if not matches:
            hints.remove_class("-visible")
            hints.clear_options()
            return
        hints.clear_options()
        hints.add_options(
            Option(f"/{command.name} — {command.description}", id=command.name)
            for command in matches
        )
        hints.highlighted = 0
        hints.add_class("-visible")

    async def _accept_command_hint(self) -> None:
        hints = self.prompt_bar.hints
        option = hints.highlighted
        command_id = hints.get_option_at_index(option).id if option is not None else None
        hints.remove_class("-visible")
        hints.clear_options()
        if command_id is None:
            return
        command = commands.COMMANDS.get(command_id)
        input_widget = self.prompt_bar.input
        if command is not None and not command.takes_args:
            # No arguments to type — running the command immediately (instead of
            # filling the text and waiting for a second Enter) matches how a
            # no-argument command behaves everywhere else: select it and it runs.
            input_widget.value = ""
            await self._handle_submitted_text(f"/{command_id}")
            return
        input_widget.value = f"/{command_id} "
        input_widget.cursor_position = len(input_widget.value)

    async def on_key(self, event: events.Key) -> None:
        hints = self.prompt_bar.hints
        if hints.has_class("-visible"):
            if event.key == "up":
                event.stop()
                event.prevent_default()
                hints.action_cursor_up()
                return
            if event.key == "down":
                event.stop()
                event.prevent_default()
                hints.action_cursor_down()
                return
            if event.key in ("tab", "enter"):
                event.stop()
                event.prevent_default()
                await self._accept_command_hint()
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                hints.remove_class("-visible")
                hints.clear_options()
                return
        if event.key == "ctrl+c":
            event.stop()
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
            return
        if event.key == "ctrl+d" and not self.prompt_bar.input.value:
            event.stop()
            self.quit_requested = True
            self.exit()

    async def _run_turn(self, prompt: str) -> None:
        self._current_assistant = None
        async for event in self.agent.run(
            prompt,
            initial_messages=self.messages,
            system_prompt=self.system_prompt,
            use_tools=self.use_tools,
        ):
            self._handle_event(event)
        self._current_assistant = None
        self._scroll_to_end()

    def _current_assistant_component(self) -> AssistantMessageComponent:
        # Created lazily (rather than one component up front for the whole turn) so a
        # tool-call round-trip inside a single `agent.run()` call renders in true
        # chronological order: any text before a tool call, then the tool line, then a
        # fresh component for text that follows — instead of every iteration's text
        # collapsing into one block mounted before any tool call ever ran.
        if self._current_assistant is None:
            self._current_assistant = AssistantMessageComponent()
            self.transcript.mount(self._current_assistant)
        return self._current_assistant

    def _handle_event(self, event: AgentEvent) -> None:
        if self.session is not None:
            self.session.record_event(event)
            if event.type == "message_end" and event.message is not None:
                self.session.record_message(event.message)
        if event.type == "message_update":
            assistant_event = event.assistantMessageEvent
            is_text_delta = assistant_event is not None and assistant_event.type == "text_delta"
            if is_text_delta and assistant_event.delta:
                self._current_assistant_component().update_delta(assistant_event.delta)
                self._scroll_to_end()
        elif event.type == "tool_execution_start":
            component = ToolExecutionComponent(event.toolName or "", event.args)
            self._tool_components[event.toolCallId or ""] = component
            self.transcript.mount(component)
            self._current_assistant = None
            self._scroll_to_end()
        elif event.type == "tool_execution_end":
            component = self._tool_components.pop(event.toolCallId or "", None)
            if component is not None:
                component.finish(is_error=bool(event.isError), result=event.result)
                self._scroll_to_end()
        elif event.type == "message_end" and isinstance(event.message, AssistantMessage):
            # Not every provider streams `text_delta`s — a provider that returns its whole
            # response (or error) in one shot only shows up here. Reconcile the component
            # with the authoritative final text so nothing renders as permanently blank.
            final_text = _assistant_text(event.message)
            if final_text:
                component = self._current_assistant_component()
                if component.text != final_text:
                    component.text = final_text
                    component.update(final_text)
                if event.message.stopReason == "error":
                    component.add_class("error")
            self._current_assistant = None
        elif event.type == "agent_end" and event.messages is not None:
            self.messages = event.messages

    def start_model_switch(self, provider_id: str, model_id: str) -> None:
        self.run_worker(self._apply_model_change(provider_id, model_id), exclusive=True)

    async def _apply_model_change(self, provider_id: str, model_id: str) -> None:
        assert self.on_model_change is not None
        label = f"{provider_id}/{model_id}"
        try:
            new_provider = await self.on_model_change(provider_id, model_id)
        except Exception as exc:  # noqa: BLE001 - surface any provider-build failure to the user
            self.transcript.mount(Static(f"Failed to switch to {label}: {exc}", classes="error"))
        else:
            self.agent.provider = new_provider
            self.model_label = label
            self.prompt_bar.set_status(self._status_line())
            self.transcript.mount(Static(f"Switched to {label}.", classes="success"))
            if self.session is not None:
                self.session.record_model_change(provider=provider_id, model_id=model_id)
        self._scroll_to_end(force=True)


async def run_interactive(
    agent: Agent,
    *,
    use_tools: bool = True,
    system_prompt: str | None = None,
    models: list[ModelInfo] | None = None,
    on_model_change: Callable[[str, str], Awaitable[ModelProvider]] | None = None,
    model_label: str | None = None,
    session: SessionManager | None = None,
    initial_entries: list[EntryRef] | None = None,
    compaction_message: SystemMessage | None = None,
    new_session_factory: Callable[[], SessionManager] | None = None,
    themes: list[Theme] | None = None,
    theme_name: str | None = None,
) -> None:
    # Unlike the old hand-rolled terminal driver (which raised cleanly from
    # `ProcessTerminal.enter_raw_mode()`), Textual's own driver doesn't reject a non-tty
    # stdin — it just starts drawing into whatever fd it's given. Guard explicitly so a
    # piped/redirected invocation fails fast with a clear message instead of hanging.
    try:
        is_tty = os.isatty(sys.stdin.fileno())
    except (OSError, ValueError):
        is_tty = False
    if not is_tty:
        raise ValueError("stdin is not a terminal; interactive mode requires a real tty.")
    app = HaikuApp(
        agent,
        use_tools=use_tools,
        system_prompt=system_prompt,
        models=models,
        on_model_change=on_model_change,
        model_label=model_label,
        session=session,
        initial_entries=initial_entries,
        compaction_message=compaction_message,
        new_session_factory=new_session_factory,
        themes=themes,
        theme_name=theme_name,
    )
    await app.run_async()
