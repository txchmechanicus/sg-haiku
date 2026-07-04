from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent import Agent
from agent.events import AgentEvent
from tui import TUI, Container, ProcessTerminal, Terminal, Text
from upstream.models import Message
from upstream.providers import ModelProvider
from upstream.registry import ModelInfo

from coding_agent.modes.interactive import commands
from coding_agent.modes.interactive.components import (
    AssistantMessageComponent,
    Footer,
    ToolExecutionComponent,
)


class InteractiveSession:
    """Owns the agent, conversation state, and the TUI component tree for a
    single interactive-mode run. Mirrors the event-wiring slice of Pi's
    `interactive-mode.ts`: session persistence and theme/session selection
    are not part of this pass, but slash commands (/help, /quit, /clear,
    /model) are.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        use_tools: bool = True,
        system_prompt: str | None = None,
        terminal: Terminal | None = None,
        models: list[ModelInfo] | None = None,
        on_model_change: Callable[[str, str], Awaitable[ModelProvider]] | None = None,
        model_label: str | None = None,
    ) -> None:
        self.agent = agent
        self.use_tools = use_tools
        self.system_prompt = system_prompt
        self.messages: list[Message] = []
        self.models = models
        self.on_model_change = on_model_change
        self.model_label = model_label or "mock"

        self.transcript = Container()
        self.footer = Footer(on_submit=self._on_submit, status=self._status_line())
        self.tui = TUI(terminal if terminal is not None else ProcessTerminal())
        self.tui.add(self.transcript)
        self.tui.add(self.footer)
        self.tui.set_focus(self.footer.editor)
        self.tui.add_input_listener(self._on_global_key)

        self._pending_input: asyncio.Queue[str] = asyncio.Queue()
        self._tool_components: dict[str, ToolExecutionComponent] = {}
        self._turn_task: asyncio.Task[None] | None = None
        self._eof = False
        self.quit_requested = False

    def _status_line(self) -> str:
        return f"model: {self.model_label} · Ctrl-C cancel · Ctrl-D exit · /help for commands"

    def _on_submit(self, text: str) -> None:
        text = text.strip()
        if text:
            self._pending_input.put_nowait(text)

    def _on_global_key(self, data: str) -> bool:
        if data == "\x03":  # Ctrl-C: cancel the in-flight turn only
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
                return True
            return False
        if data == "\x04" and not self.footer.editor.text:  # Ctrl-D on an empty line: exit
            self._eof = True
            self._pending_input.put_nowait("")
            return True
        return False

    async def run(self) -> None:
        self.tui.start()
        try:
            while True:
                text = await self._pending_input.get()
                if self._eof:
                    return

                if await commands.dispatch(self, text):
                    if self.quit_requested:
                        return
                    continue

                self.transcript.add(Text(f"> {text}", style="bold cyan"))
                assistant_component = AssistantMessageComponent()
                self.transcript.add(assistant_component)
                self.tui.request_render(force=True)

                self._turn_task = asyncio.ensure_future(self._run_turn(text, assistant_component))
                try:
                    await self._turn_task
                except asyncio.CancelledError:
                    pass
                self._turn_task = None
        finally:
            self.tui.stop()

    async def _run_turn(self, prompt: str, assistant_component: AssistantMessageComponent) -> None:
        async for event in self.agent.run(
            prompt,
            initial_messages=self.messages,
            system_prompt=self.system_prompt,
            use_tools=self.use_tools,
        ):
            self._handle_event(event, assistant_component)
        self.tui.request_render()

    def _handle_event(
        self, event: AgentEvent, assistant_component: AssistantMessageComponent
    ) -> None:
        if event.type == "message_update":
            assistant_event = event.assistantMessageEvent
            is_text_delta = assistant_event is not None and assistant_event.type == "text_delta"
            if is_text_delta and assistant_event.delta:
                assistant_component.update_delta(assistant_event.delta)
                self.tui.request_render()
        elif event.type == "tool_execution_start":
            component = ToolExecutionComponent(event.toolName or "", event.args)
            self._tool_components[event.toolCallId or ""] = component
            self.transcript.add(component)
            self.tui.request_render(force=True)
        elif event.type == "tool_execution_end":
            component = self._tool_components.pop(event.toolCallId or "", None)
            if component is not None:
                component.finish(is_error=bool(event.isError))
                self.tui.request_render()
        elif event.type == "agent_end" and event.messages is not None:
            self.messages = event.messages

    def start_model_switch(self, provider_id: str, model_id: str) -> None:
        asyncio.ensure_future(self._apply_model_change(provider_id, model_id))

    async def _apply_model_change(self, provider_id: str, model_id: str) -> None:
        assert self.on_model_change is not None
        label = f"{provider_id}/{model_id}"
        try:
            new_provider = await self.on_model_change(provider_id, model_id)
        except Exception as exc:  # noqa: BLE001 - surface any provider-build failure to the user
            self.transcript.add(Text(f"Failed to switch to {label}: {exc}", style="red"))
        else:
            self.agent.provider = new_provider
            self.model_label = label
            self.footer.set_status(self._status_line())
            self.transcript.add(Text(f"Switched to {label}.", style="green"))
        self.tui.request_render(force=True)


async def run_interactive(
    agent: Agent,
    *,
    use_tools: bool = True,
    system_prompt: str | None = None,
    models: list[ModelInfo] | None = None,
    on_model_change: Callable[[str, str], Awaitable[ModelProvider]] | None = None,
    model_label: str | None = None,
) -> None:
    session = InteractiveSession(
        agent,
        use_tools=use_tools,
        system_prompt=system_prompt,
        models=models,
        on_model_change=on_model_change,
        model_label=model_label,
    )
    await session.run()
