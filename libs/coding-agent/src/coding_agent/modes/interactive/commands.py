from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tui import SelectItem, SelectList, Text

if TYPE_CHECKING:
    from coding_agent.modes.interactive.session import InteractiveSession


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[[InteractiveSession, str], Awaitable[None]]


async def _cmd_help(session: InteractiveSession, _args: str) -> None:
    lines = ["Available commands:"]
    for command in sorted(COMMANDS.values(), key=lambda c: c.name):
        lines.append(f"  /{command.name} — {command.description}")
    session.transcript.add(Text("\n".join(lines), style="dim"))
    session.tui.request_render(force=True)


async def _cmd_quit(session: InteractiveSession, _args: str) -> None:
    session.quit_requested = True


async def _cmd_clear(session: InteractiveSession, _args: str) -> None:
    session.transcript.clear()
    session.messages = []
    session.tui.request_render(force=True)


async def _cmd_model(session: InteractiveSession, _args: str) -> None:
    if not session.models or session.on_model_change is None:
        session.transcript.add(Text("Model switching is not available.", style="red"))
        session.tui.request_render(force=True)
        return

    items = [
        SelectItem(
            id=f"{model.provider}/{model.id}",
            label=f"{model.provider}/{model.id}",
            description=model.name,
        )
        for model in session.models
    ]

    previous_focus = session.tui.focused
    select_list = SelectList(items)

    def on_select(item: SelectItem) -> None:
        session.tui.remove(select_list)
        session.tui.set_focus(previous_focus)
        provider_id, model_id = item.id.split("/", 1)
        session.start_model_switch(provider_id, model_id)

    def on_cancel() -> None:
        session.tui.remove(select_list)
        session.tui.set_focus(previous_focus)
        session.tui.request_render(force=True)

    select_list.on_select = on_select
    select_list.on_cancel = on_cancel
    session.tui.add(select_list)
    session.tui.set_focus(select_list)
    session.tui.request_render(force=True)


COMMANDS: dict[str, SlashCommand] = {
    "help": SlashCommand("help", "List available commands.", _cmd_help),
    "quit": SlashCommand("quit", "Exit interactive mode.", _cmd_quit),
    "clear": SlashCommand("clear", "Clear the conversation history.", _cmd_clear),
    "model": SlashCommand("model", "Switch the active model.", _cmd_model),
}


async def dispatch(session: InteractiveSession, text: str) -> bool:
    """Handle a slash command. Returns False if `text` isn't one."""
    if not text.startswith("/"):
        return False

    parts = text[1:].split(maxsplit=1)
    if not parts:
        return True

    name = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    command = COMMANDS.get(name)
    if command is None:
        session.transcript.add(Text(f"Unknown command: /{name}", style="red"))
        session.tui.request_render(force=True)
        return True

    await command.handler(session, rest)
    return True
