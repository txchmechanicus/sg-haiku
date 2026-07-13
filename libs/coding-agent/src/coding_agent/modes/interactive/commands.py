from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.widgets import Static
from tui import SelectItem, SelectScreen

if TYPE_CHECKING:
    from coding_agent.modes.interactive.session import HaikuApp


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[[HaikuApp, str], Awaitable[None]]
    takes_args: bool = False


async def _cmd_help(app: HaikuApp, _args: str) -> None:
    lines = ["Available commands:"]
    for command in sorted(COMMANDS.values(), key=lambda c: c.name):
        lines.append(f"  /{command.name} — {command.description}")
    app.transcript.mount(Static("\n".join(lines), classes="dim"))
    app._scroll_to_end(force=True)


async def _cmd_quit(app: HaikuApp, _args: str) -> None:
    app.quit_requested = True


async def _cmd_clear(app: HaikuApp, _args: str) -> None:
    # Matches upstream Pi's `/clear` (aliased `/new`): starts a brand-new session file
    # rather than just wiping in-memory history, so the abandoned conversation stays
    # reachable via --resume/--continue against its own session id.
    if app.new_session_factory is not None:
        app.session = app.new_session_factory()
        provider_id, _, model_id = app.model_label.partition("/")
        app.session.record_model_change(provider=provider_id, model_id=model_id)
    await app.transcript.remove_children()
    app.messages = []


async def _cmd_model(app: HaikuApp, _args: str) -> None:
    if not app.models or app.on_model_change is None:
        app.transcript.mount(Static("Model switching is not available.", classes="error"))
        app._scroll_to_end(force=True)
        return

    items = [
        SelectItem(
            id=f"{model.provider}/{model.id}",
            label=f"{model.provider}/{model.id}",
            description=model.name,
        )
        for model in app.models
    ]

    def on_dismiss(item: SelectItem | None) -> None:
        if item is None:
            return
        provider_id, model_id = item.id.split("/", 1)
        app.start_model_switch(provider_id, model_id)

    app.push_screen(SelectScreen(items), on_dismiss)


COMMANDS: dict[str, SlashCommand] = {
    "help": SlashCommand("help", "List available commands.", _cmd_help),
    "quit": SlashCommand("quit", "Exit interactive mode.", _cmd_quit),
    "clear": SlashCommand("clear", "Clear the conversation history.", _cmd_clear),
    "model": SlashCommand("model", "Switch the active model.", _cmd_model),
}


async def dispatch(app: HaikuApp, text: str) -> bool:
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
        app.transcript.mount(Static(f"Unknown command: /{name}", classes="error"))
        app._scroll_to_end(force=True)
        return True

    await command.handler(app, rest)
    return True
