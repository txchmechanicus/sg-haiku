from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static

from coding_agent.modes.interactive.components.command_hints import CommandHints


class PromptBar(Vertical):
    """Fixed bottom UI region: a slash-command hint list, an editable input line, and a
    status hint."""

    DEFAULT_CSS = """
    PromptBar {
        height: auto;
        dock: bottom;
        border-top: solid $primary-darken-2;
    }
    PromptBar > Static {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, *, status: str | None = None) -> None:
        super().__init__()
        self._initial_status = status or self._default_status()

    @staticmethod
    def _default_status() -> str:
        return "Ctrl-C cancel turn · Ctrl-D exit · /help for commands"

    def compose(self) -> ComposeResult:
        yield CommandHints()
        yield Input(placeholder="Message…")
        yield Static(self._initial_status, id="status")

    @property
    def input(self) -> Input:
        return self.query_one(Input)

    @property
    def hints(self) -> CommandHints:
        return self.query_one(CommandHints)

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)
