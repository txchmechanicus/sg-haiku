from __future__ import annotations

from collections.abc import Callable

from tui import Container, LineEditor, Text


class Footer(Container):
    """Fixed bottom UI region: an editable input line plus a status hint."""

    def __init__(self, *, on_submit: Callable[[str], None], status: str | None = None) -> None:
        super().__init__()
        self.editor = LineEditor(prompt="> ", on_submit=on_submit)
        self.status = Text(status or self._default_status(), style="dim")
        self.add(self.editor)
        self.add(self.status)

    @staticmethod
    def _default_status() -> str:
        return "Ctrl-C cancel turn · Ctrl-D exit · /help for commands"

    def set_status(self, text: str) -> None:
        self.status.set_text(text)
