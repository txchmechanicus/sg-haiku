from __future__ import annotations

from collections.abc import Callable

from tui import Container, LineEditor, Text


class Footer(Container):
    """Fixed bottom UI region: an editable input line plus a status hint."""

    def __init__(self, *, on_submit: Callable[[str], None]) -> None:
        super().__init__()
        self.editor = LineEditor(prompt="> ", on_submit=on_submit)
        self.status = Text("Ctrl-C cancel turn · Ctrl-D exit", style="dim")
        self.add(self.editor)
        self.add(self.status)
