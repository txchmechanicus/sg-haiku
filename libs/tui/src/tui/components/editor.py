from __future__ import annotations

from collections.abc import Callable

from tui.component import CURSOR_MARKER, Focusable
from tui.keys import parse_key


class LineEditor(Focusable):
    """A single-line text input with basic cursor movement and editing.

    Undo/redo, kill-ring, and word-navigation are deferred to a later phase —
    this covers character entry, backspace/delete, and left/right/home/end.
    """

    def __init__(
        self, *, prompt: str = "> ", on_submit: Callable[[str], None] | None = None
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.text = ""
        self.cursor = 0
        self.on_submit = on_submit

    def set_text(self, text: str) -> None:
        self.text = text
        self.cursor = len(text)

    def handle_input(self, data: str) -> None:
        key = parse_key(data.encode("utf-8"))

        if key.name == "enter":
            submitted = self.text
            self.text = ""
            self.cursor = 0
            if self.on_submit is not None:
                self.on_submit(submitted)
        elif key.name == "backspace":
            if self.cursor > 0:
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
        elif key.name == "delete":
            if self.cursor < len(self.text):
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        elif key.name == "left":
            self.cursor = max(0, self.cursor - 1)
        elif key.name == "right":
            self.cursor = min(len(self.text), self.cursor + 1)
        elif key.name in ("home", "ctrl+a"):
            self.cursor = 0
        elif key.name in ("end", "ctrl+e"):
            self.cursor = len(self.text)
        elif key.name == "ctrl+u":
            self.text = self.text[self.cursor :]
            self.cursor = 0
        elif key.name == "ctrl+k":
            self.text = self.text[: self.cursor]
        elif key.name == "char" and key.char is not None:
            self.text = self.text[: self.cursor] + key.char + self.text[self.cursor :]
            self.cursor += 1

    def render(self, width: int) -> list[str]:
        line = self.prompt + self.text
        if self.focused:
            marker_pos = len(self.prompt) + self.cursor
            line = line[:marker_pos] + CURSOR_MARKER + line[marker_pos:]
        return [line]
