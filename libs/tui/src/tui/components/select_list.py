from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console

from tui.component import CURSOR_MARKER, Focusable
from tui.keys import parse_key


@dataclass
class SelectItem:
    id: str
    label: str
    description: str | None = None


class SelectList(Focusable):
    """An arrow-key-navigable list. Enter confirms, escape/ctrl+c cancels."""

    def __init__(
        self,
        items: list[SelectItem],
        *,
        on_select: Callable[[SelectItem], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.items = items
        self.index = 0
        self.on_select = on_select
        self.on_cancel = on_cancel

    def handle_input(self, data: str) -> None:
        key = parse_key(data.encode("utf-8"))

        if key.name in ("escape", "ctrl+c"):
            if self.on_cancel is not None:
                self.on_cancel()
            return

        if not self.items:
            return

        if key.name == "up":
            self.index = (self.index - 1) % len(self.items)
        elif key.name == "down":
            self.index = (self.index + 1) % len(self.items)
        elif key.name == "enter":
            if self.on_select is not None:
                self.on_select(self.items[self.index])

    def render(self, width: int) -> list[str]:
        if not self.items:
            return ["(no items)"]
        lines: list[str] = []
        for i, item in enumerate(self.items):
            selected = i == self.index
            text = item.label if not item.description else f"{item.label} — {item.description}"
            line = self._render_row(text, width, selected=selected)
            if selected and self.focused:
                line = CURSOR_MARKER + line
            lines.append(line)
        return lines

    def _render_row(self, text: str, width: int, *, selected: bool) -> str:
        prefix = "> " if selected else "  "
        full = f"{prefix}{text}"
        if width > 0 and len(full) > width:
            full = full[: max(width - 1, 0)] + "…"
        if not selected:
            return full
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=max(width, 1),
            force_terminal=True,
            color_system="standard",
            highlight=False,
            markup=False,
        )
        console.print(full, style="reverse", end="")
        return buffer.getvalue()
